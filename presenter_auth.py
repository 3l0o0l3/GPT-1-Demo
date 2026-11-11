"""Persisted allowlisted-email OTP authentication, independent of HTTP and SMTP."""
from contextlib import contextmanager, closing
from datetime import datetime, timezone
import hashlib
import hmac
import math
import os
from pathlib import Path
import re
import secrets
import sqlite3
import time

from presenter_mail import DEFAULT_ALLOWED_EMAILS, normalize_allowlist

_TOKEN = re.compile(r'[A-Za-z0-9_-]{43}\Z')
_EMAIL = re.compile(r'[^\s@\r\n]+@[^\s@\r\n]+\.[^\s@\r\n]+\Z')
OTP_SECONDS = 600
SESSION_SECONDS = 8 * 60 * 60
MAX_EVENTS = 10000
MAX_SESSIONS = 100
_GENERIC = 'If this email address is allowed, a sign-in code has been sent. Check your inbox.'


class AuthError(Exception):
    def __init__(self,status,code,message):
        super().__init__(message)
        self.status,self.code,self.message = status,code,message


def _token(value):
    return isinstance(value,str) and _TOKEN.fullmatch(value) is not None


def _digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _date(value):
    return datetime.fromtimestamp(value,timezone.utc).isoformat()


class AuthService:
    def __init__(self,db_path,key_path,mailer,allowed_email=DEFAULT_ALLOWED_EMAILS,clock=time.time):
        self.db_path,self.key_path = Path(db_path),Path(key_path)
        self.mailer,self.clock = mailer,clock
        self.allowed_emails = normalize_allowlist(allowed_email)
        self.db_path.parent.mkdir(parents=True,exist_ok=True)
        self.key_path.parent.mkdir(parents=True,exist_ok=True)
        self._key = self._load_key()
        try:
            with closing(self._connect()) as conn:
                conn.execute('PRAGMA journal_mode=WAL')
                conn.execute('PRAGMA synchronous=FULL')
                tables={row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if tables and tables != {'auth_meta','challenges','sessions','events'}:
                    raise ValueError('Unexpected authentication schema.')
                conn.executescript('''
                    CREATE TABLE IF NOT EXISTS auth_meta(version INTEGER NOT NULL,key_hash TEXT NOT NULL);
                    CREATE TABLE IF NOT EXISTS challenges(
                      id TEXT PRIMARY KEY,email TEXT NOT NULL UNIQUE,browser_hash TEXT NOT NULL,
                      code_hash TEXT NOT NULL,created REAL NOT NULL,expires REAL NOT NULL,
                      attempts INTEGER NOT NULL DEFAULT 0,ready INTEGER NOT NULL DEFAULT 0);
                    CREATE TABLE IF NOT EXISTS sessions(
                      token_hash TEXT PRIMARY KEY,email TEXT NOT NULL,
                      created REAL NOT NULL,expires REAL NOT NULL);
                    CREATE TABLE IF NOT EXISTS events(
                      kind TEXT NOT NULL,scope TEXT NOT NULL,at REAL NOT NULL);
                    CREATE INDEX IF NOT EXISTS event_lookup ON events(kind,scope,at);
                    CREATE INDEX IF NOT EXISTS session_expiry ON sessions(expires);
                ''')
                version=conn.execute('SELECT version,key_hash FROM auth_meta').fetchall()
                key_hash=hashlib.sha256(self._key).hexdigest()
                if not version:
                    conn.execute('INSERT INTO auth_meta VALUES(1,?)',(key_hash,))
                elif len(version)!=1 or version[0][0]!=1 or not hmac.compare_digest(version[0][1],key_hash):
                    raise ValueError('Unsupported authentication schema.')
                if conn.execute('PRAGMA quick_check').fetchone()[0]!='ok':
                    raise ValueError('Invalid authentication database.')
            if os.name=='posix':
                os.chmod(self.db_path,0o600)
        except (sqlite3.Error,OSError,ValueError):
            raise AuthError(503,'auth_unavailable','Presenter sign-in is temporarily unavailable.') from None

    def _load_key(self):
        try:
            if not self.key_path.exists():
                if self.db_path.exists():
                    raise ValueError('A persisted database requires its original key.')
                try:
                    descriptor=os.open(self.key_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
                except FileExistsError:
                    pass
                else:
                    with os.fdopen(descriptor,'wb') as handle:
                        handle.write(secrets.token_bytes(32));handle.flush();os.fsync(handle.fileno())
            if self.key_path.is_symlink() or (os.name=='posix' and self.key_path.stat().st_mode & 0o077):
                raise ValueError('Invalid authentication key permissions.')
            with self.key_path.open('rb') as handle:
                key=handle.read(33)
            if len(key)!=32:
                raise ValueError('Invalid authentication key.')
            return key
        except (OSError,ValueError):
            raise AuthError(503,'auth_unavailable','Presenter sign-in is temporarily unavailable.') from None

    def _connect(self):
        conn=sqlite3.connect(self.db_path,timeout=5,isolation_level=None)
        conn.row_factory=sqlite3.Row
        conn.execute('PRAGMA busy_timeout=5000')
        return conn

    @contextmanager
    def _transaction(self):
        try:
            conn=self._connect()
        except sqlite3.Error:
            raise AuthError(503,'auth_unavailable','Presenter sign-in is temporarily unavailable.') from None
        try:
            conn.execute('BEGIN IMMEDIATE')
            yield conn
            conn.commit()
        except sqlite3.Error:
            conn.rollback()
            raise AuthError(503,'auth_unavailable','Presenter sign-in is temporarily unavailable.') from None
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _now(self):
        now=self.clock()
        if type(now) not in (int,float) or not math.isfinite(now) or not 0<now<253402270000:
            raise AuthError(503,'auth_unavailable','Presenter sign-in is temporarily unavailable.')
        return now

    @staticmethod
    def _cleanup(conn,now):
        conn.execute('DELETE FROM events WHERE at<=?',(now-3600,))
        conn.execute('DELETE FROM challenges WHERE expires<=?',(now,))
        conn.execute('DELETE FROM sessions WHERE expires<=?',(now,))

    @staticmethod
    def _rate(conn,kind,scope,now,limit,window=3600,cooldown=0):
        row=conn.execute('SELECT count(*),max(at) FROM events WHERE kind=? AND scope=? AND at>?',
                         (kind,scope,now-window)).fetchone()
        if row[0]>=limit or (cooldown and row[1] is not None and now-row[1]<cooldown):
            raise AuthError(429,'rate_limit','Too many sign-in attempts. Please wait before trying again.')
        if conn.execute('SELECT count(*) FROM events').fetchone()[0]>=MAX_EVENTS:
            raise AuthError(429,'rate_limit','Sign-in is busy. Please try again later.')
        conn.execute('INSERT INTO events VALUES(?,?,?)',(kind,scope,now))

    def csrf(self,token):
        if not _token(token):
            raise AuthError(403,'csrf','Reload the sign-in page before trying again.')
        return hmac.new(self._key,('presenter-csrf:'+token).encode(),hashlib.sha256).hexdigest()

    def _code_hash(self,challenge,browser_hash,email,code):
        value='presenter-otp:'+challenge+'\0'+browser_hash+'\0'+email+'\0'+code
        return hmac.new(self._key,value.encode(),hashlib.sha256).hexdigest()

    def issue(self,email,browser_token,client_key):
        if (not isinstance(email,str) or not 1<=len(email.strip())<=254 or
                not _EMAIL.fullmatch(email.strip()) or not _token(browser_token) or
                not isinstance(client_key,str) or not 1<=len(client_key)<=256):
            raise AuthError(400,'invalid_input','Enter a valid email address and reload the sign-in page if needed.')
        email=email.strip().lower()
        now=self._now();challenge=secrets.token_urlsafe(32)
        reply=dict(challengeId=challenge,expiresIn=OTP_SECONDS,retryAfter=60,message=_GENERIC)
        allowed=email in self.allowed_emails
        code=f'{secrets.randbelow(100000000):08d}'
        browser_hash=_digest(browser_token)
        with self._transaction() as conn:
            self._cleanup(conn,now)
            self._rate(conn,'issue-global','all',now,120)
            self._rate(conn,'issue-client',_digest(client_key),now,30)
            self._rate(conn,'issue-client-minute',_digest(client_key),now,10,window=60)
            if allowed:
                self._rate(conn,'issue-recipient',_digest(email),now,6,cooldown=60)
                conn.execute('DELETE FROM challenges WHERE email=?',(email,))
                conn.execute('INSERT INTO challenges(id,email,browser_hash,code_hash,created,expires) VALUES(?,?,?,?,?,?)',
                             (challenge,email,browser_hash,self._code_hash(challenge,browser_hash,email,code),now,now+OTP_SECONDS))
        if not allowed:
            return reply
        try:
            self.mailer.send_code(email,code,expires_minutes=10)
        except Exception:
            with self._transaction() as conn:
                conn.execute('DELETE FROM challenges WHERE id=?',(challenge,))
            raise AuthError(503,'mail_unavailable','The sign-in email could not be sent. Please try again later.') from None
        with self._transaction() as conn:
            cursor=conn.execute('UPDATE challenges SET ready=1 WHERE id=? AND expires>?',(challenge,self._now()))
            if cursor.rowcount!=1:
                raise AuthError(503,'mail_unavailable','This sign-in request expired. Please request a new code.')
        return reply

    def verify(self,challenge_id,code,browser_token):
        # Invalid inputs still consume the global verification budget; they never
        # reach the mailer or allocate arbitrary challenge rows.
        now=self._now();failure=None;result=None
        with self._transaction() as conn:
            self._cleanup(conn,now)
            self._rate(conn,'verify-global','all',now,300)
            row=(conn.execute('SELECT * FROM challenges WHERE id=?',(challenge_id,)).fetchone()
                 if _token(challenge_id) else None)
            bound=bool(row and _token(browser_token) and hmac.compare_digest(row['browser_hash'],_digest(browser_token)))
            if not row or not bound or row['email'] not in self.allowed_emails or not row['ready'] or row['expires']<=now:
                failure=AuthError(401,'invalid_code','The code is invalid or expired. Request a new code in this browser.')
            else:
                self._rate(conn,'verify-recipient',_digest(row['email']),now,30)
                if row['attempts']>=5:
                    failure=AuthError(429,'rate_limit','Too many attempts for this code. Please request a new code later.')
                else:
                    valid=(isinstance(code,str) and re.fullmatch(r'[0-9]{8}',code) is not None and
                           hmac.compare_digest(row['code_hash'],self._code_hash(challenge_id,row['browser_hash'],row['email'],code)))
                    if not valid:
                        conn.execute('UPDATE challenges SET attempts=attempts+1 WHERE id=?',(challenge_id,))
                        failure=(AuthError(429,'rate_limit','Too many attempts for this code. Please request a new code later.')
                                 if row['attempts']+1>=5 else
                                 AuthError(401,'invalid_code','The code is invalid or expired. Check the most recent email.'))
                    elif conn.execute('SELECT count(*) FROM sessions').fetchone()[0]>=MAX_SESSIONS:
                        failure=AuthError(429,'rate_limit','Too many active sessions. Sign out of an existing session or try later.')
                    else:
                        token=secrets.token_urlsafe(32)
                        conn.execute('DELETE FROM challenges WHERE id=?',(challenge_id,))
                        conn.execute('INSERT INTO sessions VALUES(?,?,?,?)',(_digest(token),row['email'],now,now+SESSION_SECONDS))
                        result=dict(sessionToken=token,email=row['email'],expiresAt=_date(now+SESSION_SECONDS),csrfToken=self.csrf(token))
        if failure:
            raise failure
        return result

    def require(self,session_token,csrf=None):
        if not _token(session_token):
            raise AuthError(401,'login_required','Sign in to open the presenter page.')
        try:
            conn=self._connect()
            try:
                row=conn.execute('SELECT email,expires FROM sessions WHERE token_hash=?',(_digest(session_token),)).fetchone()
            finally:
                conn.close()
        except sqlite3.Error:
            raise AuthError(503,'auth_unavailable','Presenter sign-in is temporarily unavailable.') from None
        if not row or row['email'] not in self.allowed_emails or row['expires']<=self._now():
            raise AuthError(401,'login_required','Your presenter session has ended. Sign in again.')
        expected=self.csrf(session_token)
        if csrf is not None and (not isinstance(csrf,str) or not re.fullmatch('[a-f0-9]{64}',csrf) or not hmac.compare_digest(csrf,expected)):
            raise AuthError(403,'csrf','Reload the presenter page before trying again.')
        return dict(email=row['email'],expiresAt=_date(row['expires']),csrfToken=expected)

    def logout(self,session_token):
        if _token(session_token):
            with self._transaction() as conn:
                conn.execute('DELETE FROM sessions WHERE token_hash=?',(_digest(session_token),))
