"""Authentication lifecycle and abuse budgets with an injected clock/mailer."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
import secrets
import hashlib
import hmac
import sqlite3
import tempfile
import threading
import unittest
from unittest.mock import patch

from presenter_auth import AuthError,AuthService,MAX_EVENTS
from presenter_mail import DEFAULT_ALLOWED_EMAIL

SECOND_EMAIL='collaborator@example.com'


class FakeMailer:
    configured=True
    def __init__(self):
        self.sent=[];self.fail=False;self.wait=None
    def send_code(self,email,code,expires_minutes=10):
        self.sent.append((email,code,expires_minutes))
        if self.wait:
            self.wait()
        if self.fail:
            raise OSError('fixture SMTP password must never escape')


class AuthenticationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.db=self.root/'auth.sqlite';self.key=self.root/'auth.key'
        self.now=1800000000.0;self.mailer=FakeMailer()
        self.auth=AuthService(self.db,self.key,self.mailer,clock=lambda:self.now)
        self.browser=secrets.token_urlsafe(32)
    def tearDown(self):
        self.temp.cleanup()
    def error(self,status,code,callback,*args):
        with self.assertRaises(AuthError) as caught:
            callback(*args)
        self.assertEqual((caught.exception.status,caught.exception.code),(status,code))
        return caught.exception
    def issue(self,email=DEFAULT_ALLOWED_EMAIL,client='fixture-client'):
        return self.auth.issue(email,self.browser,client)
    def login(self):
        challenge=self.issue()
        return self.auth.verify(challenge['challengeId'],self.mailer.sent[-1][1],self.browser)

    def test_normalization_exact_allowlist_and_generic_rejected_addresses(self):
        accepted=self.issue('  PRESENTER@EXAMPLE.COM  ')
        self.assertEqual(self.mailer.sent[0][0],DEFAULT_ALLOWED_EMAIL)
        for i,email in enumerate(('someone@example.org','presenter+alias@example.com','presenter@example.com.evil.invalid','用户@example.org')):
            rejected=self.issue(email,client=f'other-{i}')
            self.assertEqual(set(accepted),set(rejected))
            for field in ('expiresIn','retryAfter','message'):
                self.assertEqual(accepted[field],rejected[field])
            self.error(401,'invalid_code',self.auth.verify,rejected['challengeId'],self.mailer.sent[0][1],self.browser)
        self.assertEqual(len(self.mailer.sent),1)

    def test_code_is_bound_to_browser_single_use_session_csrf_and_logout(self):
        challenge=self.issue();code=self.mailer.sent[-1][1]
        self.error(401,'invalid_code',self.auth.verify,challenge['challengeId'],code,secrets.token_urlsafe(32))
        session=self.auth.verify(challenge['challengeId'],code,self.browser)
        self.assertEqual(self.auth.require(session['sessionToken'],session['csrfToken'])['email'],DEFAULT_ALLOWED_EMAIL)
        self.error(403,'csrf',self.auth.require,session['sessionToken'],'not-the-csrf')
        self.error(403,'csrf',self.auth.require,session['sessionToken'],'非ASCII')
        self.error(401,'invalid_code',self.auth.verify,challenge['challengeId'],code,self.browser)
        self.auth.logout(session['sessionToken']);self.auth.logout(session['sessionToken'])
        self.error(401,'login_required',self.auth.require,session['sessionToken'])

    def test_two_accounts_keep_independent_challenges_codes_and_sessions(self):
        other_browser=secrets.token_urlsafe(32)
        with patch('presenter_auth.secrets.randbelow',side_effect=[12345678,87654321]):
            first=self.issue()
            second=self.auth.issue(' COLLABORATOR@EXAMPLE.COM ',other_browser,'second-client')
        self.assertEqual([m[0] for m in self.mailer.sent],[DEFAULT_ALLOWED_EMAIL,SECOND_EMAIL])
        self.error(401,'invalid_code',self.auth.verify,second['challengeId'],'12345678',other_browser)
        self.error(401,'invalid_code',self.auth.verify,first['challengeId'],'87654321',self.browser)
        self.error(401,'invalid_code',self.auth.verify,second['challengeId'],'87654321',self.browser)
        first_session=self.auth.verify(first['challengeId'],'12345678',self.browser)
        second_session=self.auth.verify(second['challengeId'],'87654321',other_browser)
        self.assertEqual(first_session['email'],DEFAULT_ALLOWED_EMAIL)
        self.assertEqual(second_session['email'],SECOND_EMAIL)
        restarted=AuthService(self.db,self.key,self.mailer,clock=lambda:self.now)
        for expected,session in ((DEFAULT_ALLOWED_EMAIL,first_session),(SECOND_EMAIL,second_session)):
            self.assertEqual(restarted.require(session['sessionToken'],session['csrfToken'])['email'],expected)
        self.auth.logout(second_session['sessionToken'])
        self.assertEqual(self.auth.require(first_session['sessionToken'])['email'],DEFAULT_ALLOWED_EMAIL)

    def test_single_address_override_denies_removed_account_session_and_pending_code(self):
        owner=self.login()
        second=self.issue(SECOND_EMAIL)
        second_session=self.auth.verify(second['challengeId'],self.mailer.sent[-1][1],self.browser)
        self.now+=60
        pending=self.issue(SECOND_EMAIL);pending_code=self.mailer.sent[-1][1]
        restricted=AuthService(self.db,self.key,self.mailer,allowed_email=DEFAULT_ALLOWED_EMAIL,clock=lambda:self.now)
        self.assertEqual(restricted.require(owner['sessionToken'])['email'],DEFAULT_ALLOWED_EMAIL)
        self.error(401,'login_required',restricted.require,second_session['sessionToken'])
        self.error(401,'invalid_code',restricted.verify,pending['challengeId'],pending_code,self.browser)
        before=len(self.mailer.sent)
        rejected=restricted.issue(SECOND_EMAIL,self.browser,'restricted-client')
        self.assertEqual(len(self.mailer.sent),before)
        self.error(401,'invalid_code',restricted.verify,rejected['challengeId'],pending_code,self.browser)

    def test_original_single_email_digest_remains_verifiable_after_upgrade(self):
        challenge=secrets.token_urlsafe(32);code='23456789'
        browser_hash=hashlib.sha256(self.browser.encode()).hexdigest()
        # Exact digest format of the deployed single-email predecessor.
        value='presenter-otp:'+challenge+'\0'+browser_hash+'\0'+DEFAULT_ALLOWED_EMAIL+'\0'+code
        old_digest=hmac.new(self.key.read_bytes(),value.encode(),hashlib.sha256).hexdigest()
        with closing(sqlite3.connect(self.db,isolation_level=None)) as conn:
            conn.execute('INSERT INTO challenges(id,email,browser_hash,code_hash,created,expires,ready) VALUES(?,?,?,?,?,?,1)',
                         (challenge,DEFAULT_ALLOWED_EMAIL,browser_hash,old_digest,self.now,self.now+600))
        session=self.auth.verify(challenge,code,self.browser)
        self.assertEqual(self.auth.require(session['sessionToken'])['email'],DEFAULT_ALLOWED_EMAIL)

    def test_malformed_allowlists_fail_before_creating_state(self):
        for index,value in enumerate((None,[],{},True,['bad'],[DEFAULT_ALLOWED_EMAIL,None],['x@example.org']*33)):
            db=self.root/f'rejected-{index}.sqlite';key=self.root/f'rejected-{index}.key'
            with self.assertRaises(ValueError):AuthService(db,key,self.mailer,allowed_email=value)
            self.assertFalse(db.exists());self.assertFalse(key.exists())

    def test_exact_code_expiry_and_session_expiry(self):
        challenge=self.issue();code=self.mailer.sent[-1][1]
        self.now+=600
        self.error(401,'invalid_code',self.auth.verify,challenge['challengeId'],code,self.browser)
        session=self.login();self.now+=8*3600-0.01
        self.auth.require(session['sessionToken'])
        self.now+=0.01
        self.error(401,'login_required',self.auth.require,session['sessionToken'])

    def test_new_code_invalidates_previous_code(self):
        previous=self.issue();old_code=self.mailer.sent[-1][1]
        self.now+=60
        current=self.issue();new_code=self.mailer.sent[-1][1]
        self.error(401,'invalid_code',self.auth.verify,previous['challengeId'],old_code,self.browser)
        self.auth.verify(current['challengeId'],new_code,self.browser)

    def test_five_attempt_cap_and_persisted_cooldown(self):
        challenge=self.issue();code=self.mailer.sent[-1][1]
        for _ in range(4):
            self.error(401,'invalid_code',self.auth.verify,challenge['challengeId'],'x',self.browser)
        self.error(429,'rate_limit',self.auth.verify,challenge['challengeId'],'x',self.browser)
        self.error(429,'rate_limit',self.auth.verify,challenge['challengeId'],code,self.browser)
        other=AuthService(self.db,self.key,self.mailer,clock=lambda:self.now)
        self.error(429,'rate_limit',other.issue,DEFAULT_ALLOWED_EMAIL,self.browser,'changed-ip')

    def test_hourly_recipient_limit_cannot_be_evaded_by_new_browser_or_ip(self):
        for i in range(6):
            self.auth.issue(DEFAULT_ALLOWED_EMAIL,secrets.token_urlsafe(32),f'client-{i}')
            self.now+=60
        self.error(429,'rate_limit',self.auth.issue,DEFAULT_ALLOWED_EMAIL,secrets.token_urlsafe(32),'new-client')
        second=self.issue(SECOND_EMAIL,client='second-client')
        session=self.auth.verify(second['challengeId'],self.mailer.sent[-1][1],self.browser)
        self.assertEqual(session['email'],SECOND_EMAIL)
        self.now+=3600
        self.issue()

    def test_account_verification_limit_survives_challenge_rotation(self):
        for i in range(6):
            challenge=self.issue(client=f'client-{i}')
            for attempt in range(5):
                self.error(401 if attempt<4 else 429,'invalid_code' if attempt<4 else 'rate_limit',self.auth.verify,
                           challenge['challengeId'],'not-a-code',self.browser)
            self.now+=60
        # This has the still-valid sixth challenge and correct browser, but even
        # knowing its real code cannot evade the recipient-wide thirty-attempt cap.
        self.error(429,'rate_limit',self.auth.verify,challenge['challengeId'],self.mailer.sent[-1][1],self.browser)
        second=self.issue(SECOND_EMAIL,client='second-client')
        self.assertEqual(self.auth.verify(second['challengeId'],self.mailer.sent[-1][1],self.browser)['email'],SECOND_EMAIL)

    def test_smtp_failure_consumes_send_budget_but_leaves_no_usable_code(self):
        self.mailer.fail=True
        error=self.error(503,'mail_unavailable',self.issue)
        self.assertNotIn('password',str(error));self.assertNotIn('fixture',str(error))
        with closing(sqlite3.connect(self.db,isolation_level=None)) as conn:
            self.assertEqual(conn.execute('SELECT count(*) FROM challenges').fetchone()[0],0)
        self.mailer.fail=False
        self.error(429,'rate_limit',self.issue)
        self.now+=60;self.login()

    def test_pending_delivery_is_not_verifiable_and_no_transaction_spans_smtp(self):
        entered=threading.Event();release=threading.Event()
        self.mailer.wait=lambda:(entered.set(),release.wait(3))
        with ThreadPoolExecutor(max_workers=1) as pool:
            future=pool.submit(self.issue)
            self.assertTrue(entered.wait(2))
            with closing(sqlite3.connect(self.db,isolation_level=None)) as conn:
                pending=conn.execute('SELECT id FROM challenges').fetchone()[0]
            self.error(401,'invalid_code',self.auth.verify,pending,self.mailer.sent[-1][1],self.browser)
            release.set();challenge=future.result(timeout=3)
        self.auth.verify(challenge['challengeId'],self.mailer.sent[-1][1],self.browser)

    def test_concurrent_verification_consumes_once_across_instances(self):
        challenge=self.issue();code=self.mailer.sent[-1][1]
        second=AuthService(self.db,self.key,self.mailer,clock=lambda:self.now)
        barrier=threading.Barrier(8)
        def work(index):
            barrier.wait()
            try:return (self.auth if index%2 else second).verify(challenge['challengeId'],code,self.browser)
            except AuthError as error:return error.status
        with ThreadPoolExecutor(max_workers=8) as pool:
            results=list(pool.map(work,range(8)))
        self.assertEqual(sum(isinstance(r,dict) for r in results),1)
        self.assertEqual(results.count(401),7)

    def test_restart_preserves_session_and_state_contains_no_raw_secrets(self):
        with patch('presenter_auth.secrets.randbelow',return_value=12345678):
            challenge=self.issue()
        with closing(sqlite3.connect(self.db,isolation_level=None)) as conn:
            stored='\n'.join(conn.iterdump())
        self.assertNotIn('12345678',stored);self.assertNotIn(self.browser,stored)
        session=self.auth.verify(challenge['challengeId'],'12345678',self.browser)
        with closing(sqlite3.connect(self.db,isolation_level=None)) as conn:
            stored='\n'.join(conn.iterdump())
        self.assertNotIn(session['sessionToken'],stored)
        restored=AuthService(self.db,self.key,self.mailer,clock=lambda:self.now)
        self.assertEqual(restored.require(session['sessionToken'])['csrfToken'],session['csrfToken'])

    def test_malformed_or_replaced_or_missing_key_fails_closed(self):
        original=self.key.read_bytes()
        for value in (b'bad',b'x'*32):
            self.key.write_bytes(value)
            self.error(503,'auth_unavailable',AuthService,self.db,self.key,self.mailer)
        self.key.write_bytes(original);self.key.unlink()
        self.error(503,'auth_unavailable',AuthService,self.db,self.key,self.mailer)
        self.assertFalse(self.key.exists())

    def test_database_capacity_and_client_rate_prevent_unbounded_growth(self):
        for _ in range(10):self.issue('other@example.org')
        self.error(429,'rate_limit',self.issue,'other@example.org')
        with closing(sqlite3.connect(self.db,isolation_level=None)) as conn:
            conn.execute('BEGIN IMMEDIATE')
            conn.executemany('INSERT INTO events VALUES(?,?,?)',[('fixture','bounded',self.now)]*MAX_EVENTS)
            conn.commit()
        self.error(429,'rate_limit',self.issue,'other@example.org','different-client')
        self.now+=3601
        self.issue()
        with closing(sqlite3.connect(self.db,isolation_level=None)) as conn:
            self.assertLess(conn.execute('SELECT count(*) FROM events').fetchone()[0],10)

    def test_invalid_inputs_do_not_send_or_create_authentication(self):
        for email in ('','no-domain','x@example.org\r\nBcc: x@evil.invalid',None,True,[],'a'*255+'@example.org'):
            self.error(400,'invalid_input',self.issue,email)
        self.error(400,'invalid_input',self.auth.issue,DEFAULT_ALLOWED_EMAIL,'bad-browser','client')
        for token in (None,'','x',True,[],'a'*100):self.error(401,'login_required',self.auth.require,token)
        self.assertEqual(self.mailer.sent,[])


if __name__=='__main__':unittest.main()
