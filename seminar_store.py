"""Server-owned anonymous conversations and explicit, immutable submissions."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import secrets
import sqlite3
import time
import uuid


MAX_SAFE_INTEGER = 9007199254740991


class SeminarError(Exception):
    def __init__(self, status, code, message):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def timestamp(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


class SeminarStore:
    def __init__(self, path, clock=time.time):
        self.path, self.clock = str(path), clock
        with self.connection() as db:
            db.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS sessions (
                    digest TEXT PRIMARY KEY, current_id TEXT NOT NULL, touched REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, touched REAL NOT NULL,
                    pending TEXT, submitted_count INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS messages (
                    seq INTEGER PRIMARY KEY, conversation TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS messages_conversation ON messages(conversation,seq);
                CREATE TABLE IF NOT EXISTS requests (
                    owner TEXT NOT NULL, request_id TEXT NOT NULL, conversation TEXT NOT NULL,
                    status TEXT NOT NULL, PRIMARY KEY(owner,request_id));
                CREATE TABLE IF NOT EXISTS submissions (
                    cursor INTEGER PRIMARY KEY, id TEXT NOT NULL UNIQUE,
                    conversation TEXT NOT NULL, version INTEGER NOT NULL,
                    submitted_at TEXT NOT NULL, nickname TEXT NOT NULL, comment TEXT NOT NULL,
                    messages TEXT NOT NULL, model TEXT NOT NULL, UNIQUE(conversation,version));
                CREATE TABLE IF NOT EXISTS inbox_metadata (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                    last_cursor INTEGER NOT NULL CHECK(last_cursor>=0),
                    cleared_through INTEGER NOT NULL CHECK(cleared_through>=0 AND cleared_through<=last_cursor));
                INSERT OR IGNORE INTO inbox_metadata(singleton,last_cursor,cleared_through)
                    SELECT 1,COALESCE(MAX(cursor),0),0 FROM submissions;
            ''')
            # Preserve cursors across a full clear and migrate older databases
            # without changing any submitted copy. Also absorb legacy inserts.
            db.execute('UPDATE inbox_metadata SET last_cursor=MAX(last_cursor,(SELECT COALESCE(MAX(cursor),0) FROM submissions)) WHERE singleton=1')
            # A new process cannot continue requests from a previous process.
            db.execute('UPDATE conversations SET pending=NULL')
            db.execute("UPDATE requests SET status='failed' WHERE status='pending'")

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def digest(token):
        return hashlib.sha256(token.encode()).hexdigest()

    def _prune(self, db):
        cutoff = self.clock() - 86400
        # Submissions own their copied transcript, so expired conversations can go.
        db.execute('DELETE FROM messages WHERE conversation IN (SELECT id FROM conversations WHERE touched<? AND pending IS NULL)', (cutoff,))
        db.execute('DELETE FROM requests WHERE conversation IN (SELECT id FROM conversations WHERE touched<? AND pending IS NULL)', (cutoff,))
        db.execute('DELETE FROM conversations WHERE touched<? AND pending IS NULL', (cutoff,))
        db.execute('DELETE FROM sessions WHERE touched<?', (cutoff,))

    def _session(self, db, token):
        owner = self.digest(token)
        row = db.execute('SELECT * FROM sessions WHERE digest=? AND touched>=?', (owner, self.clock()-86400)).fetchone()
        if row is None:
            raise SeminarError(401, 'session_expired', 'Your anonymous session expired. Reload to start again.')
        return owner, row['current_id']

    def _conversation(self, db, conversation):
        row = db.execute('SELECT * FROM conversations WHERE id=?', (conversation,)).fetchone()
        if row is None:
            raise SeminarError(401, 'session_expired', 'This conversation expired. Start a new conversation.')
        messages = [json.loads(r['payload']) for r in db.execute('SELECT payload FROM messages WHERE conversation=? ORDER BY seq', (conversation,))]
        return {'id': conversation, 'messages': messages, 'submittedMessageCount': row['submitted_count']}

    def bootstrap(self, token=None):
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            self._prune(db)
            if token:
                try:
                    owner, current = self._session(db, token)
                    result = self._conversation(db, current)
                    db.execute('UPDATE sessions SET touched=? WHERE digest=?', (self.clock(), owner))
                    return token, result
                except SeminarError:
                    pass
            if (db.execute('SELECT count(*) FROM sessions').fetchone()[0] >= 5000 or
                    db.execute('SELECT count(*) FROM conversations').fetchone()[0] >= 20000):
                raise SeminarError(429, 'capacity', 'The seminar is at capacity. Please try later.')
            token, conversation = secrets.token_urlsafe(32), uuid.uuid4().hex
            owner = self.digest(token)
            db.execute('INSERT INTO sessions VALUES (?,?,?)', (owner, conversation, self.clock()))
            db.execute('INSERT INTO conversations(id,owner,touched) VALUES (?,?,?)', (conversation, owner, self.clock()))
            return token, self._conversation(db, conversation)

    def current(self, token):
        with self.connection() as db:
            _, conversation = self._session(db, token)
            return self._conversation(db, conversation)

    def new(self, token):
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            owner, old = self._session(db, token)
            row = db.execute('SELECT pending FROM conversations WHERE id=?', (old,)).fetchone()
            if row and row['pending']:
                raise SeminarError(409, 'pending', 'Wait for the current reply before starting again.')
            if (db.execute('SELECT count(*) FROM conversations WHERE owner=?', (owner,)).fetchone()[0] >= 50 or
                    db.execute('SELECT count(*) FROM conversations').fetchone()[0] >= 20000):
                raise SeminarError(429, 'capacity', 'The conversation limit has been reached. Please try again tomorrow.')
            current = uuid.uuid4().hex
            db.execute('INSERT INTO conversations(id,owner,touched) VALUES (?,?,?)', (current, owner, self.clock()))
            db.execute('UPDATE sessions SET current_id=?,touched=? WHERE digest=?', (current, self.clock(), owner))
            return self._conversation(db, current)

    def begin_turn(self, token, request_id, content):
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            owner, current = self._session(db, token)
            conversation = self._conversation(db, current)
            if len(conversation['messages']) >= 20:
                raise SeminarError(409, 'turn_limit', 'This conversation has 10 replies. Submit it or start a new one.')
            if db.execute('SELECT pending FROM conversations WHERE id=?', (current,)).fetchone()['pending']:
                raise SeminarError(409, 'pending', 'A reply for this visitor is already running.')
            if db.execute('SELECT 1 FROM requests WHERE owner=? AND request_id=?', (owner, request_id)).fetchone():
                raise SeminarError(409, 'duplicate_request', 'This request was already handled. Reload the conversation to check its result.')
            if (db.execute('SELECT count(*) FROM requests WHERE owner=?', (owner,)).fetchone()[0] >= 1000 or
                    db.execute('SELECT count(*) FROM requests').fetchone()[0] >= 100000):
                raise SeminarError(429, 'capacity', 'The seminar request limit has been reached. Please try later.')
            db.execute('INSERT INTO requests VALUES (?,?,?,?)', (owner, request_id, current, 'pending'))
            db.execute('UPDATE conversations SET pending=?,touched=? WHERE id=?', (request_id, self.clock(), current))
            db.execute('UPDATE sessions SET touched=? WHERE digest=?', (self.clock(), owner))
            user = {'id': uuid.uuid4().hex, 'role': 'user', 'content': content, 'createdAt': timestamp(self.clock())}
            return {'owner': owner, 'conversationId': current, 'requestId': request_id,
                    'messages': conversation['messages'], 'userMessage': user}

    def complete_turn(self, turn, content, model, stop_reason, token_count, classification=None):
        assistant = {'id': uuid.uuid4().hex, 'role': 'assistant', 'content': content,
                     'createdAt': timestamp(self.clock()), 'model': model,
                     'stopReason': stop_reason, 'tokenCount': token_count}
        if classification is not None:
            assistant.update(kind='classification', classification=classification)
            turn['userMessage'] = dict(turn['userMessage'], kind='classification')
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT pending FROM conversations WHERE id=? AND owner=?', (turn['conversationId'], turn['owner'])).fetchone()
            if row is None or row['pending'] != turn['requestId']:
                raise SeminarError(409, 'stale_request', 'This generation no longer belongs to the current conversation.')
            for message in (turn['userMessage'], assistant):
                db.execute('INSERT INTO messages(conversation,payload) VALUES (?,?)', (turn['conversationId'], json.dumps(message, ensure_ascii=False)))
            db.execute('UPDATE conversations SET pending=NULL,touched=? WHERE id=?', (self.clock(), turn['conversationId']))
            db.execute("UPDATE requests SET status='done' WHERE owner=? AND request_id=?", (turn['owner'], turn['requestId']))
            return self._conversation(db, turn['conversationId']), assistant

    def abort_turn(self, turn):
        with self.connection() as db:
            db.execute('UPDATE conversations SET pending=NULL WHERE id=? AND pending=?', (turn['conversationId'], turn['requestId']))
            db.execute("UPDATE requests SET status='failed' WHERE owner=? AND request_id=? AND status='pending'", (turn['owner'], turn['requestId']))

    def submit(self, token, nickname, comment, model, expected_conversation, expected_count):
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            _, current = self._session(db, token)
            if current != expected_conversation:
                raise SeminarError(409, 'stale_conversation', 'This conversation changed in another tab. Recover it, review the messages, and choose to share again.')
            if db.execute('SELECT pending FROM conversations WHERE id=?', (current,)).fetchone()['pending']:
                raise SeminarError(409, 'pending', 'Wait for the reply before submitting this conversation.')
            conversation = self._conversation(db, current)
            version = len(conversation['messages'])
            if version != expected_count:
                raise SeminarError(409, 'stale_conversation', 'New messages arrived in another tab. Recover the conversation and choose to share again.')
            if version < 2:
                raise SeminarError(409, 'empty_conversation', 'Generate at least one reply before submitting.')
            existing = db.execute('SELECT id,submitted_at FROM submissions WHERE conversation=? AND version=?', (current, version)).fetchone()
            if existing:
                return {'submission': {'id': existing['id'], 'submittedAt': existing['submitted_at'], 'messageCount': version}, 'alreadySubmitted': True}
            if version <= conversation['submittedMessageCount']:
                raise SeminarError(409, 'submission_cleared', 'The presenter cleared this submitted version. Continue the conversation before choosing to share a new version.')
            if db.execute('SELECT count(*) FROM submissions').fetchone()[0] >= 10000:
                raise SeminarError(429, 'capacity', 'The discussion inbox is full. Please tell the presenter.')
            last_cursor = db.execute('SELECT last_cursor FROM inbox_metadata WHERE singleton=1').fetchone()[0]
            if last_cursor >= MAX_SAFE_INTEGER:
                raise SeminarError(429, 'capacity', 'The discussion inbox cursor limit has been reached.')
            cursor = last_cursor + 1
            identifier, submitted = uuid.uuid4().hex, timestamp(self.clock())
            # Preserve the provenance of the reply, even after a later deployment.
            submitted_model = next((m['model'] for m in reversed(conversation['messages'])
                                    if m['role'] == 'assistant' and 'model' in m), model)
            db.execute('INSERT INTO submissions(cursor,id,conversation,version,submitted_at,nickname,comment,messages,model) VALUES (?,?,?,?,?,?,?,?,?)',
                       (cursor, identifier, current, version, submitted, nickname, comment, json.dumps(conversation['messages'], ensure_ascii=False), json.dumps(submitted_model)))
            db.execute('UPDATE inbox_metadata SET last_cursor=? WHERE singleton=1', (cursor,))
            db.execute('UPDATE conversations SET submitted_count=?,touched=? WHERE id=?', (version, self.clock(), current))
            return {'submission': {'id': identifier, 'submittedAt': submitted, 'messageCount': version}, 'alreadySubmitted': False}

    def submissions(self, after=0, limit=100):
        with self.connection() as db:
            # Rows, count and the clear watermark belong to one read snapshot.
            # A concurrent clear cannot leave a page with mismatched metadata.
            db.execute('BEGIN')
            status = self._inbox_status(db)
            rows = db.execute('SELECT * FROM submissions WHERE cursor>? ORDER BY cursor LIMIT ?', (after, limit+1)).fetchall()
            entries = [{'cursor': r['cursor'], 'id': r['id'], 'submittedAt': r['submitted_at'],
                        'nickname': r['nickname'], 'comment': r['comment'],
                        'messages': json.loads(r['messages']), 'model': json.loads(r['model'])} for r in rows[:limit]]
            return dict(status, submissions=entries, nextCursor=entries[-1]['cursor'] if entries else after, hasMore=len(rows)>limit)

    @staticmethod
    def _inbox_status(db):
        row = db.execute('SELECT last_cursor,cleared_through,(SELECT count(*) FROM submissions) AS total FROM inbox_metadata WHERE singleton=1').fetchone()
        return {'submissionCount': row['total'], 'lastCursor': row['last_cursor'], 'clearedThrough': row['cleared_through']}

    def inbox_status(self):
        with self.connection() as db:
            return self._inbox_status(db)

    def clear_submissions(self, through_cursor):
        if type(through_cursor) is not int or not 0 <= through_cursor <= MAX_SAFE_INTEGER:
            raise SeminarError(400, 'invalid_input', 'Choose a valid confirmed submission cursor.')
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            status = self._inbox_status(db)
            if through_cursor > status['lastCursor']:
                raise SeminarError(400, 'invalid_input', 'The confirmed cursor has not been issued. Refresh the inbox before clearing.')
            deleted = db.execute('DELETE FROM submissions WHERE cursor<=?', (through_cursor,)).rowcount
            db.execute('UPDATE inbox_metadata SET cleared_through=MAX(cleared_through,?) WHERE singleton=1', (through_cursor,))
            return dict(self._inbox_status(db), deletedCount=deleted)

    def count(self):
        with self.connection() as db:
            return db.execute('SELECT count(*) FROM submissions').fetchone()[0]
