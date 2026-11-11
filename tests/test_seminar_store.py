"""Private persistence behavior using temporary SQLite files only."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import tempfile
import threading
import unittest
import uuid
from unittest.mock import patch

from seminar_store import SeminarError, SeminarStore


MODEL = {'snapshotId': 'fixture-snapshot', 'step': 7}


class SeminarStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.now = 1000000
        self.path = Path(self.temp.name) / 'seminar.sqlite'
        self.store = SeminarStore(self.path, clock=lambda: self.now)
        self.a, self.ca = self.store.bootstrap()
        self.b, self.cb = self.store.bootstrap()

    def tearDown(self):
        self.temp.cleanup()

    def pair(self, token, text='hello', reply='actual fixture reply', request=None):
        turn = self.store.begin_turn(token, request or str(uuid.uuid4()), text)
        return self.store.complete_turn(turn, reply, MODEL, 'eos', 3)

    def submit(self, token, nickname='', comment='', model=MODEL, expected=None):
        conversation = expected if expected is not None else self.store.current(token)
        return self.store.submit(token, nickname, comment, model,
                                 conversation['id'], len(conversation['messages']))

    def test_isolation_server_owned_history_and_atomic_completion(self):
        identifier = str(uuid.uuid4())
        turn = self.store.begin_turn(self.a, identifier, 'A private prompt')
        self.assertEqual(self.store.current(self.a)['messages'], [])
        self.assertEqual(self.store.current(self.b)['messages'], [])
        completed, assistant = self.store.complete_turn(turn, 'A reply', MODEL, 'eos', 2)
        self.assertEqual([m['content'] for m in completed['messages']], ['A private prompt', 'A reply'])
        self.assertEqual(completed['messages'][1], assistant)
        self.assertEqual(self.store.current(self.b)['messages'], [])
        # Request IDs are scoped to the visitor rather than globally colliding.
        other, _ = self.pair(self.b, request=identifier)
        self.assertNotEqual(completed['id'], other['id'])
        with self.assertRaises(SeminarError) as caught:
            self.store.begin_turn(self.a, identifier, 'replay')
        self.assertEqual(caught.exception.code, 'duplicate_request')

    def test_pending_rejects_new_submit_and_second_turn_then_abort_releases(self):
        turn = self.store.begin_turn(self.a, str(uuid.uuid4()), 'hello')
        for action in (lambda: self.store.new(self.a),
                       lambda: self.submit(self.a, '', '', MODEL),
                       lambda: self.store.begin_turn(self.a, str(uuid.uuid4()), 'another')):
            with self.assertRaises(SeminarError) as caught:
                action()
            self.assertEqual(caught.exception.code, 'pending')
        self.store.abort_turn(turn)
        self.assertEqual(self.store.current(self.a)['messages'], [])
        self.pair(self.a)
        self.assertEqual(len(self.store.current(self.a)['messages']), 2)

    def test_submit_idempotency_immutable_snapshot_and_later_version(self):
        with self.assertRaises(SeminarError) as caught:
            self.submit(self.a, '', '', MODEL)
        self.assertEqual(caught.exception.code, 'empty_conversation')
        initial, _ = self.pair(self.a, '<script>alert(1)</script>', 'literal <b>model text</b>')
        receipt = self.submit(self.a, '<img src=x>', 'first note', MODEL)
        duplicate = self.submit(self.a, 'changed name', 'changed note', MODEL)
        self.assertFalse(receipt['alreadySubmitted'])
        self.assertTrue(duplicate['alreadySubmitted'])
        self.assertEqual(receipt['submission'], duplicate['submission'])
        self.pair(self.a, 'second prompt', 'second reply')
        later = self.submit(self.a, '', '', MODEL)
        self.assertNotEqual(receipt['submission']['id'], later['submission']['id'])
        inbox = self.store.submissions()
        self.assertEqual(inbox['submissions'][0]['messages'], initial['messages'])
        self.assertEqual(inbox['submissions'][0]['nickname'], '<img src=x>')
        self.assertEqual(inbox['submissions'][0]['comment'], 'first note')
        self.assertEqual(len(inbox['submissions'][1]['messages']), 4)
        self.assertEqual(self.store.current(self.a)['submittedMessageCount'], 4)
        self.assertEqual(self.store.current(self.b)['submittedMessageCount'], 0)

    def test_inbox_pagination_returns_every_submitted_snapshot_once(self):
        ids = []
        for _ in range(3):
            self.pair(self.a)
            ids.append(self.submit(self.a, '', '', MODEL)['submission']['id'])
        first = self.store.submissions(limit=2)
        self.assertTrue(first['hasMore'])
        second = self.store.submissions(after=first['nextCursor'], limit=2)
        self.assertFalse(second['hasMore'])
        self.assertEqual([row['id'] for row in first['submissions'] + second['submissions']], ids)
        self.assertEqual(self.store.count(), 3)

    def test_submission_keeps_reply_provenance_after_model_redeployment(self):
        self.pair(self.a, 'prompt before redeployment')
        newer_model = {'snapshotId': 'a-different-future-snapshot', 'step': 100}
        self.submit(self.a, '', '', newer_model)
        first = self.store.submissions()['submissions'][0]
        self.assertEqual(first['model'], MODEL)
        self.assertEqual(first['messages'][1]['model'], MODEL)
        turn = self.store.begin_turn(self.a, str(uuid.uuid4()), 'prompt after redeployment')
        self.store.complete_turn(turn, 'new model reply', newer_model, 'eos', 3)
        self.submit(self.a, '', '', newer_model)
        second = self.store.submissions()['submissions'][1]
        self.assertEqual(second['model'], newer_model)
        self.assertEqual(second['messages'][1]['model'], MODEL)
        self.assertEqual(second['messages'][3]['model'], newer_model)

    def test_stale_visible_transcript_cannot_submit_unseen_messages_or_new_conversation(self):
        first, _ = self.pair(self.a, 'reviewed prompt')
        second, _ = self.pair(self.a, 'unseen prompt from another tab')
        with self.assertRaises(SeminarError) as caught:
            self.submit(self.a, expected=first)
        self.assertEqual(caught.exception.code, 'stale_conversation')
        self.assertEqual(self.store.count(), 0)
        self.store.new(self.a)
        self.pair(self.a, 'different conversation in another tab')
        with self.assertRaises(SeminarError) as caught:
            self.submit(self.a, expected=second)
        self.assertEqual(caught.exception.code, 'stale_conversation')
        self.assertEqual(self.store.count(), 0)
        self.submit(self.a)
        self.assertEqual(self.store.count(), 1)

    def test_expired_unsubmitted_history_is_removed_but_submitted_snapshot_remains(self):
        self.pair(self.a, 'shared prompt')
        receipt = self.submit(self.a, '', '', MODEL)
        self.pair(self.b, 'unsubmitted private prompt')
        self.now += 86401
        new_token, conversation = self.store.bootstrap(self.a)
        self.assertNotEqual(new_token, self.a)
        self.assertEqual(conversation['messages'], [])
        with self.assertRaises(SeminarError):
            self.store.current(self.b)
        inbox = self.store.submissions()['submissions']
        self.assertEqual(len(inbox), 1)
        self.assertEqual(inbox[0]['id'], receipt['submission']['id'])
        self.assertEqual(inbox[0]['messages'][0]['content'], 'shared prompt')
        with self.store.connection() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM messages').fetchone()[0], 0)

    def test_restart_clears_pending_without_committing_partial_turn(self):
        turn = self.store.begin_turn(self.a, str(uuid.uuid4()), 'interrupted prompt')
        restarted = SeminarStore(self.path, clock=lambda: self.now)
        self.assertEqual(restarted.current(self.a)['messages'], [])
        with self.assertRaises(SeminarError) as caught:
            restarted.complete_turn(turn, 'late reply', MODEL, 'eos', 1)
        self.assertEqual(caught.exception.code, 'stale_request')
        with self.assertRaises(SeminarError) as caught:
            restarted.begin_turn(self.a, turn['requestId'], 'same request')
        self.assertEqual(caught.exception.code, 'duplicate_request')
        fresh = restarted.begin_turn(self.a, str(uuid.uuid4()), 'fresh prompt')
        self.assertEqual(restarted.complete_turn(fresh, 'new reply', MODEL, 'eos', 2)[0]['messages'][1]['content'], 'new reply')

    def test_ten_turn_and_per_visitor_conversation_caps(self):
        for _ in range(10):
            self.pair(self.a)
        with self.assertRaises(SeminarError) as caught:
            self.store.begin_turn(self.a, str(uuid.uuid4()), 'eleventh')
        self.assertEqual(caught.exception.code, 'turn_limit')
        for _ in range(49):
            self.store.new(self.b)
        with self.assertRaises(SeminarError) as caught:
            self.store.new(self.b)
        self.assertEqual(caught.exception.status, 429)

    def test_clear_removes_only_submitted_copies_and_retains_later_arrivals(self):
        self.pair(self.a)
        first = self.submit(self.a)
        boundary = self.store.inbox_status()['lastCursor']
        self.pair(self.b)
        later = self.submit(self.b)
        before = {token: self.store.current(token) for token in (self.a, self.b)}
        with self.store.connection() as db:
            requests_before = [tuple(row) for row in db.execute('SELECT * FROM requests ORDER BY owner,request_id')]
        result = self.store.clear_submissions(boundary)
        self.assertEqual(result, {'deletedCount': 1, 'submissionCount': 1, 'lastCursor': 2, 'clearedThrough': 1})
        page = self.store.submissions()
        self.assertEqual([item['id'] for item in page['submissions']], [later['submission']['id']])
        self.assertNotEqual(first['submission']['id'], later['submission']['id'])
        self.assertEqual({token: self.store.current(token) for token in before}, before)
        with self.store.connection() as db:
            self.assertEqual([tuple(row) for row in db.execute('SELECT * FROM requests ORDER BY owner,request_id')], requests_before)
        repeated = self.store.clear_submissions(boundary)
        self.assertEqual(repeated, dict(result, deletedCount=0))
        self.store.clear_submissions(2)
        self.assertEqual(self.store.clear_submissions(0), {'deletedCount': 0, 'submissionCount': 0, 'lastCursor': 2, 'clearedThrough': 2})

    def test_full_clear_cannot_reuse_cursor_or_resurrect_cleared_version_after_restart(self):
        self.pair(self.a)
        self.submit(self.a)
        self.store.clear_submissions(1)
        self.assertEqual(self.store.count(), 0)
        self.assertEqual(self.store.submissions(after=1)['submissions'], [])
        self.store = SeminarStore(self.path, clock=lambda: self.now)
        self.assertEqual(self.store.inbox_status(), {'submissionCount': 0, 'lastCursor': 1, 'clearedThrough': 1})
        with self.assertRaises(SeminarError) as caught:
            self.submit(self.a)
        self.assertEqual((caught.exception.status, caught.exception.code), (409, 'submission_cleared'))
        self.assertEqual(self.store.count(), 0)
        self.pair(self.a, 'new explicit version')
        receipt = self.submit(self.a)
        page = self.store.submissions(after=1)
        self.assertEqual([row['cursor'] for row in page['submissions']], [2])
        self.assertEqual(page['submissions'][0]['id'], receipt['submission']['id'])
        self.assertEqual((page['lastCursor'], page['clearedThrough'], page['submissionCount']), (2, 1, 1))

    def test_existing_database_migration_preserves_all_submitted_copies(self):
        self.pair(self.a)
        self.submit(self.a)
        self.pair(self.b)
        self.submit(self.b)
        original = self.store.submissions()['submissions']
        # Simulate the schema immediately preceding the clear feature.
        with self.store.connection() as db:
            db.execute('DROP TABLE inbox_metadata')
        migrated = SeminarStore(self.path, clock=lambda: self.now)
        self.assertEqual(migrated.submissions()['submissions'], original)
        self.assertEqual(migrated.inbox_status(), {'submissionCount': 2, 'lastCursor': 2, 'clearedThrough': 0})
        migrated.clear_submissions(2)
        again = SeminarStore(self.path, clock=lambda: self.now)
        self.assertEqual(again.inbox_status(), {'submissionCount': 0, 'lastCursor': 2, 'clearedThrough': 2})

    def test_clear_rejects_invalid_and_unissued_boundaries_without_changing_state(self):
        initial = self.store.inbox_status()
        for value in (True, False, None, '0', 0.0, -1, 1, 9007199254740991, 9007199254740992):
            with self.subTest(value=value), self.assertRaises(SeminarError) as caught:
                self.store.clear_submissions(value)
            self.assertEqual((caught.exception.status, caught.exception.code), (400, 'invalid_input'))
            self.assertEqual(self.store.inbox_status(), initial)
        self.assertEqual(self.store.clear_submissions(0), dict(initial, deletedCount=0))

    def test_concurrent_clear_and_new_submission_preserve_the_newer_cursor(self):
        self.pair(self.a)
        self.submit(self.a)
        self.pair(self.b)
        barrier = threading.Barrier(2)
        def clear():
            barrier.wait(timeout=3)
            return self.store.clear_submissions(1)
        def submit():
            barrier.wait(timeout=3)
            return self.submit(self.b)
        with ThreadPoolExecutor(max_workers=2) as pool:
            clearing, submitting = pool.submit(clear), pool.submit(submit)
            self.assertEqual(clearing.result(timeout=5)['deletedCount'], 1)
            receipt = submitting.result(timeout=5)
        page = self.store.submissions()
        self.assertEqual([(row['cursor'], row['id']) for row in page['submissions']], [(2, receipt['submission']['id'])])
        self.assertEqual(self.store.inbox_status(), {'submissionCount': 1, 'lastCursor': 2, 'clearedThrough': 1})

    def test_page_rows_and_clear_metadata_share_one_read_snapshot(self):
        self.pair(self.a)
        self.submit(self.a)
        entered, release = threading.Event(), threading.Event()
        reader_identity = [None]
        original = self.store._inbox_status
        def observed_status(db):
            status = original(db)
            if threading.get_ident() == reader_identity[0]:
                entered.set()
                self.assertTrue(release.wait(5))
            return status
        def read_page():
            reader_identity[0] = threading.get_ident()
            return self.store.submissions()
        with patch.object(self.store, '_inbox_status', side_effect=observed_status), ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(read_page)
            try:
                self.assertTrue(entered.wait(3))
                self.assertEqual(self.store.clear_submissions(1)['deletedCount'], 1)
            finally:
                release.set()
            page = future.result(timeout=5)
        self.assertEqual((page['submissionCount'], page['clearedThrough']), (1, 0))
        self.assertEqual([row['cursor'] for row in page['submissions']], [1])
        current = self.store.submissions()
        self.assertEqual((current['submissionCount'], current['clearedThrough'], current['submissions']), (0, 1, []))


if __name__ == '__main__':
    unittest.main()
