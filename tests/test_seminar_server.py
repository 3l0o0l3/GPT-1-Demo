"""Actual loopback HTTP boundaries using a controlled fixture provider.

These tests validate integration contracts, not production model quality.
"""
import copy
import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest
import uuid
from unittest.mock import patch

from seminar_server import AudienceApplication, AudienceHandler, AudienceHTTPServer, RateLimits
from seminar_store import SeminarError, SeminarStore


class FixtureModel:
    def __init__(self):
        self.calls = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.active = self.maximum_active = 0
        self.lock = threading.Lock()
        self.fail = False

    def metadata(self):
        return {'snapshotId': 'fixture-only', 'step': 7, 'parameters': 1,
                'trainContext': 64, 'modelContext': 128, 'savedAt': '2026-01-01T00:00:00+00:00'}

    def classify(self, text):
        if self.fail:
            raise RuntimeError('Controlled classification failure')
        return {'text': text, 'snapshotId': 'a' * 64, 'branches': [
            {'branch': name, 'prediction': 'positive',
             'probabilities': [{'label': 'negative', 'p': .25}, {'label': 'positive', 'p': .75}],
             'step': 500, 'source': {'step': 1000, 'seed': 42, 'source': 'pretrained', 'modelSha256': 'b' * 64},
             'runId': 'c' * 32} for name in ('pretrained', 'scratch')]}

    def generate(self, text, max_new_tokens, temperature, seed, on_event):
        with self.lock:
            self.calls.append(text)
            self.active += 1
            self.maximum_active = max(self.active, self.maximum_active)
        self.entered.set()
        try:
            if not self.release.wait(8):
                raise TimeoutError('Fixture did not release')
            on_event({'type': 'start'})
            on_event({'type': 'token', 'index': 0, 'continuation': 'Fixture reply.'})
            if self.fail:
                raise RuntimeError('Controlled fixture provider failure')
            on_event({'type': 'token', 'index': 1, 'continuation': 'Fixture reply.'})
            result = {'continuation': 'Fixture reply.', 'stopReason': 'eos',
                      'tokens': [{'id': 5}, {'id': 2}]}
            on_event({'type': 'done', 'result': result})
            return result
        finally:
            with self.lock:
                self.active -= 1


class SeminarHTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for name in ('index.html', 'app.js', 'style.css'):
            (self.root / name).write_text('static fixture')
        self.store = SeminarStore(self.root / 'state.sqlite')
        self.model = FixtureModel()
        self.app = AudienceApplication(self.store, self.model, 'https://seminar.example/seminar-gpt/', self.root)
        self.public = AudienceHTTPServer(('127.0.0.1', 0), self.app)
        self.private = AudienceHTTPServer(('127.0.0.1', 0), self.app, private=True)
        self.threads = []
        for server in (self.public, self.private):
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            self.threads.append(thread)

    def tearDown(self):
        self.model.release.set()
        for server in (self.public, self.private):
            server.shutdown()
            server.server_close()
        for thread in self.threads:
            thread.join(timeout=2)
        self.temp.cleanup()

    def open_request(self, path, data=None, session=None, headers=None, private=False, method=None):
        server = self.private if private else self.public
        connection = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=8)
        outgoing = {'Host': f'127.0.0.1:{server.server_port}' if private else 'seminar.example'}
        if data is not None:
            outgoing['Content-Type'] = 'application/json'
        if session:
            outgoing['Cookie'] = session['cookie']
            outgoing['X-CSRF-Token'] = session['csrf']
        outgoing.update(headers or {})
        connection.request(method or ('POST' if data is not None else 'GET'), path,
                           json.dumps(data) if data is not None else None, outgoing)
        return connection, connection.getresponse()

    def request(self, *args, **kwargs):
        connection, response = self.open_request(*args, **kwargs)
        try:
            raw = response.read()
            if 'application/x-ndjson' in response.getheader('Content-Type', ''):
                body = [json.loads(line) for line in raw.splitlines()]
            elif 'application/json' in response.getheader('Content-Type', ''):
                body = json.loads(raw)
            else:
                body = raw.decode()
            return response.status, body, dict(response.getheaders())
        finally:
            connection.close()

    def session(self):
        status, body, headers = self.request('/seminar-gpt/api/session', {})
        self.assertEqual(status, 200)
        self.assertIn('HttpOnly', headers['Set-Cookie'])
        self.assertIn('Secure', headers['Set-Cookie'])
        self.assertIn('SameSite=Lax', headers['Set-Cookie'])
        return {'cookie': headers['Set-Cookie'].split(';')[0], 'csrf': body['csrf'], 'body': body}

    def chat(self, session, text='hello', request_id=None):
        return self.request('/seminar-gpt/api/chat', {'message': text, 'requestId': request_id or str(uuid.uuid4())}, session)

    def test_model_sync_is_private_disabled_by_default_and_strict_when_enabled(self):
        status = self.request('/admin/status', private=True)[1]
        self.assertEqual(status['modelPublication'],
                         {'enabled': False, 'releaseId': None, 'publishedAt': None, 'lastRequestId': None})
        body = {'requestId': str(uuid.uuid4()), 'expectedReleaseId': 'a' * 64}
        self.assertEqual(self.request('/admin/model/sync', body, private=True)[0], 503)
        for path in ('/admin/model/sync', '/seminar-gpt/admin/model/sync', '/seminar-gpt/api/model/sync'):
            self.assertEqual(self.request(path, body)[0], 404)
        self.assertEqual(self.request('/admin/model/sync', body, private=True,
                                      headers={'Origin': 'https://evil.example'})[0], 403)

    def test_chat_and_classification_pin_one_provider_even_if_active_reference_changes(self):
        old = self.model
        new = FixtureModel()
        old_metadata = dict(old.metadata(), classifier={'available': True})
        new_metadata = dict(new.metadata(), step=99, snapshotId='new-fixture')
        class Holder:
            current_model = old
        holder = Holder()
        self.app.publisher = holder
        # A concurrent reference replacement must not mix old inference with new
        # metadata. Real publication additionally waits for the worker lock.
        original_generate = old.generate
        def generate(*args, **kwargs):
            holder.current_model = new
            return original_generate(*args, **kwargs)
        a = self.session()
        with patch.object(old, 'generate', side_effect=generate):
            events = self.chat(a)[1]
        self.assertEqual(events[1]['model']['step'], 7)
        self.assertEqual(events[-1]['assistantMessage']['model']['step'], 7)
        holder.current_model = old
        original_classify = old.classify
        def classify(text):
            holder.current_model = new
            return original_classify(text)
        with patch.object(old, 'metadata', return_value=old_metadata), \
                patch.object(new, 'metadata', return_value=new_metadata), \
                patch.object(old, 'classify', side_effect=classify):
            code, result, _ = self.request('/seminar-gpt/api/classify',
                {'message': 'good story', 'requestId': str(uuid.uuid4())}, a)
        self.assertEqual(code, 200)
        self.assertEqual(result['assistantMessage']['model'], old_metadata)
        self.assertEqual(result['model'], old_metadata)
        holder.current_model = old

    def test_classifier_availability_is_rechecked_after_waiting_for_worker(self):
        old, new = self.model, FixtureModel()
        class Holder:
            current_model = old
        holder = Holder()
        self.app.publisher = holder
        a = self.session()
        entered = threading.Event()
        actual_acquire = self.app.worker.acquire
        class Worker:
            def acquire(inner, **kwargs):
                entered.set()
                holder.current_model = new
                return actual_acquire(**kwargs)
            def release(inner):
                return self.app.worker_original.release()
        self.app.worker_original = self.app.worker
        self.app.worker = Worker()
        metadata = dict(old.metadata(), classifier={'available': True})
        with patch.object(old, 'metadata', return_value=metadata), patch.object(new, 'classify') as classify:
            code, result, _ = self.request('/seminar-gpt/api/classify',
                {'message': 'good story', 'requestId': str(uuid.uuid4())}, a)
        self.assertEqual((code, result['error']['code']), (503, 'classifier_unavailable'))
        self.assertTrue(entered.is_set())
        classify.assert_not_called()
        self.assertEqual(self.store.current(a['cookie'].split('=', 1)[1])['messages'], [])
        self.assertFalse(self.app.worker_original.locked())

    def test_classification_is_server_owned_submittable_and_excluded_from_story_context(self):
        a, b = self.session(), self.session()
        payload = {'message': 'great film', 'requestId': str(uuid.uuid4())}
        route = '/seminar-gpt/api/classify'
        self.assertEqual(self.request(route, payload, a)[1]['error']['code'], 'classifier_unavailable')
        metadata = dict(self.model.metadata(), classifier={'available': True})
        with patch.object(self.model, 'metadata', return_value=metadata):
            self.assertEqual(self.request(route, payload, a, {'X-CSRF-Token': 'bad'})[0], 403)
            self.assertEqual(self.request(route, dict(payload, classification={}), a)[0], 400)
            code, result, _ = self.request(route, payload, a)
            self.assertEqual(code, 200)
            assistant = result['assistantMessage']
            self.assertEqual(assistant['kind'], 'classification')
            self.assertEqual(assistant['classification'], self.model.classify('great film'))
            self.assertEqual(self.request(route, payload, a)[0], 409)
            self.assertEqual(self.request('/seminar-gpt/api/session', {}, b)[1]['conversation']['messages'], [])
            current = result['conversation']
            submit = {'nickname': '', 'comment': 'Compare the two heads', 'consent': True,
                      'expectedConversationId': current['id'], 'expectedMessageCount': 2}
            self.assertEqual(self.request('/admin/status', private=True)[1]['submissionCount'], 0)
            self.assertEqual(self.request('/seminar-gpt/api/submit', submit, a)[0], 200)
            inbox = self.request('/admin/submissions', private=True)[1]['submissions']
            self.assertEqual(inbox[0]['messages'][-1]['classification'], assistant['classification'])
            self.chat(a, 'the story')
            self.assertEqual(self.model.calls[-1], 'the story')
            self.chat(a, 'next')
            self.assertEqual(self.model.calls[-1], 'User: the story\nModel: Fixture reply.\nUser: next\nModel:')
            self.assertEqual(self.request('/seminar-gpt/api/submit', submit, a)[1]['error']['code'], 'stale_conversation')

    def test_failed_classification_cleans_pending_and_never_commits_partial_result(self):
        a = self.session()
        metadata = dict(self.model.metadata(), classifier={'available': True})
        with patch.object(self.model, 'metadata', return_value=metadata):
            self.model.fail = True
            self.assertEqual(self.request('/seminar-gpt/api/classify',
                {'message': 'great', 'requestId': str(uuid.uuid4())}, a)[0], 503)
            current = self.request('/seminar-gpt/api/session', {}, a)[1]['conversation']
            self.assertEqual(current['messages'], [])
            self.model.fail = False
            self.assertEqual(self.request('/seminar-gpt/api/classify',
                {'message': 'great', 'requestId': str(uuid.uuid4())}, a)[0], 200)

    def test_public_route_allowlist_headers_csrf_and_strict_fields(self):
        a = self.session()
        for path in ('/admin/status', '/admin/submissions', '/seminar-gpt/admin/submissions',
                     '/seminar-gpt/model.pt', '/seminar-gpt/state.sqlite', '/seminar-gpt/../seminar_server.py',
                     '/seminar-gpt/api/training/train', '/seminar-gpt/api/submissions'):
            self.assertEqual(self.request(path)[0], 404, path)
        self.assertEqual(self.request('/seminar-gpt/api/training/train', {})[0], 404)
        self.assertEqual(self.request('/seminar-gpt/api/info', headers={'Host': 'evil.example'})[0], 403)
        self.assertEqual(self.request('/seminar-gpt/api/session', {}, headers={'Origin': 'https://evil.example'})[0], 403)
        self.assertEqual(self.request('/seminar-gpt/api/new', {}, a, {'X-CSRF-Token': 'bad'})[0], 403)
        self.assertEqual(self.request('/seminar-gpt/api/new', {}, None)[0], 403)
        self.assertEqual(self.request('/seminar-gpt/api/chat', {'message': 'hello', 'requestId': str(uuid.uuid4()), 'history': []}, a)[0], 400)
        status, _, headers = self.request('/seminar-gpt/')
        self.assertEqual(status, 200)
        self.assertNotIn('Access-Control-Allow-Origin', headers)
        self.assertIn("frame-ancestors 'none'", headers['Content-Security-Policy'])
        self.assertEqual(self.request('/admin/status', private=True, headers={'Host': 'evil.example'})[0], 403)
        self.assertEqual(self.request('/admin/status', private=True, headers={'Origin': 'https://evil.example'})[0], 403)

    def test_real_http_stream_history_truth_session_isolation_and_submit_consent(self):
        a, b = self.session(), self.session()
        status, events, _ = self.chat(a, '<script>alert(1)</script>')
        self.assertEqual(status, 200)
        self.assertEqual([e['type'] for e in events], ['queued', 'start', 'token', 'token', 'done'])
        done = events[-1]
        self.assertEqual(done['assistantMessage']['stopReason'], 'eos')
        self.assertEqual(done['assistantMessage']['tokenCount'], 2)
        self.assertEqual(len(done['conversation']['messages']), 2)
        self.assertEqual(self.request('/seminar-gpt/api/session', {}, b)[1]['conversation']['messages'], [])
        self.chat(a, 'follow up')
        self.assertEqual(self.model.calls, ['<script>alert(1)</script>',
                         'User: <script>alert(1)</script>\nModel: Fixture reply.\nUser: follow up\nModel:'])
        current = self.request('/seminar-gpt/api/session', {}, a)[1]['conversation']
        payload = {'nickname': '<img src=x>', 'comment': 'discuss this', 'consent': False,
                   'expectedConversationId': current['id'], 'expectedMessageCount': len(current['messages'])}
        self.assertEqual(self.request('/seminar-gpt/api/submit', payload, a)[0], 400)
        self.assertEqual(self.request('/admin/status', private=True)[1]['submissionCount'], 0)
        payload['consent'] = True
        first = self.request('/seminar-gpt/api/submit', payload, a)[1]
        second = self.request('/seminar-gpt/api/submit', payload, a)[1]
        self.assertFalse(first['alreadySubmitted'])
        self.assertTrue(second['alreadySubmitted'])
        self.assertEqual(first['submission'], second['submission'])
        inbox = self.request('/admin/submissions', private=True)[1]['submissions']
        self.assertEqual(len(inbox), 1)
        self.assertEqual(inbox[0]['nickname'], '<img src=x>')
        self.assertEqual(inbox[0]['messages'][0]['content'], '<script>alert(1)</script>')
        self.assertEqual(len(inbox[0]['messages']), 4)
        self.assertEqual(self.request('/seminar-gpt/api/submit', payload, b)[0], 409)

    def test_multitab_submit_rejects_changed_history_and_replaced_conversation(self):
        a = self.session()
        first = self.chat(a, 'reviewed first prompt')[1][-1]['conversation']
        payload = {'nickname': '', 'comment': '', 'consent': True,
                   'expectedConversationId': first['id'], 'expectedMessageCount': len(first['messages'])}
        second = self.chat(a, 'unseen prompt from another tab')[1][-1]['conversation']
        status, error, _ = self.request('/seminar-gpt/api/submit', payload, a)
        self.assertEqual(status, 409)
        self.assertEqual(error['error']['code'], 'stale_conversation')
        self.assertEqual(self.request('/admin/status', private=True)[1]['submissionCount'], 0)
        payload['expectedMessageCount'] = len(second['messages'])
        self.assertEqual(self.request('/seminar-gpt/api/new', {}, a)[0], 200)
        latest = self.chat(a, 'new conversation in other tab')[1][-1]['conversation']
        status, error, _ = self.request('/seminar-gpt/api/submit', payload, a)
        self.assertEqual(status, 409)
        self.assertEqual(error['error']['code'], 'stale_conversation')
        self.assertEqual(self.request('/admin/status', private=True)[1]['submissionCount'], 0)
        payload.update(expectedConversationId=latest['id'], expectedMessageCount=len(latest['messages']))
        self.assertEqual(self.request('/seminar-gpt/api/submit', payload, a)[0], 200)
        self.assertEqual(self.request('/admin/status', private=True)[1]['submissionCount'], 1)

    def test_clear_http_is_private_bounded_and_preserves_visitor_history(self):
        visitor = self.session()
        current = self.chat(visitor)[1][-1]['conversation']
        payload = {'nickname': '', 'comment': '', 'consent': True,
                   'expectedConversationId': current['id'], 'expectedMessageCount': len(current['messages'])}
        self.assertEqual(self.request('/seminar-gpt/api/submit', payload, visitor)[0], 200)
        initial = self.request('/admin/status', private=True)[1]
        self.assertEqual((initial['submissionCount'], initial['lastCursor'], initial['clearedThrough']), (1, 1, 0))
        for route in ('/admin/submissions/clear', '/seminar-gpt/admin/submissions/clear', '/seminar-gpt/api/submissions/clear'):
            self.assertEqual(self.request(route, {'throughCursor': 1}, visitor)[0], 404)
        for headers in ({'Host': 'evil.example'}, {'Origin': 'https://evil.example'}, {'Sec-Fetch-Site': 'cross-site'}):
            self.assertEqual(self.request('/admin/submissions/clear', {'throughCursor': 1}, private=True, headers=headers)[0], 403)
        for body in ({}, {'throughCursor': True}, {'throughCursor': -1}, {'throughCursor': 2}, {'throughCursor': 9007199254740992}, {'throughCursor': 1, 'all': True}):
            self.assertEqual(self.request('/admin/submissions/clear', body, private=True)[0], 400)
        self.assertEqual(self.request('/admin/submissions/clear', private=True)[0], 404)
        self.assertEqual(self.request('/admin/submissions/clear?extra=1', {'throughCursor': 1}, private=True)[0], 404)
        self.assertEqual(self.store.count(), 1)
        status, result, _ = self.request('/admin/submissions/clear', {'throughCursor': 1}, private=True)
        self.assertEqual(status, 200)
        self.assertEqual(result, {'deletedCount': 1, 'submissionCount': 0, 'lastCursor': 1, 'clearedThrough': 1})
        repeated = self.request('/admin/submissions/clear', {'throughCursor': 1}, private=True)[1]
        self.assertEqual(repeated, dict(result, deletedCount=0))
        recovered = self.request('/seminar-gpt/api/session', {}, visitor)[1]['conversation']
        self.assertEqual(recovered['messages'], current['messages'])
        self.assertEqual(self.request('/seminar-gpt/api/submit', payload, visitor)[1]['error']['code'], 'submission_cleared')
        page = self.request('/admin/submissions?after=1', private=True)[1]
        self.assertEqual((page['submissionCount'], page['lastCursor'], page['clearedThrough'], page['submissions']), (0, 1, 1, []))

    def test_provider_failure_commits_nothing_and_request_is_not_replayed(self):
        a = self.session()
        identifier = str(uuid.uuid4())
        self.model.fail = True
        status, events, _ = self.chat(a, 'will fail', identifier)
        self.assertEqual(status, 200)
        self.assertEqual(events[-1]['type'], 'error')
        self.assertNotIn('done', [e['type'] for e in events])
        self.assertEqual(self.request('/seminar-gpt/api/session', {}, a)[1]['conversation']['messages'], [])
        self.assertEqual(self.chat(a, 'will fail', identifier)[0], 409)
        self.model.fail = False
        self.assertEqual(self.chat(a, 'fresh')[1][-1]['type'], 'done')

    def test_disconnect_before_commit_discards_partial_and_after_commit_keeps_result(self):
        original = AudienceHandler._event
        for fail_at, expected_messages in (('token', 0), ('done', 2)):
            a = self.session()
            identifier = str(uuid.uuid4())

            def disconnected(handler, event):
                if event['type'] == fail_at:
                    raise BrokenPipeError('simulated socket write failure')
                return original(handler, event)

            with self.subTest(fail_at=fail_at), patch.object(AudienceHandler, '_event', new=disconnected):
                status, events, _ = self.chat(a, 'disconnect', identifier)
                self.assertEqual(status, 200)
                self.assertNotIn('done', [e['type'] for e in events])
            current = self.request('/seminar-gpt/api/session', {}, a)[1]['conversation']
            self.assertEqual(len(current['messages']), expected_messages)
            self.assertEqual(self.chat(a, 'disconnect', identifier)[0], 409)
            self.assertEqual(self.chat(a, 'next request')[1][-1]['type'], 'done')

    def test_two_visitors_are_queued_and_single_worker_releases(self):
        a, b = self.session(), self.session()
        self.model.release.clear()
        first, response1 = self.open_request('/seminar-gpt/api/chat',
            {'message': 'visitor one', 'requestId': str(uuid.uuid4())}, a)
        second = None
        try:
            self.assertEqual(json.loads(response1.readline())['type'], 'queued')
            self.assertEqual(json.loads(response1.readline())['type'], 'start')
            self.assertTrue(self.model.entered.wait(2))
            second, response2 = self.open_request('/seminar-gpt/api/chat',
                {'message': 'visitor two', 'requestId': str(uuid.uuid4())}, b)
            self.assertEqual(json.loads(response2.readline())['type'], 'queued')
            self.assertEqual(self.model.calls, ['visitor one'])
            self.assertEqual(self.request('/seminar-gpt/api/new', {}, a)[0], 409)
            self.assertEqual(self.chat(a, 'same visitor collision')[0], 409)
            self.model.release.set()
            remaining1 = [json.loads(line) for line in response1.read().splitlines()]
            remaining2 = [json.loads(line) for line in response2.read().splitlines()]
            self.assertEqual(remaining1[-1]['type'], 'done')
            self.assertEqual(remaining2[0]['type'], 'start')
            self.assertEqual(remaining2[-1]['type'], 'done')
            self.assertEqual(self.model.maximum_active, 1)
            self.assertEqual(self.model.calls, ['visitor one', 'visitor two'])
        finally:
            self.model.release.set()
            first.close()
            if second:
                second.close()

    def test_full_queue_rejects_before_turn_reservation_and_accepts_after_release(self):
        a = self.session()
        for _ in range(16):
            self.assertTrue(self.app.slots.acquire(blocking=False))
        try:
            status, value, _ = self.chat(a, 'queue full')
            self.assertEqual(status, 429)
            self.assertEqual(value['error']['code'], 'queue_full')
            self.assertEqual(self.model.calls, [])
        finally:
            for _ in range(16):
                self.app.slots.release()
        self.assertEqual(self.chat(a, 'available again')[1][-1]['type'], 'done')


class RateLimitTests(unittest.TestCase):
    def test_per_key_window_and_global_counter_capacity(self):
        now = [0]
        limiter = RateLimits(clock=lambda: now[0])
        limiter.check('visitor', 1)
        with self.assertRaises(SeminarError) as caught:
            limiter.check('visitor', 1)
        self.assertEqual(caught.exception.status, 429)
        now[0] = 61
        limiter.check('visitor', 1)
        for index in range(9999):
            limiter.check(str(index), 1)
        with self.assertRaises(SeminarError):
            limiter.check('overflow', 1)
        self.assertLessEqual(len(limiter.items), 10000)


if __name__ == '__main__':
    unittest.main()
