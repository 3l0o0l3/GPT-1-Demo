"""Isolated real CPU bundle loading and controlled publication scheduling.

Random fixture weights exercise the exact decoder/checkpoint schema, not model
quality. No trainer, production endpoint, optimization, or real session is used.
"""
import base64
import copy
import hashlib
import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest
import uuid
from unittest.mock import Mock, patch

import torch
from tokenizers import Tokenizer, models, pre_tokenizers

from training_engine import GPT, CONFIG
from seminar_model import FrozenLanguageModel, file_sha256
from seminar_publish import EXPORT_URL, SeminarPublisher, _decode_export, _fetch_export, release_id
from seminar_store import SeminarError


def make_bundle(directory, seed, classifier=False):
    directory.mkdir()
    words = ['[PAD]', '[BOS]', '[EOS]', '[CLS]', '[UNK]', 'the', 'good', 'bad', 'story', '.']
    tokenizer = Tokenizer(models.WordLevel({word: i for i, word in enumerate(words)}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(directory / 'tokenizer.json'))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = GPT(len(words))
    checkpoint = {'formatVersion': 1, 'config': dict(CONFIG), 'vocabSize': len(words),
                  'model': model.state_dict(), 'source': {'step': 0, 'seed': seed, 'source': 'random'},
                  'tokenizerSha256': file_sha256(directory / 'tokenizer.json'),
                  'dataManifestSha256': 'a' * 64, 'savedAt': '2026-01-01T00:00:00+00:00'}
    torch.save(checkpoint, directory / 'model.pt')
    names = ['model.pt', 'tokenizer.json']
    if classifier:
        # Synthetic unit-test provenance; the tensor math and byte preservation
        # are real, but this fixture does not claim classifier training quality.
        pair = {key: copy.deepcopy(checkpoint[key]) for key in
                ('formatVersion', 'config', 'vocabSize', 'tokenizerSha256', 'dataManifestSha256', 'savedAt')}
        pair.update(source={**checkpoint['source'], 'modelSha256': file_sha256(directory / 'model.pt')},
                    step=1, runId='b' * 32, branches={name: copy.deepcopy(model.state_dict())
                                                   for name in ('pretrained', 'scratch')})
        torch.save(pair, directory / 'classifier.pt')
        names.append('classifier.pt')
    manifest = {'bundleVersion': 2 if classifier else 1, 'snapshotId': file_sha256(directory / 'model.pt'),
                'files': {name: file_sha256(directory / name) for name in names}}
    (directory / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')


def bundle_export(directory):
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    manifest = {'bundleVersion': 1, 'snapshotId': manifest['snapshotId'],
                'files': {name: manifest['files'][name] for name in ('model.pt', 'tokenizer.json')}}
    return {'exportVersion': 1, 'manifest': manifest,
            'files': {name: base64.b64encode((directory / name).read_bytes()).decode('ascii')
                      for name in manifest['files']}}


class SeminarPublishTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.fixtures = tempfile.TemporaryDirectory()
        root = Path(cls.fixtures.name)
        cls.old, cls.new = root / 'old', root / 'new'
        make_bundle(cls.old, 41, classifier=True)
        make_bundle(cls.new, 42)
        cls.export = bundle_export(cls.new)

    @classmethod
    def tearDownClass(cls):
        cls.fixtures.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.fetch = Mock(return_value=copy.deepcopy(self.export))
        self.publisher = SeminarPublisher(self.root, self.old, EXPORT_URL, fetcher=self.fetch,
            clock=lambda: '2026-01-02T00:00:00+00:00', worker_timeout=.02)
        self.worker = threading.Lock()

    def tearDown(self):
        self.temp.cleanup()

    def request(self):
        return {'requestId': str(uuid.uuid4()), 'expectedReleaseId': self.publisher.status()['releaseId']}

    def assert_failure_preserves_active(self, body, code=None):
        provider = self.publisher.current_model
        before = self.publisher.describe()
        pointer = self.publisher.pointer.read_bytes() if self.publisher.pointer.exists() else None
        with self.assertRaises(SeminarError) as caught:
            self.publisher.sync(body, self.worker)
        if code:
            self.assertEqual(caught.exception.code, code)
        self.assertIs(self.publisher.current_model, provider)
        self.assertEqual(self.publisher.describe(), before)
        self.assertEqual(self.publisher.pointer.read_bytes() if self.publisher.pointer.exists() else None, pointer)
        self.assertFalse(self.worker.locked())

    def test_real_cpu_swap_preserves_classifier_and_restores_durable_selection(self):
        old = self.publisher.current_model.metadata()
        classification = self.publisher.current_model.classify('the good story')
        rng = torch.random.get_rng_state().clone()
        body = self.request()
        result = self.publisher.sync(body, self.worker)
        self.assertEqual(set(result), {'model', 'modelPublication', 'alreadyPublished'})
        self.assertFalse(result['alreadyPublished'])
        self.assertEqual(result['model']['seed'], 42)
        self.assertEqual(result['model']['classifier'], old['classifier'])
        self.assertEqual(self.publisher.current_model.classify('the good story'), classification)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        pointer = json.loads(self.publisher.pointer.read_text())
        self.assertEqual((self.root / pointer['bundle'] / 'classifier.pt').read_bytes(), (self.old / 'classifier.pt').read_bytes())
        restored = SeminarPublisher(self.root, self.root / 'missing-original', EXPORT_URL, fetcher=Mock())
        self.assertEqual(restored.describe(), self.publisher.describe())
        repeat = restored.sync(body, self.worker)
        self.assertTrue(repeat['alreadyPublished'])
        restored.fetcher.assert_not_called()
        disabled = SeminarPublisher(self.root, self.old)
        self.assertFalse(disabled.status()['enabled'])
        self.assertEqual(disabled.current_model.metadata(), result['model'])

    def test_cas_and_idempotency_never_fetch_or_replay_mutation(self):
        body = self.request()
        self.publisher.sync(body, self.worker)
        self.assertTrue(self.publisher.sync(body, self.worker)['alreadyPublished'])
        self.fetch.assert_called_once()
        self.assert_failure_preserves_active({**body, 'expectedReleaseId': 'f' * 64}, 'request_conflict')
        self.assert_failure_preserves_active({**body, 'requestId': str(uuid.uuid4())}, 'stale_release')
        self.fetch.assert_called_once()

    def test_hash_base64_schema_and_tokenizer_failures_do_not_replace(self):
        invalid = []
        value = copy.deepcopy(self.export)
        value['manifest']['files']['tokenizer.json'] = 'f' * 64
        invalid.append(value)
        value = copy.deepcopy(self.export)
        value['files']['model.pt'] = '!!not base64!!'
        invalid.append(value)
        value = copy.deepcopy(self.export)
        value['files']['../model.pt'] = value['files'].pop('model.pt')
        invalid.append(value)
        value = copy.deepcopy(self.export)
        value['exportVersion'] = True
        invalid.append(value)
        # Hash-valid altered tokenizer is incompatible with the retained heads.
        value = copy.deepcopy(self.export)
        data = base64.b64decode(value['files']['tokenizer.json']) + b'\n'
        value['files']['tokenizer.json'] = base64.b64encode(data).decode('ascii')
        value['manifest']['files']['tokenizer.json'] = hashlib.sha256(data).hexdigest()
        invalid.append(value)
        value = copy.deepcopy(self.export)
        data = b'not a valid torch checkpoint'
        value['files']['model.pt'] = base64.b64encode(data).decode('ascii')
        value['manifest']['snapshotId'] = hashlib.sha256(data).hexdigest()
        value['manifest']['files']['model.pt'] = value['manifest']['snapshotId']
        invalid.append(value)
        for value in invalid:
            with self.subTest(case=len(value['files'])):
                self.fetch.return_value = value
                self.assert_failure_preserves_active(self.request(), 'publication_failed')

    def test_training_busy_is_reported_without_fallback_or_retry(self):
        self.fetch.side_effect = SeminarError(409, 'training_busy', 'busy')
        self.assert_failure_preserves_active(self.request(), 'training_busy')
        self.fetch.assert_called_once()

    def test_export_transport_is_fixed_bounded_and_does_not_follow_redirects_or_retry(self):
        for status, payload, expected in (
                (409, {'error': {'code': 'training_busy', 'message': 'private details'}}, 'training_busy'),
                (302, {}, 'export_unavailable'), (503, {}, 'export_unavailable'),
                (409, {'error': []}, 'export_unavailable')):
            with self.subTest(status=status):
                connection = Mock()
                connection.getresponse.return_value.status = status
                connection.getresponse.return_value.read.return_value = json.dumps(payload).encode('utf-8')
                with patch('seminar_publish.http.client.HTTPConnection', return_value=connection) as constructor:
                    with self.assertRaises(SeminarError) as caught:
                        _fetch_export()
                self.assertEqual(caught.exception.code, expected)
                constructor.assert_called_once_with('127.0.0.1', 18769, timeout=10)
                connection.request.assert_called_once_with('GET', '/api/training/audience-export',
                                                           headers={'Host': '127.0.0.1:18769'})
                connection.close.assert_called_once()
        connection = Mock()
        connection.getresponse.return_value.status = 200
        connection.getresponse.return_value.read.return_value = b'12345'
        with patch('seminar_publish.http.client.HTTPConnection', return_value=connection), \
                patch('seminar_publish.MAX_EXPORT_BYTES', 4):
            with self.assertRaises(SeminarError) as caught:
                _fetch_export()
        self.assertEqual(caught.exception.code, 'export_unavailable')
        connection.getresponse.return_value.read.assert_called_once_with(5)

    def test_strict_checkpoint_failure_and_atomic_write_failure_keep_previous_release(self):
        body = self.request()
        self.publisher.sync(body, self.worker)
        with patch.object(self.publisher, 'loader', side_effect=ValueError('bad tensor shape')):
            self.assert_failure_preserves_active(self.request(), 'publication_failed')
        with patch('seminar_publish.os.replace', side_effect=OSError('disk failure')):
            self.assert_failure_preserves_active(self.request(), 'publication_failed')
        restored = SeminarPublisher(self.root, self.old)
        self.assertEqual(restored.current_model.metadata(), self.publisher.current_model.metadata())

    def test_busy_worker_and_concurrent_sync_do_not_swap_or_hold_worker(self):
        self.worker.acquire()
        old = self.publisher.current_model
        try:
            with self.assertRaises(SeminarError) as caught:
                self.publisher.sync(self.request(), self.worker)
            self.assertEqual(caught.exception.code, 'audience_busy')
            self.assertIs(self.publisher.current_model, old)
            self.assertTrue(self.worker.locked())
        finally:
            self.worker.release()
        entered, release = threading.Event(), threading.Event()
        outcome = []
        def fetch():
            entered.set()
            release.wait(3)
            return self.export
        self.publisher.fetcher = fetch
        def publish():
            outcome.append(self.publisher.sync(self.request(), self.worker))
        thread = threading.Thread(target=publish)
        thread.start()
        try:
            self.assertTrue(entered.wait(2))
            self.assertFalse(self.worker.locked())
            with self.assertRaises(SeminarError) as caught:
                self.publisher.sync(self.request(), self.worker)
            self.assertEqual(caught.exception.code, 'publication_busy')
        finally:
            release.set()
            thread.join(5)
        self.assertEqual(len(outcome), 1)
        self.assertFalse(self.worker.locked())

    def test_error_guards_strict_ids_disabled_and_controlled_restore_path(self):
        for body in ({}, {**self.request(), 'extra': 1}, {**self.request(), 'requestId': uuid.uuid4().hex},
                     {**self.request(), 'expectedReleaseId': 'A' * 64}):
            self.assert_failure_preserves_active(body, 'invalid_input')
        self.fetch.assert_not_called()
        for url in ('https://127.0.0.1:18769/api/training/audience-export',
                    EXPORT_URL + '?file=/etc/passwd', 'http://example.org/'):
            with self.assertRaises(ValueError):
                SeminarPublisher(self.root, self.old, url)
        self.publisher.enabled = False
        self.assert_failure_preserves_active(self.request(), 'publication_disabled')
        self.publisher.enabled = True
        self.publisher.sync(self.request(), self.worker)
        pointer = json.loads(self.publisher.pointer.read_text())
        pointer['bundle'] = '../old'
        self.publisher.pointer.write_text(json.dumps(pointer))
        with self.assertRaises(ValueError):
            SeminarPublisher(self.root, self.old)

    def test_release_id_accounts_for_classifier_and_decode_has_size_bounds(self):
        metadata = self.publisher.current_model.metadata()
        changed = copy.deepcopy(metadata)
        changed['classifier']['snapshotId'] = 'f' * 64
        self.assertNotEqual(release_id(metadata), release_id(changed))
        value = copy.deepcopy(self.export)
        with patch('seminar_publish.FILE_LIMITS', {'model.pt': 1, 'tokenizer.json': 1}), self.assertRaises(ValueError):
            _decode_export(value)

    def test_private_http_sync_waits_for_inflight_turn_and_preserves_stored_provenance(self):
        from seminar_server import AudienceApplication, AudienceHTTPServer
        from seminar_store import SeminarStore
        store = SeminarStore(self.root / 'state.sqlite')
        app = AudienceApplication(store, self.publisher.current_model,
                                  'http://seminar.example/seminar-gpt/', self.root, publisher=self.publisher)
        public = AudienceHTTPServer(('127.0.0.1', 0), app)
        private = AudienceHTTPServer(('127.0.0.1', 0), app, private=True)
        servers = (public, private)
        threads = [threading.Thread(target=server.serve_forever, daemon=True) for server in servers]
        for thread in threads:
            thread.start()
        self.publisher.worker_timeout = 3
        entered, release = threading.Event(), threading.Event()
        fetched = threading.Event()
        self.fetch.side_effect = lambda: (fetched.set(), self.export)[1]
        old_model = self.publisher.current_model
        old_metadata = old_model.metadata()
        original_generate = old_model.generate
        outcomes, failures = {}, []
        def generate(*args, **kwargs):
            entered.set()
            if not release.wait(3):
                raise TimeoutError('Controlled test response was not released')
            kwargs['max_new_tokens'] = 2
            return original_generate(*args, **kwargs)
        def request(server, path, body=None, headers=None):
            connection = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=6)
            outgoing = {'Host': f'127.0.0.1:{server.server_port}' if server is private else 'seminar.example'}
            outgoing.update(headers or {})
            if body is not None:
                outgoing['Content-Type'] = 'application/json'
            try:
                connection.request('POST' if body is not None else 'GET', path,
                                   json.dumps(body) if body is not None else None, outgoing)
                response = connection.getresponse()
                data = response.read()
                value = ([json.loads(line) for line in data.splitlines()]
                         if 'ndjson' in response.getheader('Content-Type', '') else json.loads(data))
                return response.status, value
            finally:
                connection.close()
        def chat():
            try:
                outcomes['chat'] = request(public, '/seminar-gpt/api/chat',
                    {'message': 'the good story', 'requestId': str(uuid.uuid4())},
                    {'Cookie': 'seminar_session=' + token, 'X-CSRF-Token': app.csrf(token)})
            except Exception as error:
                failures.append(error)
        def publish():
            try:
                outcomes['publish'] = request(private, '/admin/model/sync', body)
            except Exception as error:
                failures.append(error)
        token, _ = store.bootstrap()
        body = self.request()
        conversation_thread = threading.Thread(target=chat)
        publication_thread = threading.Thread(target=publish)
        try:
            with patch.object(old_model, 'generate', side_effect=generate):
                conversation_thread.start()
                self.assertTrue(entered.wait(2))
                publication_thread.start()
                self.assertTrue(fetched.wait(2))
                # The model may be fetched while a visitor generates, but the
                # active version remains A until that complete turn releases.
                self.assertIs(self.publisher.current_model, old_model)
                release.set()
                conversation_thread.join(6)
                publication_thread.join(6)
            self.assertEqual(failures, [])
            self.assertEqual(outcomes['chat'][0], 200)
            self.assertEqual(outcomes['publish'][0], 200)
            done = outcomes['chat'][1][-1]
            self.assertEqual(done['type'], 'done')
            self.assertEqual(done['model'], old_metadata)
            self.assertEqual(done['assistantMessage']['model'], old_metadata)
            self.assertEqual(self.publisher.current_model.metadata()['seed'], 42)
            stored = store.current(token)
            store.submit(token, '', '', self.publisher.current_model.metadata(), stored['id'], len(stored['messages']))
            self.assertEqual(store.submissions()['submissions'][0]['model'], old_metadata)
            code, status = request(private, '/admin/status')
            self.assertEqual(code, 200)
            self.assertEqual(status['modelPublication'], self.publisher.status())
            self.assertEqual(status['model'], self.publisher.current_model.metadata())
            self.assertTrue(request(private, '/admin/model/sync', body)[1]['alreadyPublished'])
            self.fetch.assert_called_once()
        finally:
            release.set()
            for server in servers:
                server.shutdown()
                server.server_close()
            for thread in threads:
                thread.join(2)


if __name__ == '__main__':
    unittest.main()
