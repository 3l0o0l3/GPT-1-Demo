"""Portable HTTP boundaries and real socket streaming; no SSH or trained model."""
import contextlib
import http.client
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from public_server import create_server, main, parse_args
from seminar_gateway import SeminarGatewayError
from training_service import TrainingApiError


class Dispatcher:
    def __init__(self):
        self.calls = []
        self.release = None
        self.incomplete = False
        self.busy = False

    def dispatch(self, method, path, body=None, on_event=None):
        self.calls.append((method, path, body))
        if self.busy:
            raise TrainingApiError(409, 'training_busy', 'Wait for training.')
        if path == '/api/training/generate-stream':
            on_event({'type': 'start', 'text': body['text'], 'source': {'model': 'live', 'step': 3}})
            on_event({'type': 'token', 'index': 0, 'token': {'id': 7, 'piece': '猫', 'p': .3}, 'continuation': '猫'})
            if self.release is not None:
                self.release.wait(3)
            if not self.incomplete:
                on_event({'type': 'done', 'result': {'continuation': '猫', 'stopReason': 'eos'}})
            return None
        return {'step': 0, 'requested': path, 'body': body}


class PublicServerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / 'dist/vendor').mkdir(parents=True)
        (self.root / 'dist/index.html').write_text('<h1>Fixture</h1>')
        (self.root / 'dist/vendor/test.js').write_text('export const fixture=true;')
        (self.root / 'secret.json').write_text('{"private":"outside-dist"}')
        self.dispatcher = Dispatcher()
        self.server = create_server(self.root, port=0, training=self.dispatcher)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
        self.thread.start()
        self.clients = []

    def tearDown(self):
        if self.dispatcher.release is not None:
            self.dispatcher.release.set()
        for client in self.clients:
            client.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)
        self.temporary.cleanup()

    def request(self, path='/', method='GET', body=None, headers=None):
        client = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=2)
        self.clients.append(client)
        supplied = dict(headers or {})
        if body is not None:
            supplied.setdefault('Content-Type', 'application/json')
        client.request(method, path, body=body, headers=supplied)
        return client.getresponse()

    def test_static_root_assets_and_head(self):
        self.assertEqual(self.server.server_address[0], '127.0.0.1')
        self.assertEqual(self.request().read(), b'<h1>Fixture</h1>')
        response = self.request('/vendor/test.js?v=1')
        self.assertEqual(response.status, 200)
        self.assertIn('javascript', response.getheader('Content-Type'))
        response.read()
        response = self.request('/', method='HEAD')
        self.assertEqual(response.status, 200)
        self.assertEqual(response.read(), b'')
        self.assertFalse(self.dispatcher.calls)

    def test_static_traversal_directory_and_private_sources_are_denied(self):
        for path in ('/../secret.json', '/%2e%2e/secret.json', '/vendor/../../secret.json',
                     '/%5c..%5csecret.json', '/vendor/', '/public_server.py', '/.env', '/C:/secret.json'):
            with self.subTest(path=path):
                response = self.request(path)
                self.assertEqual(response.status, 404)
                self.assertNotIn(b'outside-dist', response.read())

    def test_resolved_static_target_cannot_escape_dist(self):
        with patch('public_server.Path.resolve', return_value=self.root / 'secret.json'):
            response = self.request('/vendor/test.js')
            self.assertEqual(response.status, 404)
            response.read()

    def test_host_origin_and_cross_site_fail_before_dispatch(self):
        port = self.server.server_port
        cases = [{'Host': 'attacker.example'}, {'Origin': 'https://attacker.example'},
                 {'Origin': f'http://localhost:{port}'}, {'Sec-Fetch-Site': 'cross-site'}]
        for headers in cases:
            response = self.request('/api/training/train', 'POST', '{}', headers)
            self.assertEqual(response.status, 403)
            response.read()
        self.assertFalse(self.dispatcher.calls)

    def test_duplicate_host_and_content_length_are_rejected(self):
        for repeated in ('Host', 'Content-Length'):
            client = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=2)
            self.clients.append(client)
            client.putrequest('POST', '/api/training/train', skip_host=True)
            host = f'127.0.0.1:{self.server.server_port}'
            client.putheader('Host', host)
            client.putheader('Content-Length', '2')
            client.putheader('Content-Type', 'application/json')
            client.putheader(repeated, host if repeated == 'Host' else '2')
            client.endheaders(b'{}')
            response = client.getresponse()
            self.assertIn(response.status, (400, 403))
            response.read()
        self.assertFalse(self.dispatcher.calls)

    def test_same_origin_and_cli_requests_use_existing_api(self):
        origin = f'http://127.0.0.1:{self.server.server_port}'
        response = self.request('/api/training/train', 'POST', '{"steps":1}', {'Origin': origin})
        self.assertEqual(response.status, 200)
        self.assertEqual(json.load(response)['body'], {'steps': 1})
        self.assertEqual(self.request('/api/training/audience-export').status, 200)

    def test_bad_body_and_options_never_dispatch(self):
        response = self.request('/api/training/train', 'POST', '{}', {'Content-Type': 'text/plain'})
        self.assertEqual(response.status, 415)
        response.read()
        response = self.request('/api/training/train', 'POST', '{')
        self.assertEqual(response.status, 400)
        response.read()
        response = self.request('/api/training/train', 'OPTIONS')
        self.assertEqual(response.status, 403)
        self.assertIsNone(response.getheader('Access-Control-Allow-Origin'))
        self.assertFalse(self.dispatcher.calls)

    def test_stream_flushes_utf8_token_before_generation_finishes(self):
        self.dispatcher.release = threading.Event()
        response = self.request('/api/training/generate-stream', 'POST', '{"text":"the"}')
        self.assertEqual(response.status, 200)
        self.assertIn('application/x-ndjson', response.getheader('Content-Type'))
        self.assertIsNone(response.getheader('Content-Length'))
        self.assertEqual(json.loads(response.readline())['type'], 'start')
        self.assertEqual(json.loads(response.readline())['continuation'], '猫')
        self.assertFalse(self.dispatcher.release.is_set())
        self.dispatcher.release.set()
        self.assertEqual(json.loads(response.readline())['type'], 'done')

    def test_incomplete_stream_gets_one_terminal_error_and_busy_is_json(self):
        self.dispatcher.incomplete = True
        response = self.request('/api/training/generate-stream', 'POST', '{"text":"the"}')
        events = [json.loads(line) for line in response.read().splitlines()]
        self.assertEqual([item['type'] for item in events], ['start', 'token', 'error'])
        self.assertEqual(events[-1]['error']['code'], 'incomplete_stream')
        self.dispatcher.busy = True
        response = self.request('/api/training/generate-stream', 'POST', '{"text":"the"}')
        self.assertEqual(response.status, 409)
        self.assertEqual(json.load(response)['error']['code'], 'training_busy')

    def test_default_seminar_is_disabled_and_telemetry_is_honest(self):
        with patch('seminar_gateway.subprocess.Popen', side_effect=AssertionError('No SSH allowed')):
            response = self.request('/api/seminar/status')
            self.assertEqual(response.status, 503)
            self.assertEqual(json.load(response)['error']['code'], 'seminar_unavailable')
            value = json.load(self.request('/api/telemetry'))
        self.assertEqual(value['sources'][0]['status'], 'unavailable')
        self.assertIsNone(value['sources'][0]['cpu']['utilizationPercent'])

    def test_static_audience_status_adds_only_unavailable_invitation(self):
        self.server.seminar = Mock()
        self.server.seminar.get.return_value = {'model': {'step': 12}, 'publicUrl': 'http://localhost:18870/seminar-gpt/'}
        result = json.load(self.request('/api/seminar/status'))
        self.assertEqual(result['share'], {'state': 'unavailable', 'remainingSeconds': 0,
                                          'expiresAt': None, 'publicUrl': None, 'qrDataUrl': None})
        self.assertEqual(result['model']['step'], 12)
        real = {'state': 'inactive', 'remainingSeconds': 0, 'expiresAt': None, 'publicUrl': None, 'qrDataUrl': None}
        self.server.seminar.get.return_value = {'share': real}
        self.assertEqual(json.load(self.request('/api/seminar/status'))['share'], real)
        self.server.seminar.post.side_effect = SeminarGatewayError(503, 'share_unavailable', 'Temporary sharing is not configured.')
        response = self.request('/api/seminar/share/revoke', 'POST', '{}')
        self.assertEqual(response.status, 503)
        self.assertEqual(json.load(response)['error']['code'], 'share_unavailable')


class StartupTests(unittest.TestCase):
    def test_cli_defaults_and_explicit_acceleration(self):
        args = parse_args([])
        self.assertEqual((args.device, args.precision, args.restore_session, args.compile_training, args.seminar_admin_port),
                         ('cpu', 'fp32', False, False, None))
        self.assertEqual(parse_args(['--device', 'cuda:0', '--precision', 'fp16', '--compile']).device, 'cuda:0')
        self.assertTrue(parse_args(['--restore']).restore_session)
        with contextlib.redirect_stderr(io.StringIO()):
            for flags in (['--precision', 'fp16'], ['--compile'], ['--device', 'mps'], ['--port', '0'], ['--seminar-admin-port', '18876']):
                with self.assertRaises(SystemExit):
                    parse_args(flags)

    def test_startup_does_not_restore_or_load_pretrained_without_option(self):
        for restore in (False, True):
            with patch('public_server.LocalTrainingLab') as factory, patch('public_server.create_server') as create:
                factory.return_value.device = 'cpu'
                factory.return_value.state.return_value = {'step': 0}
                factory.return_value.metadata.return_value = {'execution': {}}
                main(['--restore-session'] if restore else [])
                self.assertEqual(factory.return_value.restore_live.call_count, int(restore))
                factory.return_value.reset.assert_not_called()
                create.return_value.serve_forever.assert_called_once()
                create.return_value.server_close.assert_called_once()

    def test_explicit_local_admin_constructs_no_ssh_tunnel(self):
        with tempfile.TemporaryDirectory() as root, patch('seminar_gateway.ShowcaseTunnel', side_effect=AssertionError('No SSH allowed')):
            server = create_server(root, port=0, seminar_admin_port=18871)
            try:
                self.assertEqual(server.seminar.tunnel.port, 18871)
                server.seminar.tunnel.ensure_running()
            finally:
                server.server_close()


if __name__ == '__main__':
    unittest.main()
