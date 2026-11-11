import http.client
import json
import os
import threading
import unittest
from unittest.mock import patch

from runtime_service import RuntimeServer, RuntimeHandler, RuntimeFailure


class FakeGpu:
    def __init__(self):
        self.calls = []
        self.fail = False

    def request(self, method, path, body=None):
        self.calls.append((method, path, body))
        if self.fail:
            raise RuntimeFailure('Invalid remote reply.')
        return 200, {'step': 1010, 'device': 'cuda'}


class FakeTelemetry:
    def snapshot(self):
        return {'schemaVersion': 1, 'sources': []}


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = RuntimeServer(('127.0.0.1', 0), RuntimeHandler)
        cls.server.telemetry = FakeTelemetry()
        cls.worker = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.worker.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.worker.join()

    def setUp(self):
        self.server.gpu = FakeGpu()

    def request(self, method='GET', path='/api/training/state', body=None, headers=None):
        client = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=3)
        try:
            client.request(method, path, body, headers or {})
            response = client.getresponse()
            raw = response.read()
            return response.status, dict(response.getheaders()), json.loads(raw) if raw else None
        finally:
            client.close()

    def test_training_and_telemetry_are_separate_routes(self):
        status, _, data = self.request()
        self.assertEqual((status, data['device']), (200, 'cuda'))
        self.assertEqual(len(self.server.gpu.calls), 1)
        self.assertEqual(self.request(path='/api/telemetry')[2]['schemaVersion'], 1)
        self.assertEqual(len(self.server.gpu.calls), 1)

    def test_host_and_origin_protect_training(self):
        for headers in ({'Host': 'attacker.example'}, {'Origin': 'https://attacker.example'}):
            self.assertEqual(self.request(headers=headers)[0], 403)
        self.assertEqual(self.server.gpu.calls, [])

    def test_live_token_history_routes_use_existing_guarded_gpu_bridge(self):
        self.assertEqual(self.request(path='/api/training/token-history')[0], 200)
        for action, value in (('token-history-start', {'text': 'the good story'}), ('token-history-stop', {})):
            self.assertEqual(self.request('POST', '/api/training/' + action, json.dumps(value),
                                          {'Content-Type': 'application/json'})[0], 200)
        self.assertEqual([item[:2] for item in self.server.gpu.calls], [
            ('GET', '/api/training/token-history'), ('POST', '/api/training/token-history-start'),
            ('POST', '/api/training/token-history-stop')])

    def test_only_expected_cors_preflight_is_allowed(self):
        headers = {'Origin': 'http://127.0.0.1:8768', 'Access-Control-Request-Method': 'POST',
                   'Access-Control-Request-Headers': 'Content-Type'}
        status, result, _ = self.request('OPTIONS', '/api/training/train', headers=headers)
        self.assertEqual(status, 204)
        self.assertEqual(result['Access-Control-Allow-Origin'], headers['Origin'])
        del headers['Origin']
        self.assertEqual(self.request('OPTIONS', '/api/training/train', headers=headers)[0], 403)

    def test_unknown_paths_and_oversized_or_invalid_bodies_do_not_forward(self):
        self.assertEqual(self.request(path='/api/training/../../anything')[0], 404)
        self.assertEqual(self.request(path='/api/training/state?path=a')[0], 400)
        for body in ('[]', '{', '{"steps":NaN}'):
            self.assertEqual(self.request('POST', '/api/training/train', body,
                                         {'Content-Type': 'application/json'})[0], 400)
        # Reject the advertised size before reading a body. Sending an unread
        # oversized body concurrently with close can produce a Windows TCP reset.
        self.assertEqual(self.request('POST', '/api/training/train', headers={
            'Content-Type': 'application/json', 'Content-Length': '65537'})[0], 413)
        self.assertEqual(self.server.gpu.calls, [])

    def test_post_sent_once_and_uncertain_reply_requires_state_refresh(self):
        self.server.gpu.fail = True
        status, _, data = self.request('POST', '/api/training/train', '{"steps":10}',
                                       {'Content-Type': 'application/json'})
        self.assertEqual(status, 503)
        self.assertIn('may have completed', data['error']['message'])
        self.assertEqual(len(self.server.gpu.calls), 1)
        self.assertEqual(self.request(path='/api/telemetry')[0], 200)

    def test_live_routes_forward_one_request_with_unchanged_payload(self):
        requests = {
            'generate': {'text':'A film', 'model':'live','maxNewTokens':12},
            'transfer-start': {'seed':42,'auxiliaryWeight':.5},
            'transfer-train': {'steps':10,'learningRate':.0003},
            'transfer-predict': {'text':'A good film','branch':'scratch'},
            'save-live': {}, 'restore-live': {},
            'job-start': {'kind':'lm','steps':1000,'learningRate':.0003,'requestId':'65db2951-9d10-4c24-b3ab-29770a1b97f5'},
            'job-pause': {'jobId':'job-fixture'},
        }
        for action, payload in requests.items():
            with self.subTest(action=action):
                raw = json.dumps(payload)
                before = len(self.server.gpu.calls)
                status, headers, data = self.request('POST','/api/training/'+action, raw,
                    {'Content-Type':'application/json','Origin':'http://127.0.0.1:8768'})
                self.assertEqual(status,200)
                self.assertEqual(headers['Access-Control-Allow-Origin'],'http://127.0.0.1:8768')
                self.assertEqual(len(self.server.gpu.calls),before+1)
                self.assertEqual(self.server.gpu.calls[-1],('POST','/api/training/'+action,raw.encode()))
        self.assertEqual(self.request(path='/api/training/transfer-state')[0],200)
        self.assertEqual(self.server.gpu.calls[-1],('GET','/api/training/transfer-state',None))
        self.assertEqual(self.request(path='/api/training/job')[0],200)
        self.assertEqual(self.server.gpu.calls[-1],('GET','/api/training/job',None))

    def test_live_routes_reject_wrong_method_query_and_unlisted_actions(self):
        for path in ('generate','transfer-start','transfer-train','transfer-predict','save-live','restore-live','job-start','job-pause'):
            with self.subTest(path=path):
                self.assertEqual(self.request(path='/api/training/'+path)[0],404)
        for action in ('transfer-state','job','save-as','restore','export','__dict__'):
            self.assertEqual(self.request('POST','/api/training/'+action,'{}',
                                         {'Content-Type':'application/json'})[0],404)
        self.assertEqual(self.request(path='/api/training/transfer-state?path=other.pt')[0],400)
        self.assertEqual(self.request(path='/api/training/job?host=evil')[0],400)
        self.assertEqual(self.request('POST','/api/training/save-live?path=other.pt','{}',
                                     {'Content-Type':'application/json'})[0],400)
        self.assertEqual(self.server.gpu.calls,[])

    def test_audience_export_uses_only_the_fixed_read_route(self):
        self.assertEqual(self.request(path='/api/training/audience-export')[0], 200)
        self.assertEqual(self.server.gpu.calls[-1], ('GET', '/api/training/audience-export', None))
        calls = len(self.server.gpu.calls)
        self.assertEqual(self.request(path='/api/training/audience-export?path=other.pt')[0], 400)
        self.assertEqual(self.request('POST', '/api/training/audience-export', '{}',
                                     {'Content-Type': 'application/json'})[0], 404)
        self.assertEqual(len(self.server.gpu.calls), calls)

    def test_live_operation_lost_acknowledgement_never_replays(self):
        self.server.gpu.fail = True
        for action in ('generate','transfer-start','transfer-train','transfer-predict','save-live','restore-live','job-start','job-pause'):
            with self.subTest(action=action):
                before = len(self.server.gpu.calls)
                status, _, data = self.request('POST','/api/training/'+action,'{}',
                                               {'Content-Type':'application/json'})
                self.assertEqual(status,503)
                self.assertEqual(data['error']['code'],'gpu_unavailable')
                self.assertIn('may have completed',data['error']['message'])
                self.assertEqual(len(self.server.gpu.calls),before+1)


if __name__ == '__main__':
    unittest.main()


class ConfiguredGpuTunnelTests(unittest.TestCase):
    def test_missing_or_invalid_training_destination_never_spawns_ssh(self):
        from runtime_service import GpuTunnel
        for value in ('','-oProxyCommand=bad','host name'):
            with self.subTest(value=value), patch.dict(os.environ,{'GPT_TRAINING_SSH_HOST':value}), patch('runtime_service.socket.socket') as socket, patch('runtime_service.subprocess.Popen') as spawn:
                with self.assertRaisesRegex(RuntimeFailure,'GPT_TRAINING_SSH_HOST'):
                    GpuTunnel().ensure_running()
                socket.assert_not_called();spawn.assert_not_called()

    def test_configured_training_destination_is_used_without_a_private_alias_fallback(self):
        from runtime_service import GpuTunnel
        with patch.dict(os.environ,{'GPT_TRAINING_SSH_HOST':'test-user@gpu.example.com'}), patch('runtime_service.shutil.which',return_value='ssh'), patch('runtime_service.socket.socket') as socket, patch('runtime_service.subprocess.Popen') as spawn:
            socket.return_value.__enter__.return_value.connect_ex.side_effect=[1,0]
            spawn.return_value.poll.return_value=None
            tunnel=GpuTunnel();tunnel.ensure_running()
            self.assertEqual(spawn.call_args.args[0][-1],'test-user@gpu.example.com')
            self.assertEqual(spawn.call_count,1);tunnel.stop()
