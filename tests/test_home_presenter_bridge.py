import http.client
import json
import threading
import unittest
from unittest.mock import Mock

from home_presenter_bridge import HomePresenterHandler, HomeTelemetry, APP_ID
from runtime_service import RuntimeServer
from telemetry_probe import empty_source


class HomeBridgeTests(unittest.TestCase):
    def setUp(self):
        self.server = RuntimeServer(('127.0.0.1',0),HomePresenterHandler)
        self.server.gpu = Mock()
        self.server.gpu.request.return_value = (200,{'step':123})
        self.server.telemetry = HomeTelemetry(sampler=lambda:empty_source('home','fixture','Linux',False))
        self.server.telemetry.sample_once()
        self.thread = threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def request(self,path,body=None,headers=None):
        conn = http.client.HTTPConnection('127.0.0.1',self.server.server_port)
        h = {'Content-Type':'application/json',**(headers or {})}
        conn.request('POST' if body is not None else 'GET',path,json.dumps(body) if body is not None else None,h)
        response = conn.getresponse()
        result = response.status,json.loads(response.read())
        conn.close()
        return result

    def test_health_and_home_only_telemetry(self):
        self.assertEqual(self.request('/api/health')[1]['app'],APP_ID)
        code,payload = self.request('/api/telemetry')
        self.assertEqual(code,200)
        self.assertEqual(len(payload['sources']),1)
        self.assertEqual(payload['sources'][0]['id'],'home')
        self.assertNotIn('home',payload['sources'][0]['label'])
        self.assertNotIn('Qwen',payload['sources'][0]['label'])
        self.assertTrue(payload['sources'][0]['isTrainingHost'])
        self.assertEqual(payload['pollIntervalMs'],1000)
        self.server.gpu.request.assert_not_called()

    def test_no_browser_origin_or_rebinding_host(self):
        for headers in ({'Host':'evil.example'},{'Origin':'http://127.0.0.1:8768'},
                        {'Origin':'https://demo.example.com'},{'Origin':''},
                        {'Host':f'localhost:{self.server.server_port}'},{'Sec-Fetch-Site':'cross-site'}):
            self.assertEqual(self.request('/api/training/state',headers=headers)[0],403)
        self.server.gpu.request.assert_not_called()

    def test_only_existing_training_contract_is_forwarded_once(self):
        self.assertEqual(self.request('/api/training/state'),(200,{'step':123}))
        self.server.gpu.request.assert_called_once_with('GET','/api/training/state',None)
        for path in ('/api/shell','/api/training/arbitrary','/api/training/state?host=evil'):
            self.assertIn(self.request(path)[0],(400,404))
        self.server.gpu.request.reset_mock()
        self.assertEqual(self.request('/api/training/train',{'steps':10})[0],200)
        self.assertEqual(self.server.gpu.request.call_count,1)

    def test_sensor_failure_retains_sample_without_false_freshness(self):
        cache = self.server.telemetry
        cache.sampler = Mock(side_effect=RuntimeError('private details'))
        cache.sample_once()
        snapshot = cache.snapshot()
        self.assertEqual(snapshot['sources'][0]['status'],'stale')
        self.assertNotIn('private details',json.dumps(snapshot))


if __name__=='__main__':
    unittest.main()
