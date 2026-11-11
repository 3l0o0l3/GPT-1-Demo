import http.client
import json
import os
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seminar_gateway import SeminarGateway, SeminarGatewayError
from local_server import GatewayHandler, GatewayServer


class RouteTests(unittest.TestCase):
    def test_fixed_readonly_routes_and_pagination(self):
        self.assertEqual(SeminarGateway.route('/api/seminar/status'), '/admin/status')
        self.assertEqual(SeminarGateway.route('/api/seminar/submissions?after=21&limit=10'), '/admin/submissions?after=21&limit=10')
        self.assertEqual(SeminarGateway.route('/api/seminar/submissions'), '/admin/submissions?after=0&limit=100')

    def test_arbitrary_destinations_and_bad_queries_never_connect(self):
        tunnel = Mock()
        gateway = SeminarGateway(tunnel)
        for path in ('https://example.com/api/seminar/status', '/api/seminar/train',
                     '/api/seminar/submissions?host=home','/api/seminar/status?x=1',
                     '/api/seminar/submissions?after=-1','/api/seminar/submissions?limit=101',
                     '/api/seminar/submissions?after=1&after=2','/api/seminar/submissions?after=NaN'):
            with self.subTest(path=path), self.assertRaises(SeminarGatewayError):
                gateway.get(path)
        tunnel.ensure_running.assert_not_called()

    def test_connection_failure_is_explicit(self):
        tunnel = Mock()
        tunnel.ensure_running.side_effect = OSError('secret must not reach response')
        with self.assertRaises(SeminarGatewayError) as caught:
            SeminarGateway(tunnel).get('/api/seminar/status')
        self.assertEqual(caught.exception.status, 503)
        self.assertNotIn('secret', str(caught.exception))

    def test_mutation_allowlist_is_validated_before_connection(self):
        tunnel = Mock()
        gateway = SeminarGateway(tunnel)
        for path,body in (('/api/seminar/share?host=evil',{}),('/api/seminar/share',{'durationMinutes':True}),
                          ('/api/seminar/share',{'durationMinutes':1}),('/api/seminar/share/revoke',{'x':1})):
            with self.assertRaises(SeminarGatewayError):
                gateway.post(path,body)
        tunnel.ensure_running.assert_not_called()

    def test_clear_route_accepts_only_bounded_integer_and_exact_fields(self):
        tunnel = Mock()
        gateway = SeminarGateway(tunnel)
        for path in ('/api/seminar/submissions/clear?x=1', 'https://evil.example/api/seminar/submissions/clear', '/api/seminar/submissions/clear/'):
            with self.assertRaises(SeminarGatewayError):
                gateway.post(path, {'throughCursor': 1})
        for body in ({}, None, [], {'throughCursor': True}, {'throughCursor': -1}, {'throughCursor': '1'}, {'throughCursor': 1.0}, {'throughCursor': 9007199254740992}, {'throughCursor': 1, 'all': True}):
            with self.subTest(body=body), self.assertRaises(SeminarGatewayError) as caught:
                gateway.post('/api/seminar/submissions/clear', body)
            self.assertEqual(caught.exception.status, 400)
        tunnel.ensure_running.assert_not_called()
        gateway._request = Mock(return_value={'deletedCount': 0})
        for boundary in (0, 1, 9007199254740991):
            self.assertEqual(gateway.post('/api/seminar/submissions/clear', {'throughCursor': boundary}), {'deletedCount': 0})
            gateway._request.assert_called_with('POST', '/admin/submissions/clear', {'throughCursor': boundary})


    def test_sync_route_accepts_only_request_id_and_release_cas(self):
        tunnel = Mock()
        gateway = SeminarGateway(tunnel)
        valid = {'requestId': '6a07a104-e951-4a0a-934f-00fd984e6e51', 'expectedReleaseId': 'a' * 64}
        for path in ('/api/seminar/model/sync?url=http://evil', '/api/seminar/model/sync/', 'https://evil/api/seminar/model/sync'):
            with self.assertRaises(SeminarGatewayError):
                gateway.post(path, valid)
        for body in (None, {}, {'requestId': valid['requestId']}, {**valid, 'path': '/tmp/weights'}, {**valid, 'url': 'http://evil'}, {**valid, 'requestId': True}, {**valid, 'requestId': 'not-a-uuid'}, {**valid, 'requestId': 'a'*32}, {**valid, 'expectedReleaseId': 'bad'}):
            with self.subTest(body=body), self.assertRaises(SeminarGatewayError):
                gateway.post('/api/seminar/model/sync', body)
        tunnel.ensure_running.assert_not_called()
        gateway._request = Mock(return_value={'alreadyPublished': False})
        self.assertEqual(gateway.post('/api/seminar/model/sync', valid), {'alreadyPublished': False})
        gateway._request.assert_called_once_with('POST', '/admin/model/sync', valid)

    def test_only_model_sync_uses_extended_timeout_and_never_retries(self):
        tunnel = Mock(port=18872)
        response = Mock(status=200)
        response.read.return_value = b'{}'
        connection = Mock()
        connection.getresponse.return_value = response
        with patch('seminar_gateway.http.client.HTTPConnection', return_value=connection) as connect:
            gateway = SeminarGateway(tunnel)
            gateway._request('POST', '/admin/model/sync', {})
            connect.assert_called_with('127.0.0.1', 18872, timeout=45)
            gateway._request('GET', '/admin/status')
            connect.assert_called_with('127.0.0.1', 18872, timeout=12)
            response.status = 409
            response.read.return_value = json.dumps({'error': {'code': 'release_changed', 'message': 'Release changed.'}}).encode()
            with self.assertRaises(SeminarGatewayError) as caught:
                gateway._request('POST', '/admin/model/sync', {})
            self.assertEqual(caught.exception.status, 409)
            connection.getresponse.side_effect = TimeoutError('private /server/path')
            before = connection.request.call_count
            with self.assertRaises(SeminarGatewayError) as caught:
                gateway._request('POST', '/admin/model/sync', {})
            self.assertEqual(connection.request.call_count, before + 1)
            self.assertEqual(caught.exception.status, 503)
            self.assertEqual(caught.exception.message, 'Could not confirm model sync. Refresh status before trying again.')
            self.assertNotIn('/server/path', caught.exception.message)

class LocalBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.server = GatewayServer(('127.0.0.1', 0), GatewayHandler)
        self.server.seminar = Mock()
        self.server.seminar.get.return_value = {'submissionCount':1}
        self.thread = threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def request(self, method='GET', headers=None):
        conn = http.client.HTTPConnection('127.0.0.1',self.server.server_port)
        conn.request(method,'/api/seminar/status',headers=headers or {})
        response = conn.getresponse()
        result = response.status,json.loads(response.read())
        conn.close()
        return result

    def test_local_read_is_wired_to_fixed_gateway(self):
        status, data = self.request()
        self.assertEqual((status,data), (200, {'submissionCount':1}))
        self.server.seminar.get.assert_called_once_with('/api/seminar/status')

    def test_foreign_origin_and_rebinding_host_cannot_read_inbox(self):
        for headers in ({'Origin':'https://evil.example'}, {'Host':'evil.example'}, {'Sec-Fetch-Site':'cross-site'}):
            with self.subTest(headers=headers):
                self.assertEqual(self.request(headers=headers)[0],403)
        self.server.seminar.get.assert_not_called()

    def test_post_cannot_mutate_presenter_route(self):
        self.assertEqual(self.request('POST')[0],404)
        self.server.seminar.get.assert_not_called()

    def test_local_sharing_post_is_wired_and_foreign_origin_blocked(self):
        self.server.seminar.post.return_value={'share':{'state':'active'}}
        for headers,expected in (({},200),({'Origin':'https://evil.example'},403),({'Host':'evil.example'},403)):
            conn=http.client.HTTPConnection('127.0.0.1',self.server.server_port)
            conn.request('POST','/api/seminar/share',json.dumps({'durationMinutes':15}),{'Content-Type':'application/json',**headers})
            response=conn.getresponse()
            self.assertEqual(response.status,expected)
            response.read()
            conn.close()
        self.server.seminar.post.assert_called_once_with('/api/seminar/share',{'durationMinutes':15})

    def test_local_clear_post_is_wired_and_foreign_origin_blocked(self):
        self.server.seminar.post.return_value = {'deletedCount': 0, 'submissionCount': 0, 'lastCursor': 0, 'clearedThrough': 0}
        for headers, expected in (({}, 200), ({'Origin': 'https://evil.example'}, 403), ({'Host': 'evil.example'}, 403), ({'Sec-Fetch-Site': 'cross-site'}, 403)):
            connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port)
            connection.request('POST', '/api/seminar/submissions/clear', json.dumps({'throughCursor': 0}), {'Content-Type': 'application/json', **headers})
            response = connection.getresponse()
            self.assertEqual(response.status, expected)
            response.read()
            connection.close()
        self.server.seminar.post.assert_called_once_with('/api/seminar/submissions/clear', {'throughCursor': 0})


    def test_local_model_sync_is_wired_and_foreign_origin_is_blocked(self):
        body = {'requestId': '6a07a104-e951-4a0a-934f-00fd984e6e51', 'expectedReleaseId': 'a' * 64}
        self.server.seminar.post.return_value = {'alreadyPublished': False}
        for headers, expected in (({}, 200), ({'Origin': 'https://evil.example'}, 403), ({'Host': 'evil.example'}, 403), ({'Sec-Fetch-Site': 'cross-site'}, 403)):
            connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port)
            connection.request('POST', '/api/seminar/model/sync', json.dumps(body), {'Content-Type': 'application/json', **headers})
            response = connection.getresponse()
            self.assertEqual(response.status, expected)
            response.read()
            connection.close()
        self.server.seminar.post.assert_called_once_with('/api/seminar/model/sync', body)


class ConfiguredAudienceTunnelTests(unittest.TestCase):
    def test_missing_or_invalid_audience_destination_never_spawns_ssh(self):
        from seminar_gateway import ShowcaseTunnel
        for value in ('','-oProxyCommand=bad','host name'):
            with self.subTest(value=value), patch.dict(os.environ,{'GPT_AUDIENCE_SSH_HOST':value}), patch('seminar_gateway.socket.socket') as socket, patch('seminar_gateway.subprocess.Popen') as spawn:
                with self.assertRaisesRegex(OSError,'GPT_AUDIENCE_SSH_HOST'):
                    ShowcaseTunnel().ensure_running()
                socket.assert_not_called();spawn.assert_not_called()

    def test_configured_audience_destination_is_used_without_a_private_alias_fallback(self):
        from seminar_gateway import ShowcaseTunnel
        with patch.dict(os.environ,{'GPT_AUDIENCE_SSH_HOST':'test-user@audience.example.com'}), patch('seminar_gateway.shutil.which',return_value='ssh'), patch('seminar_gateway.socket.socket') as socket, patch('seminar_gateway.subprocess.Popen') as spawn:
            socket.return_value.__enter__.return_value.connect_ex.side_effect=[1,0]
            spawn.return_value.poll.return_value=None
            tunnel=ShowcaseTunnel();tunnel.ensure_running()
            self.assertEqual(spawn.call_args.args[0][-1],'test-user@audience.example.com')
            self.assertEqual(spawn.call_count,1);tunnel.stop()
