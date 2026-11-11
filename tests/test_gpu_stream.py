"""Real loopback HTTP fixtures; no SSH, model, GPU, or running demo is used."""
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import copy
import json
import socket
import threading
import unittest
from unittest.mock import patch

from runtime_service import GpuUpstream, RuntimeHandler, RuntimeServer


START = {'type': 'start', 'text': 'A cat', 'maxNewTokens': 2,
         'source': {'model': 'live', 'step': 1000}, 'sampling': {'temperature': .8, 'seed': 42}}
TOKEN = {'type': 'token', 'index': 0, 'token': {'id': 7, 'piece': '猫', 'p': .25}, 'continuation': '猫'}
DONE = {'type': 'done', 'result': {'tokens': [TOKEN['token']], 'continuation': '猫', 'stopReason': 'length'}}


def line(event):
    return (json.dumps(event, ensure_ascii=False) + '\n').encode('utf-8')


class NoTunnel:
    def __init__(self):
        self.calls = 0

    def ensure_running(self):
        self.calls += 1


class SyntheticHome(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def begin(self, content_type='application/x-ndjson; charset=utf-8'):
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Connection', 'close')
        self.end_headers()
        self.close_connection = True

    def emit(self, event):
        self.wfile.write(line(event))
        self.wfile.flush()

    def do_POST(self):
        body = self.rfile.read(int(self.headers['Content-Length']))
        self.server.calls.append((self.path, body))
        try:
            self.server.behavior(self)
        except (OSError, BrokenPipeError):
            pass  # Expected when a proxy deliberately closes a failed stream.


class StreamingProxyTests(unittest.TestCase):
    def setUp(self):
        self.home = ThreadingHTTPServer(('127.0.0.1', 0), SyntheticHome)
        self.home.daemon_threads = True
        self.home.calls = []
        self.home.behavior = self.emit_events([START, TOKEN, DONE])
        self.runtime = RuntimeServer(('127.0.0.1', 0), RuntimeHandler)
        self.tunnel = NoTunnel()
        self.runtime.gpu = GpuUpstream(self.tunnel, port=self.home.server_port)
        self.workers = []
        for server in (self.home, self.runtime):
            worker = threading.Thread(target=lambda s=server: s.serve_forever(poll_interval=.02), daemon=True)
            worker.start()
            self.workers.append(worker)
        self.clients = []

    def tearDown(self):
        for response, client in self.clients:
            response.close()
            client.close()
        for server in (self.runtime, self.home):
            server.shutdown()
            server.server_close()
        for worker in self.workers:
            worker.join(timeout=2)

    @staticmethod
    def emit_events(events):
        def behavior(handler):
            handler.begin()
            for event in events:
                if isinstance(event, bytes):
                    handler.wfile.write(event)
                    handler.wfile.flush()
                else:
                    handler.emit(event)
        return behavior

    def open(self, body='{"text":"A cat"}', headers=None, method='POST', path='/api/training/generate-stream'):
        client = http.client.HTTPConnection('127.0.0.1', self.runtime.server_port, timeout=2)
        supplied = {'Content-Type': 'application/json', 'Origin': 'http://127.0.0.1:8768'}
        if headers:
            supplied.update(headers)
        client.request(method, path, body=body, headers=supplied)
        response = client.getresponse()
        self.clients.append((response, client))
        return response

    def events(self, response):
        return [json.loads(raw) for raw in response.read().splitlines()]

    def test_first_token_arrives_before_home_finishes_and_utf8_chunks_survive(self):
        release = threading.Event()
        waiting = threading.Event()
        def delayed(handler):
            handler.begin()
            handler.emit(START)
            raw = line(TOKEN)
            split = raw.index('猫'.encode()) + 1
            handler.wfile.write(raw[:split])
            handler.wfile.flush()
            handler.wfile.write(raw[split:])
            handler.wfile.flush()
            waiting.set()
            release.wait(4)
            handler.emit(DONE)
        self.home.behavior = delayed
        try:
            response = self.open()
            self.assertEqual(response.status, 200)
            self.assertEqual(response.getheader('Content-Type'), 'application/x-ndjson; charset=utf-8')
            self.assertEqual(response.getheader('Access-Control-Allow-Origin'), 'http://127.0.0.1:8768')
            self.assertEqual(response.getheader('Cache-Control'), 'no-store')
            self.assertEqual(response.getheader('Connection'), 'close')
            self.assertIsNone(response.getheader('Content-Length'))
            self.assertEqual(json.loads(response.readline()), START)
            self.assertEqual(json.loads(response.readline()), TOKEN)
            self.assertTrue(waiting.is_set())
            self.assertFalse(release.is_set(), 'Upstream is deliberately unfinished when the token arrives.')
            release.set()
            self.assertEqual(self.events(response), [DONE])
            self.assertEqual(len(self.home.calls), 1)
        finally:
            release.set()

    def test_done_is_not_delivered_until_upstream_eof_releases_operation(self):
        release, terminal_written, downstream_finished = threading.Event(), threading.Event(), threading.Event()
        def delayed_eof(handler):
            handler.begin()
            handler.emit(START)
            handler.emit(TOKEN)
            handler.emit(DONE)
            terminal_written.set()
            release.wait(4)
        self.home.behavior = delayed_eof
        result = []
        try:
            response = self.open()
            self.assertEqual(json.loads(response.readline()), START)
            self.assertEqual(json.loads(response.readline()), TOKEN)
            self.assertTrue(terminal_written.wait(1))
            def read_terminal():
                result.extend(self.events(response))
                downstream_finished.set()
            reader = threading.Thread(target=read_terminal, daemon=True)
            reader.start()
            self.assertFalse(downstream_finished.wait(.15))
            release.set()
            self.assertTrue(downstream_finished.wait(2))
            reader.join(timeout=1)
            self.assertEqual(result, [DONE])
        finally:
            release.set()

    def test_prestart_json_errors_keep_status_and_cors_without_retry(self):
        payload = {'error': {'code': 'training_busy', 'message': 'An operation is running.'}}
        def rejected(handler):
            raw = json.dumps(payload).encode()
            handler.send_response(409)
            handler.send_header('Content-Type', 'application/json')
            handler.send_header('Content-Length', str(len(raw)))
            handler.end_headers()
            handler.wfile.write(raw)
        self.home.behavior = rejected
        response = self.open()
        self.assertEqual(response.status, 409)
        self.assertTrue(response.getheader('Content-Type').startswith('application/json'))
        self.assertEqual(response.getheader('Access-Control-Allow-Origin'), 'http://127.0.0.1:8768')
        self.assertEqual(json.load(response), payload)
        self.assertEqual(len(self.home.calls), 1)
        self.assertEqual(self.tunnel.calls, 1)

    def test_upstream_error_after_start_is_preserved(self):
        error = {'type': 'error', 'error': {'code': 'generation_failed', 'message': 'Fixture failure.'}}
        self.home.behavior = self.emit_events([START, TOKEN, error])
        response = self.open()
        self.assertEqual(response.status, 200)
        self.assertEqual(self.events(response), [START, TOKEN, error])

    def test_missing_terminal_and_partial_line_fail_explicitly_without_replay(self):
        for tail in ([], [line(TOKEN).rstrip(b'\n')]):
            with self.subTest(tail=tail):
                before = len(self.home.calls)
                self.home.behavior = self.emit_events([START, *tail])
                response = self.open()
                events = self.events(response)
                self.assertEqual(response.status, 200)
                self.assertEqual(events[0], START)
                self.assertEqual(events[-1]['type'], 'error')
                self.assertEqual(events[-1]['error']['code'], 'generation_stream_failed')
                self.assertNotIn('done', [event['type'] for event in events])
                self.assertEqual(len(self.home.calls), before + 1)

    def test_invalid_first_event_is_json_failure_before_stream_headers(self):
        for event in (TOKEN, b'not-json\n', {**START, 'maxNewTokens': True}):
            with self.subTest(event=event):
                self.home.behavior = self.emit_events([event])
                response = self.open()
                self.assertEqual(response.status, 503)
                self.assertTrue(response.getheader('Content-Type').startswith('application/json'))
                self.assertEqual(json.load(response)['error']['code'], 'generation_stream_failed')

    def test_invalid_poststart_events_and_trailing_data_never_report_done(self):
        sequences = [
            [START, {**TOKEN, 'index': 1}],
            [START, TOKEN, TOKEN],
            [START, {'type': 'token', 'index': 0, 'token': {**TOKEN['token'], 'p': True}, 'continuation': 'x'}],
            [START, b'{"type":"token","value":1e999}\n'],
            [START, b'\xff\n'],
            [START, START],
            [START, TOKEN, DONE, TOKEN],
        ]
        for sequence in sequences:
            with self.subTest(sequence=sequence):
                self.home.behavior = self.emit_events(sequence)
                events = self.events(self.open())
                self.assertEqual(events[-1]['type'], 'error')
                self.assertNotIn('done', [event['type'] for event in events])

    def test_line_and_total_byte_limits_are_enforced_incrementally(self):
        cases = [('MAX_STREAM_LINE', 1024, b'x' * 1025 + b'\n'),
                 ('MAX_STREAM_BYTES', len(line(START)) + len(line(TOKEN)) - 1, line(TOKEN))]
        for constant, limit, tail in cases:
            with self.subTest(constant=constant), patch('runtime_service.' + constant, limit):
                self.home.behavior = self.emit_events([START, tail])
                events = self.events(self.open())
                self.assertEqual(events[0], START)
                self.assertEqual(events[-1]['type'], 'error')
                self.assertIn('size limit', events[-1]['error']['message'])

    def test_client_disconnect_closes_stalled_upstream_without_waiting_for_token(self):
        disconnected = threading.Event()
        def stalled(handler):
            handler.begin()
            handler.emit(START)
            handler.emit(TOKEN)
            handler.connection.settimeout(3)
            try:
                if handler.connection.recv(1) == b'':
                    disconnected.set()
            except (ConnectionResetError, ConnectionAbortedError):
                disconnected.set()
        self.home.behavior = stalled
        response = self.open()
        self.assertEqual(json.loads(response.readline()), START)
        self.assertEqual(json.loads(response.readline()), TOKEN)
        response.close()
        self.clients[-1][1].close()
        self.assertTrue(disconnected.wait(2), 'Browser disconnect must shut down the upstream socket even when it sends nothing.')
        self.assertEqual(len(self.home.calls), 1)

    def test_connection_loss_before_headers_returns_json_503_and_never_replays(self):
        def disconnected(handler):
            handler.connection.shutdown(socket.SHUT_RDWR)
            handler.connection.close()
        self.home.behavior = disconnected
        response = self.open()
        self.assertEqual(response.status, 503)
        self.assertIn('not replayed', json.load(response)['error']['message'])
        self.assertEqual(len(self.home.calls), 1)
        self.assertEqual(self.tunnel.calls, 1)

    def test_client_disconnect_before_response_headers_closes_upstream(self):
        received, disconnected = threading.Event(), threading.Event()
        def before_headers(handler):
            received.set()
            handler.connection.settimeout(3)
            try:
                if handler.connection.recv(1) == b'':
                    disconnected.set()
            except (ConnectionResetError, ConnectionAbortedError):
                disconnected.set()
        self.home.behavior = before_headers
        client = http.client.HTTPConnection('127.0.0.1', self.runtime.server_port, timeout=2)
        try:
            client.request('POST', '/api/training/generate-stream', '{"text":"A cat"}',
                           {'Content-Type': 'application/json', 'Origin': 'http://127.0.0.1:8768'})
            self.assertTrue(received.wait(2))
            client.close()
            self.assertTrue(disconnected.wait(2), 'Stop must close upstream before its response headers arrive.')
            self.assertEqual(len(self.home.calls), 1)
        finally:
            client.close()

    def test_stream_route_keeps_host_origin_method_and_body_guards(self):
        for headers in ({'Host': 'attacker.example'}, {'Origin': 'https://attacker.example'}):
            self.assertEqual(self.open(headers=headers).status, 403)
        self.assertEqual(self.open(body='[]').status, 400)
        self.assertEqual(self.open(body='{"temperature":NaN}').status, 400)
        self.assertEqual(self.open(body=None, headers={'Content-Length': '65537'}).status, 413)
        self.assertEqual(self.open(headers={'Content-Type': 'text/plain'}).status, 415)
        self.assertEqual(self.open(method='GET', body=None).status, 404)
        self.assertEqual(self.open(path='/api/training/generate-stream?path=anything').status, 400)
        headers = {'Access-Control-Request-Method': 'POST', 'Access-Control-Request-Headers': 'Content-Type'}
        response = self.open(body=None, method='OPTIONS', headers=headers)
        self.assertEqual(response.status, 204)
        self.assertEqual(response.getheader('Access-Control-Allow-Origin'), 'http://127.0.0.1:8768')
        self.assertEqual(self.home.calls, [])

    @staticmethod
    def generated_events(count, context=None, budget=None, piece='x'):
        start = {**START, 'maxNewTokens': count if budget is None else budget}
        if context is not None:
            start['context'] = context
        events, tokens, final_window = [start], [], None
        for index in range(count):
            token = {'id': index + 7, 'piece': piece, 'p': .25}
            event = {'type': 'token', 'index': index, 'token': token, 'continuation': piece * (index + 1)}
            if context is not None:
                total = context['promptTokens'] + index
                input_tokens = min(total, context['windowSize'])
                final_window = {'inputTokens': input_tokens, 'droppedTokens': total - input_tokens}
                event['window'] = final_window
            events.append(event)
            tokens.append(token)
        result = {'tokens': tokens, 'continuation': piece * count, 'stopReason': 'length'}
        if context is not None:
            result.update({'context': context, 'window': final_window})
        return events + [{'type': 'done', 'result': result}]

    def test_budgets_above_64_through_1000_and_optional_window_metadata_forward(self):
        cases = [(65, None, 'x'),
                 (65, {'mode': 'fixed', 'windowSize': 128, 'modelContext': 128, 'promptTokens': 5}, 'x'),
                 (65, {'mode': 'sliding', 'windowSize': 1, 'modelContext': 128, 'promptTokens': 140}, 'x'),
                 (65, {'mode': 'sliding', 'windowSize': 128, 'modelContext': 128, 'promptTokens': 140}, 'x'),
                 (1000, {'mode': 'sliding', 'windowSize': 64, 'modelContext': 128, 'promptTokens': 140}, 'abcdefghijklmnop')]
        for count, context, piece in cases:
            with self.subTest(count=count, context=context):
                expected = self.generated_events(count, context, piece=piece)
                self.home.behavior = self.emit_events(expected)
                payload = {'text': 'A cat', 'maxNewTokens': count}
                if context is not None:
                    payload.update(slidingWindow=context['mode'] == 'sliding', windowSize=context['windowSize'])
                raw = json.dumps(payload)
                before = len(self.home.calls)
                response = self.open(body=raw)
                self.assertEqual(response.status, 200)
                self.assertEqual(self.events(response), expected)
                self.assertEqual(self.home.calls[before:], [('/api/training/generate-stream', raw.encode())])
                if count == 1000:
                    # This legitimate long stream also exceeds the former 2MiB total cap.
                    self.assertGreater(sum(len(line(event)) for event in expected), 2 * 1024 * 1024)

    def test_start_budget_1001_rejected_and_1001st_token_never_forwarded(self):
        self.home.behavior = self.emit_events([{**START, 'maxNewTokens': 1001}])
        response = self.open()
        self.assertEqual(response.status, 503)
        self.assertEqual(json.load(response)['error']['code'], 'generation_stream_failed')
        context = {'mode': 'sliding', 'windowSize': 128, 'modelContext': 128, 'promptTokens': 2}
        self.home.behavior = self.emit_events(self.generated_events(1001, context, budget=1000))
        events = self.events(self.open())
        self.assertEqual(sum(event['type'] == 'token' for event in events), 1000)
        self.assertEqual(events[-1]['type'], 'error')
        self.assertNotIn('done', [event['type'] for event in events])

    def test_invalid_start_context_is_rejected_before_stream_headers(self):
        valid = {'mode': 'sliding', 'windowSize': 64, 'modelContext': 128, 'promptTokens': 140}
        contexts = [None, {**valid, 'mode': 'unknown'}, {**valid, 'modelContext': 127},
                    {**valid, 'promptTokens': 0}, {**valid, 'promptTokens': True},
                    {**valid, 'mode': 'fixed'},
                    {**valid, 'mode': 'fixed', 'windowSize': 128}]
        contexts.extend({**valid, 'windowSize': value} for value in (0, 129, True, '64', 1.5))
        for context in contexts:
            with self.subTest(context=context):
                self.home.behavior = self.emit_events([{**START, 'context': context}])
                response = self.open()
                self.assertEqual(response.status, 503)
                self.assertTrue(response.getheader('Content-Type').startswith('application/json'))
                self.assertEqual(json.load(response)['error']['code'], 'generation_stream_failed')

    def test_invalid_or_inconsistent_token_windows_fail_after_start(self):
        context = {'mode': 'sliding', 'windowSize': 64, 'modelContext': 128, 'promptTokens': 140}
        start, valid_token, _ = self.generated_events(1, context)
        windows = [None, {'inputTokens': 0, 'droppedTokens': 140},
                   {'inputTokens': 129, 'droppedTokens': 11},
                   {'inputTokens': True, 'droppedTokens': 139},
                   {'inputTokens': 64, 'droppedTokens': True},
                   {'inputTokens': 64, 'droppedTokens': -1},
                   {'inputTokens': 64, 'droppedTokens': 77}]
        for window in windows:
            with self.subTest(window=window):
                self.home.behavior = self.emit_events([start, {**valid_token, 'window': window}])
                events = self.events(self.open())
                self.assertEqual(events[0], start)
                self.assertEqual(events[-1]['type'], 'error')
                self.assertNotIn('token', [event['type'] for event in events])
        self.home.behavior = self.emit_events([START, {**TOKEN, 'window': {'inputTokens': 2, 'droppedTokens': 0}}])
        self.assertEqual(self.events(self.open())[-1]['type'], 'error')

    def test_terminal_tokens_continuation_context_and_window_match_stream(self):
        context = {'mode': 'sliding', 'windowSize': 64, 'modelContext': 128, 'promptTokens': 140}
        expected = self.generated_events(3, context)
        variants = []
        for field, value in (
            ('tokens', []), ('continuation', 'different'), ('context', {**context, 'windowSize': 63}),
            ('window', {'inputTokens': 63, 'droppedTokens': 79}), ('window', None), ('context', None)):
            changed = copy.deepcopy(expected)
            changed[-1]['result'][field] = value
            variants.append(changed)
        changed = copy.deepcopy(expected)
        changed[-1]['result']['tokens'] = copy.deepcopy(changed[-1]['result']['tokens'])
        changed[-1]['result']['tokens'][0]['id'] = True
        variants.append(changed)
        for sequence in variants:
            with self.subTest(result=sequence[-1]['result']):
                self.home.behavior = self.emit_events(sequence)
                events = self.events(self.open())
                self.assertEqual(events[-1]['type'], 'error')
                self.assertNotIn('done', [event['type'] for event in events])

    def test_window_request_validation_errors_from_home_keep_http_status(self):
        # Request semantics remain owned by TrainingService; the proxy forwards
        # its normal validation envelope without an automatic retry.
        payload = {'error': {'code': 'invalid_input', 'message': 'Invalid window or token budget.'}}
        def rejected(handler):
            raw = json.dumps(payload).encode()
            handler.send_response(400)
            handler.send_header('Content-Type', 'application/json')
            handler.send_header('Content-Length', str(len(raw)))
            handler.end_headers()
            handler.wfile.write(raw)
        self.home.behavior = rejected
        for fields in ({'maxNewTokens': 1001}, {'windowSize': 0}, {'windowSize': 129},
                       {'windowSize': True}, {'slidingWindow': 'true'}):
            with self.subTest(fields=fields):
                before = len(self.home.calls)
                response = self.open(body=json.dumps({'text': 'A cat', **fields}))
                self.assertEqual(response.status, 400)
                self.assertEqual(json.load(response), payload)
                self.assertEqual(len(self.home.calls), before + 1)


if __name__ == '__main__':
    unittest.main()
