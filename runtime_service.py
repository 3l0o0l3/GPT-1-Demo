"""Loopback resource monitor and fixed SSH proxy for a configured remote CUDA trainer."""
from __future__ import annotations

import atexit
from contextlib import contextmanager
import http.client
import json
import math
import os
import re
import select
import shutil
import signal
import socket
import subprocess
import threading
import time
from http.server import ThreadingHTTPServer
from urllib.parse import urlsplit

from telemetry_service import TelemetryCache, TelemetryHandler, ORIGINS

APP_ID = 'gpt-home-runtime-v1'
REMOTE_APP = 'gpt-gpu-training-home-v1'
PORT = 8769
TUNNEL_PORT = 18768
MAX_BODY = 65536
MAX_REPLY = 16 * 1024 * 1024
MAX_STREAM_LINE = 1024 * 1024
MAX_STREAM_BYTES = 64 * 1024 * 1024
GET_ROUTES = {'meta', 'state', 'recordings', 'examples', 'health', 'transfer-state', 'job', 'token-history', 'audience-export'}
POST_ROUTES = {'inspect', 'preview', 'commit', 'train', 'reset', 'predict', 'generate',
               'generate-stream', 'transfer-start', 'transfer-train', 'transfer-predict',
               'save-live', 'restore-live', 'job-start', 'job-pause', 'token-history-start', 'token-history-stop'}


class RuntimeFailure(Exception):
    pass


class GpuResponseStream:
    """Own the transport so a browser abort can interrupt a blocked upstream read."""
    def __init__(self, response, transport):
        self.response, self.transport = response, transport

    def abort(self):
        if self.transport is not None:
            try:
                self.transport.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


class GpuTunnel:
    def __init__(self):
        self.process = None
        self.lock = threading.Lock()
        self.closing = False
        atexit.register(self.stop)

    def ensure_running(self):
        with self.lock:
            if self.closing:
                raise RuntimeFailure('The GPU runtime is shutting down.')
            if self.process is not None and self.process.poll() is None:
                return
            host = os.environ.get('GPT_TRAINING_SSH_HOST', '').strip()
            if not host:
                raise RuntimeFailure('Set GPT_TRAINING_SSH_HOST before connecting the remote GPU trainer.')
            if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@:\[\]-]{0,254}', host):
                raise RuntimeFailure('GPT_TRAINING_SSH_HOST must be a single SSH destination, without options or whitespace.')
            with socket.socket() as probe:
                if probe.connect_ex(('127.0.0.1', TUNNEL_PORT)) == 0:
                    raise RuntimeFailure('The GPU tunnel port is already in use by another process.')
            ssh = shutil.which('ssh')
            if not ssh:
                raise RuntimeFailure('OpenSSH is required for the remote GPU connection.')
            command = [ssh, '-N', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=6',
                       '-o', 'ExitOnForwardFailure=yes', '-o', 'ServerAliveInterval=10',
                       '-o', 'ServerAliveCountMax=2', '-L',
                       f'127.0.0.1:{TUNNEL_PORT}:127.0.0.1:18768', host]
            options = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}
            self.process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **options)
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeFailure('SSH could not connect to the configured training host.')
                with socket.socket() as probe:
                    probe.settimeout(.2)
                    if probe.connect_ex(('127.0.0.1', TUNNEL_PORT)) == 0:
                        return
                time.sleep(.1)
            self._stop_process()
            raise RuntimeFailure('The remote GPU tunnel did not become ready.')

    def _stop_process(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)

    def stop(self):
        with self.lock:
            self.closing = True
            self._stop_process()


class GpuUpstream:
    def __init__(self, tunnel, port=TUNNEL_PORT):
        self.tunnel, self.port = tunnel, port

    def request(self, method, path, body=None, timeout=90):
        # Connection setup can recover; the actual model operation is sent once.
        self.tunnel.ensure_running()
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=timeout)
        try:
            connection.request(method, path, body=body,
                               headers={'Content-Type': 'application/json'} if body is not None else {})
            response = connection.getresponse()
            raw = response.read(MAX_REPLY + 1)
            if len(raw) > MAX_REPLY:
                raise RuntimeFailure('The remote training reply exceeded the size limit.')
            try:
                value = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
            except (ValueError, UnicodeDecodeError) as exc:
                raise RuntimeFailure('The remote training service returned an invalid response.') from exc
            if not isinstance(value, dict):
                raise RuntimeFailure('The remote training response was not an object.')
            return response.status, value
        except (OSError, http.client.HTTPException) as exc:
            raise RuntimeFailure('Connection to the remote GPU trainer was lost. An interrupted update may have completed; use Retry to refresh the model state before continuing.') from exc
        finally:
            connection.close()

    @contextmanager
    def stream(self, path, body, timeout=90, on_open=None):
        """Send exactly once; keep ownership even when HTTP/1.0 detaches the socket."""
        self.tunnel.ensure_running()
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=timeout)
        response = transport = None
        try:
            connection.connect()
            transport = connection.sock
            opened = GpuResponseStream(None, transport)
            if on_open is not None:
                on_open(opened)
            connection.request('POST', path, body=body, headers={'Content-Type': 'application/json'})
            response = connection.getresponse()
            opened.response = response
            yield opened
        except (OSError, http.client.HTTPException) as exc:
            raise RuntimeFailure('Connection to the remote generation stream was lost. The request was not replayed.') from exc
        finally:
            GpuResponseStream(response, transport).abort()
            if response is not None:
                response.close()
            connection.close()


def stream_event(raw):
    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError('Non-finite event value.')
        return result
    try:
        value = json.loads(raw.decode('utf-8'), parse_float=finite_float,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    except (ValueError, UnicodeDecodeError) as exc:
        raise RuntimeFailure('The remote generation stream returned invalid JSON or UTF-8.') from exc
    if not isinstance(value, dict):
        raise RuntimeFailure('A generation event must be a JSON object.')
    return value


def stream_context(value):
    """Validate declared context metadata without tokenizing or running a model."""
    if (not isinstance(value, dict) or value.get('mode') not in ('fixed', 'sliding') or
            type(value.get('modelContext')) is not int or value['modelContext'] != 128 or
            type(value.get('windowSize')) is not int or not 1 <= value['windowSize'] <= 128 or
            type(value.get('promptTokens')) is not int or value['promptTokens'] < 1):
        raise RuntimeFailure('The remote generation stream returned invalid context metadata.')
    if value['mode'] == 'fixed' and (value['windowSize'] != 128 or value['promptTokens'] >= 128):
        raise RuntimeFailure('The remote generation stream returned inconsistent fixed context metadata.')
    return value


def stream_window(value, context, index):
    if (not isinstance(value, dict) or type(value.get('inputTokens')) is not int or
            not 1 <= value['inputTokens'] <= context['windowSize'] or
            type(value.get('droppedTokens')) is not int or value['droppedTokens'] < 0 or
            value['inputTokens'] + value['droppedTokens'] != context['promptTokens'] + index or
            (context['mode'] == 'fixed' and value['droppedTokens'] != 0)):
        raise RuntimeFailure('The remote generation stream returned inconsistent token window metadata.')
    return value


def stream_token_valid(token):
    return (isinstance(token, dict) and type(token.get('id')) is int and token['id'] >= 0 and
            isinstance(token.get('piece'), str) and type(token.get('p')) in (int, float) and
            0 <= token['p'] <= 1)


class RuntimeHandler(TelemetryHandler):
    server_version = 'HomeTrainingRuntime/1'

    def _reject_boundary(self):
        if self._boundary():
            return False
        self._json(403, {'error': {'code': 'forbidden', 'message': 'Only the local demo may access this runtime.'}})
        return True

    def _forward(self, method, body=None):
        parts = urlsplit(self.path)
        action = parts.path.removeprefix('/api/training/')
        allowed = GET_ROUTES if method == 'GET' else POST_ROUTES
        if not parts.path.startswith('/api/training/') or action not in allowed:
            return self._json(404, {'error': {'code': 'not_found', 'message': 'Unknown GPU training route.'}})
        path = '/api/health' if action == 'health' else self.path
        if parts.fragment or (parts.query and action != 'examples'):
            return self._json(400, {'error': {'code': 'invalid_input', 'message': 'Unsupported query parameters.'}})
        if action == 'generate-stream':
            return self._forward_stream(path, body)
        try:
            status, payload = self.server.gpu.request(method, path, body)
            self._json(status, payload)
        except RuntimeFailure as exc:
            message = str(exc)
            if method == 'POST' and 'may have completed' not in message:
                message += ' An interrupted update may have completed; use Retry to refresh the actual model state before continuing.'
            self._json(503, {'error': {'code': 'gpu_unavailable', 'message': message}})

    def _stream_headers(self):
        self.close_connection = True
        self.send_response(200)
        self.send_header('Content-Type', 'application/x-ndjson; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Connection', 'close')
        origin = self.headers.get('Origin')
        if origin in ORIGINS:
            self.send_header('Access-Control-Allow-Origin', origin)
        self.send_header('Vary', 'Origin')
        self.end_headers()

    def _forward_stream(self, path, body):
        started = False
        stopped, disconnected = threading.Event(), threading.Event()
        watcher = None
        def observe_connection(upstream):
            nonlocal watcher
            def watch_client():
                # Observe before sending POST / waiting for response headers too.
                while not stopped.is_set():
                    try:
                        ready, _, _ = select.select([self.connection], [], [], .1)
                        if ready and not self.connection.recv(1, socket.MSG_PEEK):
                            disconnected.set()
                            upstream.abort()
                            return
                        if ready:
                            stopped.wait(.05)  # Ignore unread pipelined bytes.
                    except (OSError, ValueError):
                        disconnected.set()
                        upstream.abort()
                        return
            watcher = threading.Thread(target=watch_client, daemon=True)
            watcher.start()
        try:
            with self.server.gpu.stream(path, body, on_open=observe_connection) as upstream:
                response = upstream.response
                content_type = response.getheader('Content-Type', '').split(';')[0].strip().lower()
                if response.status != 200:
                    raw = response.read(MAX_REPLY + 1)
                    if len(raw) > MAX_REPLY or response.status < 400 or content_type != 'application/json':
                        raise RuntimeFailure('The remote generation stream returned an invalid error response.')
                    payload = stream_event(raw)
                    if not isinstance(payload.get('error'), dict):
                        raise RuntimeFailure('The remote generation stream returned an invalid error response.')
                    return self._json(response.status, payload)
                if content_type != 'application/x-ndjson':
                    raise RuntimeFailure('The remote service did not return a generation stream.')
                if (response.getheader('Content-Length') is not None or
                        response.getheader('Transfer-Encoding') is not None or not response.will_close):
                    raise RuntimeFailure('The remote generation stream must use connection-close framing.')
                total, count, max_tokens, terminal = 0, 0, None, None
                context, last_window, continuation, tokens = None, None, '', []
                while True:
                    raw = response.readline(MAX_STREAM_LINE + 1)
                    if disconnected.is_set():
                        return
                    total += len(raw)
                    if not raw:
                        if terminal is not None:
                            # EOF follows the remote handler's lock-release finally.
                            # Hold only this final line, never the preceding tokens.
                            self.wfile.write(terminal)
                            self.wfile.flush()
                            return
                        raise RuntimeFailure('The remote generation stream ended before a done or error event.')
                    if len(raw) > MAX_STREAM_LINE or total > MAX_STREAM_BYTES:
                        raise RuntimeFailure('The remote generation stream exceeded its size limit.')
                    if terminal is not None:
                        raise RuntimeFailure('The remote generation stream returned data after its terminal event.')
                    if not raw.endswith(b'\n'):
                        raise RuntimeFailure('The remote generation stream ended with an incomplete event.')
                    event = stream_event(raw)
                    kind = event.get('type')
                    if not started:
                        max_tokens = event.get('maxNewTokens')
                        if (kind != 'start' or type(max_tokens) is not int or not 1 <= max_tokens <= 1000 or
                                not isinstance(event.get('text'), str) or
                                not isinstance(event.get('source'), dict) or
                                not isinstance(event.get('sampling'), dict)):
                            raise RuntimeFailure('The first generation event was not a valid start event.')
                        if 'context' in event:
                            context = stream_context(event['context'])
                        self._stream_headers()
                        started = True
                    elif kind == 'token':
                        token = event.get('token')
                        if (type(event.get('index')) is not int or event['index'] != count or count >= max_tokens or
                                not stream_token_valid(token) or
                                not isinstance(event.get('continuation'), str)):
                            raise RuntimeFailure('The remote generation stream returned an invalid or out-of-order token.')
                        if context is not None:
                            last_window = stream_window(event.get('window'), context, count)
                        elif 'window' in event:
                            raise RuntimeFailure('Token window metadata requires matching start context metadata.')
                        tokens.append(token)
                        continuation = event['continuation']
                        count += 1
                    elif kind == 'done':
                        result = event.get('result')
                        if (not isinstance(result, dict) or not isinstance(result.get('tokens'), list) or
                                not all(stream_token_valid(token) for token in result['tokens']) or
                                result['tokens'] != tokens or result.get('continuation') != continuation):
                            raise RuntimeFailure('The remote generation completion does not match its streamed tokens.')
                        if context is not None:
                            final_context = stream_context(result.get('context'))
                            final_window = (stream_window(result.get('window'), context, count - 1)
                                            if count else result.get('window'))
                            if final_context != context or final_window != last_window:
                                raise RuntimeFailure('The remote generation completion changed its context or final window.')
                        elif 'context' in result or 'window' in result:
                            raise RuntimeFailure('Completion context metadata requires matching start context metadata.')
                    elif kind == 'error':
                        error = event.get('error')
                        if (not isinstance(error, dict) or not isinstance(error.get('code'), str) or
                                not isinstance(error.get('message'), str)):
                            raise RuntimeFailure('The remote generation stream returned an invalid error event.')
                    else:
                        raise RuntimeFailure('The remote generation stream returned an unexpected event.')
                    if kind in ('done', 'error'):
                        terminal = raw
                    else:
                        self.wfile.write(raw)
                        self.wfile.flush()
        except (RuntimeFailure, OSError, http.client.HTTPException) as exc:
            if disconnected.is_set():
                return
            message = str(exc) if isinstance(exc, RuntimeFailure) else 'The generation connection was interrupted.'
            payload = {'error': {'code': 'generation_stream_failed', 'message': message}}
            try:
                if started:
                    self.wfile.write((json.dumps({'type': 'error', **payload}) + '\n').encode('utf-8'))
                    self.wfile.flush()
                else:
                    self._json(503, payload)
            except OSError:
                pass  # Browser already stopped; the upstream context still closes.
        finally:
            stopped.set()
            if watcher is not None:
                watcher.join(timeout=.3)

    def do_GET(self):
        if self._reject_boundary():
            return
        if self.path == '/api/health':
            return self._json(200, {'app': APP_ID, 'ok': True, 'trainingTarget': 'remote / CUDA'})
        if self.path.startswith('/api/training/'):
            return self._forward('GET')
        super().do_GET()

    def do_POST(self):
        if self._reject_boundary():
            return
        if self.headers.get('Transfer-Encoding') or len(self.headers.get_all('Content-Length', [])) != 1:
            return self._json(400, {'error': {'code': 'invalid_input', 'message': 'A single Content-Length is required.'}})
        try:
            length = int(self.headers['Content-Length'])
            if not 1 <= length <= MAX_BODY:
                return self._json(413, {'error': {'code': 'too_large', 'message': 'Request size exceeds the training limit.'}})
            if self.headers.get('Content-Type', '').split(';')[0].strip().lower() != 'application/json':
                return self._json(415, {'error': {'code': 'invalid_input', 'message': 'Use JSON.'}})
            body = self.rfile.read(length)
            value = json.loads(body, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
            if not isinstance(value, dict):
                raise ValueError()
        except (ValueError, UnicodeDecodeError):
            return self._json(400, {'error': {'code': 'invalid_input', 'message': 'A valid JSON object is required.'}})
        self._forward('POST', body)

    def do_OPTIONS(self):
        if self._reject_boundary():
            return
        headers = {x.strip().lower() for x in self.headers.get('Access-Control-Request-Headers', '').split(',') if x.strip()}
        if (self.headers.get('Origin') not in ORIGINS or
                not self.path.startswith('/api/training/') or
                self.headers.get('Access-Control-Request-Method') not in ('GET', 'POST') or
                headers - {'content-type'}):
            return self._json(403, {'error': {'code': 'forbidden', 'message': 'Unsupported preflight request.'}})
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', self.headers['Origin'])
        self.send_header('Vary', 'Origin')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Access-Control-Max-Age', '600')
        self.send_header('Content-Length', '0')
        self.end_headers()


class RuntimeServer(ThreadingHTTPServer):
    daemon_threads = True

    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(95)
        return connection, address


def main():
    server = RuntimeServer(('127.0.0.1', PORT), RuntimeHandler)
    tunnel = GpuTunnel()
    cache = TelemetryCache(training_host='home')
    server.gpu, server.telemetry = GpuUpstream(tunnel), cache
    try:
        status, health = server.gpu.request('GET', '/api/health', timeout=8)
        if status != 200 or health.get('app') != REMOTE_APP:
            raise RuntimeFailure('The remote endpoint is not the expected remote GPU trainer.')
        cache.start()
        def stop(*_):
            threading.Thread(target=server.shutdown, daemon=True).start()
        signal.signal(signal.SIGINT, stop)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, stop)
        print(f'{APP_ID} ready at http://127.0.0.1:{PORT}', flush=True)
        server.serve_forever(poll_interval=.25)
    finally:
        cache.stop()
        tunnel.stop()
        server.server_close()


if __name__ == '__main__':
    main()
