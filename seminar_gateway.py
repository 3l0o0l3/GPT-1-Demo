"""Fixed presenter inbox and invitation relay through an owned SSH tunnel."""
import atexit
import http.client
import json
import os
import re
import uuid
import shutil
import socket
import subprocess
import threading
import time
from urllib.parse import parse_qs, urlsplit


class SeminarGatewayError(Exception):
    def __init__(self, status, code, message):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


class ShowcaseTunnel:
    def __init__(self, port=18872):
        self.port, self.process, self.closing = port, None, False
        self.lock = threading.Lock()
        atexit.register(self.stop)

    def ensure_running(self):
        with self.lock:
            if self.closing:
                raise OSError('The local gateway is stopping.')
            if self.process is not None and self.process.poll() is None:
                return
            host = os.environ.get('GPT_AUDIENCE_SSH_HOST', '').strip()
            if not host:
                raise OSError('Set GPT_AUDIENCE_SSH_HOST before connecting the audience service.')
            if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@:\[\]-]{0,254}', host):
                raise OSError('GPT_AUDIENCE_SSH_HOST must be a single SSH destination, without options or whitespace.')
            with socket.socket() as probe:
                if probe.connect_ex(('127.0.0.1', self.port)) == 0:
                    raise OSError('The private presenter tunnel port is already in use.')
            ssh = shutil.which('ssh')
            if not ssh:
                raise OSError('OpenSSH is unavailable.')
            command = [ssh,'-N','-T','-o','BatchMode=yes','-o','ConnectTimeout=6',
                       '-o','ExitOnForwardFailure=yes','-o','ServerAliveInterval=10',
                       '-o','ServerAliveCountMax=2','-L',
                       f'127.0.0.1:{self.port}:127.0.0.1:18871',host]
            options = {'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {}
            self.process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                            stderr=subprocess.DEVNULL, **options)
            deadline = time.monotonic()+8
            while time.monotonic()<deadline:
                if self.process.poll() is not None:
                    raise OSError('SSH could not connect to the configured audience host.')
                with socket.socket() as probe:
                    probe.settimeout(.2)
                    if probe.connect_ex(('127.0.0.1',self.port)) == 0:
                        return
                time.sleep(.1)
            self._stop()
            raise OSError('The private presenter tunnel timed out.')

    def _stop(self):
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
            self._stop()


class SeminarGateway:
    def __init__(self, tunnel=None):
        self.tunnel = tunnel or ShowcaseTunnel()

    @staticmethod
    def route(url):
        parsed = urlsplit(url)
        if parsed.scheme or parsed.netloc or parsed.fragment:
            raise SeminarGatewayError(400, 'invalid_input', 'Only the configured seminar inbox is supported.')
        if parsed.path == '/api/seminar/status' and not parsed.query:
            return '/admin/status'
        if parsed.path == '/api/seminar/submissions':
            q = parse_qs(parsed.query, keep_blank_values=True)
            if set(q)-{'after','limit'} or any(len(v)!=1 for v in q.values()):
                raise SeminarGatewayError(400,'invalid_input','Invalid inbox pagination.')
            try:
                after, limit = int(q.get('after',['0'])[0]), int(q.get('limit',['100'])[0])
                if after<0 or not 1<=limit<=100:
                    raise ValueError()
            except ValueError:
                raise SeminarGatewayError(400,'invalid_input','Invalid inbox pagination.')
            return f'/admin/submissions?after={after}&limit={limit}'
        raise SeminarGatewayError(404,'not_found','Unknown seminar route.')

    def get(self, path):
        route = self.route(path)  # Validate before any SSH connection.
        return self._request('GET', route)

    def post(self, path, body):
        if path not in ('/api/seminar/share', '/api/seminar/share/revoke', '/api/seminar/submissions/clear', '/api/seminar/model/sync'):
            raise SeminarGatewayError(404, 'not_found', 'Unknown seminar action.')
        if not isinstance(body, dict):
            raise SeminarGatewayError(400, 'invalid_input', 'Send a JSON object.')
        if path == '/api/seminar/model/sync':
            if set(body) != {'requestId', 'expectedReleaseId'} or not isinstance(body['requestId'], str) or not isinstance(body['expectedReleaseId'], str) or not re.fullmatch(r'[0-9a-f]{64}', body['expectedReleaseId']):
                raise SeminarGatewayError(400, 'invalid_input', 'Send a request ID and the current release ID.')
            try:
                if str(uuid.UUID(body['requestId'])) != body['requestId'].lower():
                    raise ValueError()
            except ValueError:
                raise SeminarGatewayError(400, 'invalid_input', 'Send a canonical UUID request ID.')
        elif path == '/api/seminar/submissions/clear':
            if set(body) != {'throughCursor'} or type(body['throughCursor']) is not int or not 0 <= body['throughCursor'] <= 9007199254740991:
                raise SeminarGatewayError(400, 'invalid_input', 'Choose a valid confirmed submission cursor.')
        elif path == '/api/seminar/share':
            if set(body) != {'durationMinutes'} or type(body['durationMinutes']) is not int or body['durationMinutes'] not in (15,30,60,120,240):
                raise SeminarGatewayError(400, 'invalid_input', 'Choose 15, 30, 60, 120 or 240 minutes.')
        elif body:
            raise SeminarGatewayError(400, 'invalid_input', 'The revoke action takes no fields.')
        return self._request('POST', path.replace('/api/seminar/', '/admin/', 1), body)

    def _request(self, method, route, body=None):
        connection = None
        try:
            self.tunnel.ensure_running()
            connection = http.client.HTTPConnection('127.0.0.1', self.tunnel.port, timeout=45 if method == 'POST' and route == '/admin/model/sync' else 12)
            headers = {'Host':'127.0.0.1:18871'}
            if body is not None:
                headers['Content-Type'] = 'application/json'
            connection.request(method, route, json.dumps(body) if body is not None else None, headers=headers)
            response = connection.getresponse()
            raw = response.read(16*1024*1024+1)
            if len(raw)>16*1024*1024:
                raise ValueError('Invalid presenter response.')
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError('Invalid presenter response.')
            if response.status != 200:
                error = value.get('error', {})
                if response.status in (400,403,404,409,410,429,503) and isinstance(error.get('code'),str) and isinstance(error.get('message'),str):
                    raise SeminarGatewayError(response.status, error['code'], error['message'])
                raise ValueError('Invalid presenter response.')
            return value
        except (OSError, ValueError, http.client.HTTPException) as exc:
            message = ('Could not confirm model sync. Refresh status before trying again.' if method == 'POST' and route == '/admin/model/sync' else
                       'Could not confirm the inbox or sharing change. Refresh status before trying again.' if method == 'POST' else
                       'Cannot reach the configured seminar inbox. Your local training is unaffected; try Refresh later.')
            raise SeminarGatewayError(503,'seminar_unavailable',message) from exc
        finally:
            if connection is not None:
                connection.close()
