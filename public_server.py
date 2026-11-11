"""Portable, same-origin GPT Training Lab on 127.0.0.1.

This entry point owns process configuration and the HTTP/static-file boundary.
TrainingService owns API validation and serialized model access; TrainingLab owns
weights and optimizers. GPUTrainingHandler supplies the unchanged NDJSON stream.
No SSH connection, hosted login, pretrained weight download or restore is implicit.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
from pathlib import Path
import platform
import re
from urllib.parse import unquote, urlsplit

import torch

from remote_training_server import (GPUTrainingHandler, GPUTrainingServer,
                                    MAX_BODY_BYTES, configure_compilation_environment)
from seminar_gateway import SeminarGateway, SeminarGatewayError
from telemetry_probe import empty_source
from training_engine import TrainingLab
from training_service import TrainingApiError, TrainingService


ROOT = Path(__file__).resolve().parent
APP_ID = 'gpt-training-lab-local-v1'
PERSISTENCE = ('Starts with random weights unless --restore-session is explicitly selected. '
               'Save session keeps weights, Adam, RNG, training history and the Transfer pair. '
               'Unsaved updates stay in this process only.')
STATIC_TYPES = {'.html', '.js', '.css', '.json', '.svg', '.png', '.jpg', '.jpeg',
                '.webp', '.ico', '.woff', '.woff2', '.ttf', '.otf'}


class LocalTrainingLab(TrainingLab):
    def metadata(self):
        result = super().metadata()
        result['runtime'] = {'location': 'local', 'device': str(self.device),
                             'trainingHost': 'Local machine', 'persistence': PERSISTENCE,
                             'weightsDevice': str(next(self.model.parameters()).device),
                             'torchVersion': torch.__version__, 'cudaVersion': torch.version.cuda}
        return result


class FixedLocalConnection:
    """Only an explicitly configured local audience admin; never launches SSH."""
    port = 18871

    def ensure_running(self):
        pass


def unavailable_telemetry():
    source = empty_source('local', 'Local training process', platform.system(), True)
    source['status'] = 'unavailable'
    source['notes'] = ['This portable server does not collect hardware telemetry.']
    return {'schemaVersion': 1, 'pollIntervalMs': 1000, 'sources': [source]}


class PublicHandler(GPUTrainingHandler):
    server_version = 'GPTTrainingLab/1'

    def setup(self):
        super().setup()
        self.connection.settimeout(30)

    def _boundary(self):
        port = self.server.server_port
        hosts = self.headers.get_all('Host', [])
        origins = self.headers.get_all('Origin', [])
        if len(hosts) != 1 or hosts[0] not in (f'127.0.0.1:{port}', f'localhost:{port}'):
            raise TrainingApiError(403, 'invalid_host', 'Use this server\'s loopback URL.')
        if origins and (len(origins) != 1 or origins[0] != 'http://' + hosts[0]):
            raise TrainingApiError(403, 'invalid_origin', 'Requests must use the same origin as this page.')
        if self.headers.get('Sec-Fetch-Site') == 'cross-site':
            raise TrainingApiError(403, 'invalid_origin', 'Cross-site requests are not supported.')
        if self.headers.get_all('Transfer-Encoding', []):
            raise TrainingApiError(400, 'invalid_request', 'Chunked request bodies are unsupported.')
        if self.command == 'POST' and len(self.headers.get_all('Content-Length', [])) != 1:
            raise TrainingApiError(400, 'invalid_request', 'Send one Content-Length header.')

    def _static(self, method):
        parsed = urlsplit(self.path)
        if parsed.scheme or parsed.netloc or parsed.fragment:
            raise TrainingApiError(404, 'not_found', 'Unknown local resource.')
        name = unquote(parsed.path)
        if name == '/':
            name = '/index.html'
        parts = name.lstrip('/').split('/')
        if (not name.startswith('/') or any(not part or part.startswith('.') for part in parts)
                or any(char in name for char in ('\\', '\x00', ':'))):
            raise TrainingApiError(404, 'not_found', 'Unknown local resource.')
        target = (self.server.static_root / '/'.join(parts)).resolve()
        if (not target.is_relative_to(self.server.static_root) or not target.is_file()
                or target.suffix.lower() not in STATIC_TYPES):
            raise TrainingApiError(404, 'not_found', 'Unknown local resource.')
        body = target.read_bytes()
        content_type = mimetypes.guess_type(target.name)[0] or 'application/octet-stream'
        if target.suffix == '.js':
            content_type = 'text/javascript'
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        if method != 'HEAD':
            self.wfile.write(body)

    def _seminar(self, method):
        if self.server.seminar is None:
            raise TrainingApiError(503, 'seminar_unavailable',
                                   'Audience features are disabled. Configure --seminar-admin-port 18871 explicitly.')
        if method == 'GET':
            value = self.server.seminar.get(self.path)
            if self.path == '/api/seminar/status' and 'share' not in value:
                value = {**value, 'share': {'state': 'unavailable', 'remainingSeconds': 0,
                                          'expiresAt': None, 'publicUrl': None, 'qrDataUrl': None}}
            return self._json(200, value)
        if method != 'POST':
            raise TrainingApiError(405, 'method_not_allowed', 'Use GET or POST.')
        if self.headers.get('Content-Type', '').split(';')[0].strip() != 'application/json':
            raise TrainingApiError(415, 'invalid_content_type', 'Send application/json.')
        try:
            length = int(self.headers.get('Content-Length', '0'))
        except ValueError as error:
            raise TrainingApiError(400, 'invalid_input', 'Invalid Content-Length.') from error
        if not 0 < length <= MAX_BODY_BYTES:
            raise TrainingApiError(413, 'body_too_large', 'JSON body must contain 1–65536 bytes.')
        try:
            body = json.loads(self.rfile.read(length).decode('utf-8'))
        except (ValueError, UnicodeError) as error:
            raise TrainingApiError(400, 'invalid_json', 'Malformed JSON body.') from error
        return self._json(200, self.server.seminar.post(self.path, body))

    def _run(self, method):
        self._stream_started = self._stream_finished = False
        try:
            self._boundary()
            if self.path.startswith('/api/training/') or self.path == '/api/health':
                if method not in ('GET', 'POST'):
                    raise TrainingApiError(405, 'method_not_allowed', 'Use GET or POST.')
                return super()._run(method)
            if self.path == '/api/telemetry' and method == 'GET':
                return self._json(200, unavailable_telemetry())
            if self.path.startswith('/api/seminar/'):
                return self._seminar(method)
            if method not in ('GET', 'HEAD'):
                raise TrainingApiError(405, 'method_not_allowed', 'Use GET or HEAD for static resources.')
            return self._static(method)
        except (TrainingApiError, SeminarGatewayError) as error:
            self._request_error(error.status, error.code, error.message)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError):
            self.close_connection = True

    def do_HEAD(self):
        self._run('HEAD')

    def do_OPTIONS(self):
        self._json(403, {'error': {'code': 'invalid_origin', 'message': 'Cross-origin access is disabled.'}})


def create_server(root=ROOT, *, port=8768, training=None, health=None, seminar_admin_port=None):
    """Assembly seam: inject a TrainingService-compatible dispatcher for tests."""
    if seminar_admin_port not in (None, 18871):
        raise ValueError('The fixed local audience admin port is 18871.')
    root = Path(root).resolve()
    server = GPUTrainingServer(('127.0.0.1', port), PublicHandler)
    server.static_root = (root / 'dist').resolve()
    server.training = training if training is not None else TrainingService(root)
    server.health = health or {'app': APP_ID, 'ok': True, 'device': 'cpu'}
    server.seminar = None if seminar_admin_port is None else SeminarGateway(FixedLocalConnection())
    return server


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8768)
    parser.add_argument('--device', default='cpu', help='cpu (default) or cuda:0')
    parser.add_argument('--precision', choices=('fp32', 'fp16'), default='fp32')
    parser.add_argument('--compile', action='store_true', dest='compile_training')
    parser.add_argument('--restore-session', '--restore', action='store_true', help='Explicitly restore the saved live/session.pt')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--artifact-dir', type=Path, default=ROOT / '.training')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seminar-admin-port', type=int, choices=(18871,), default=None)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error('--port must be between 1 and 65535.')
    if not re.fullmatch(r'cpu|cuda(?::[0-9]+)?', args.device):
        parser.error('--device must be cpu or cuda:<index>.')
    if args.device == 'cuda':
        args.device = 'cuda:0'
    if args.device == 'cpu' and (args.precision == 'fp16' or args.compile_training):
        parser.error('--precision fp16 and --compile require an explicit CUDA device.')
    if not 0 <= args.seed <= 2147483647:
        parser.error('--seed must be between 0 and 2147483647.')
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA was explicitly requested but is unavailable; no CPU fallback was used.')
        torch.cuda.set_device(torch.device(args.device))
    if args.compile_training:
        configure_compilation_environment(ROOT)
    lab = LocalTrainingLab(args.data_dir, args.artifact_dir, seed=args.seed, device=args.device,
                           training_precision=args.precision, compile_training=args.compile_training)
    if args.restore_session:
        lab.restore_live()
    lab.prepare_training_execution()
    health = {'app': APP_ID, 'ok': True, 'device': str(lab.device),
              'initialStep': lab.state()['step'], 'restoredFrom': 'live-session' if args.restore_session else 'random',
              'persistence': PERSISTENCE, 'execution': lab.metadata()['execution']}
    service = TrainingService(ROOT, factory=lambda **_: lab)
    server = create_server(port=args.port, training=service, health=health,
                           seminar_admin_port=args.seminar_admin_port)
    print(f'GPT Training Lab: http://127.0.0.1:{args.port}/ ({args.device}, {args.precision})', flush=True)
    try:
        server.serve_forever(poll_interval=.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
