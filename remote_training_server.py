"""CUDA-only GPT Training Lab API, bound to server loopback behind SSH.

The Windows gateway proxies this fixed API. It exposes no shell, arbitrary paths,
or CPU fallback. A new process restores the explicitly saved live session when
present; only an absent live session falls back to the pretraining checkpoint.
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import platform
import sys
import time

import torch

from training_engine import TrainingLab
from training_service import TrainingApiError, TrainingService

ROOT = Path(__file__).resolve().parent
APP_ID = 'gpt-gpu-training-home-v1'
MAX_BODY_BYTES = 65536
PERSISTENCE_DESCRIPTION = (
    'Save session explicitly preserves current language-model weights, optimizer, '
    'RNG, training history and the live transfer pair. Restart restores that saved '
    'session when present; otherwise it loads the original pretrained checkpoint. '
    'Unsaved updates remain in server memory. Recorded experiments stay separate.'
)


def restore_startup_session(lab, artifact_dir):
    """Restore only the fixed live session; a corrupt saved session fails startup."""
    if (Path(artifact_dir) / 'live' / 'session.pt').exists():
        lab.restore_live()
        source = 'live-session'
    else:
        lab.reset('pretrained')
        source = 'pretrained-checkpoint'
    # Warmup completes before main binds the service listener. Reset/restore may
    # already prepare the new model; this public operation is idempotent.
    lab.prepare_training_execution()
    return source


def configure_compilation_environment(root):
    """Private service cache and bounded compiler concurrency, current process only."""
    cache = Path(root) / '.runtime' / 'torchinductor-cache'
    if cache.resolve() != cache.absolute():
        raise RuntimeError('The service compilation cache must not be a symlink.')
    cache.mkdir(mode=0o700, parents=True, exist_ok=True)
    if hasattr(os, 'geteuid') and cache.stat().st_uid != os.geteuid():
        raise RuntimeError('The service compilation cache must belong to the service user.')
    cache.chmod(0o700)
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(cache)
    os.environ['TRITON_CACHE_DIR'] = str(cache / 'triton')
    os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
    os.environ['MAX_JOBS'] = '1'
    return cache


class GPUTrainingLab(TrainingLab):
    def metadata(self):
        result = super().metadata()
        result['hostname'] = platform.node()
        result['host'] = 'remote / ' + platform.node()
        result['gpuName'] = torch.cuda.get_device_name(self.device)
        result['description'] = result['description'].replace(
            'CPU training windows', 'GPU training windows')
        result['runtime'] = {
            'hostname': platform.node(), 'location': 'remote',
            'device': str(self.device), 'gpuName': torch.cuda.get_device_name(self.device),
            'torchVersion': torch.__version__, 'cudaVersion': torch.version.cuda,
            'trainingHost': 'Linux GPU host', 'weightsDevice': str(next(self.model.parameters()).device),
            'persistence': PERSISTENCE_DESCRIPTION,
        }
        return result


class GPUTrainingHandler(BaseHTTPRequestHandler):
    server_version = 'GPTTrainingHome/1'

    def _json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(body)

    def _boundary(self):
        port = self.server.server_address[1]
        if self.headers.get('Host') not in (f'127.0.0.1:{port}', f'localhost:{port}'):
            raise TrainingApiError(403, 'invalid_host', 'Only the loopback training endpoint is accepted.')
        # Browsers reach the Windows proxy, not this remote endpoint directly.
        if self.headers.get('Origin'):
            raise TrainingApiError(403, 'invalid_origin', 'Use the local training gateway.')
        if self.headers.get('Transfer-Encoding'):
            raise TrainingApiError(400, 'invalid_request', 'Chunked request bodies are unsupported.')

    def _stream_event(self, event):
        line = (json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
        if not self._stream_started:
            if event.get('type') != 'start':
                raise ValueError('A generation stream must begin with start.')
            self.send_response(200)
            self.send_header('Content-Type', 'application/x-ndjson; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Connection', 'close')
            self.close_connection = True
            self._stream_started = True
            self.end_headers()
        self.wfile.write(line)
        self.wfile.flush()
        if event.get('type') in ('done', 'error'):
            self._stream_finished = True

    def _request_error(self, status, code, message):
        payload = {'error': {'code': code, 'message': message}}
        try:
            if self._stream_started:
                if not self._stream_finished:
                    self._stream_event({'type': 'error', **payload})
            else:
                self._json(status, payload)
        except OSError:
            # The client has already gone away; never attempt another response.
            self.close_connection = True

    def _run(self, method):
        self._stream_started = self._stream_finished = False
        try:
            self._boundary()
            if method == 'GET' and self.path == '/api/health':
                return self._json(200, self.server.health)
            if not self.path.startswith('/api/training/'):
                raise TrainingApiError(404, 'not_found', 'Unknown training API route.')
            body = None
            if method == 'POST':
                if self.headers.get('Content-Type', '').split(';')[0].strip() != 'application/json':
                    raise TrainingApiError(415, 'invalid_content_type', 'Send application/json.')
                try:
                    length = int(self.headers.get('Content-Length', '0'))
                except ValueError as exc:
                    raise TrainingApiError(400, 'invalid_input', 'Invalid Content-Length.') from exc
                if not 0 < length <= MAX_BODY_BYTES:
                    raise TrainingApiError(413, 'body_too_large', 'JSON body must contain 1–65536 bytes.')
                try:
                    body = json.loads(self.rfile.read(length).decode('utf-8'))
                except (ValueError, UnicodeError) as exc:
                    raise TrainingApiError(400, 'invalid_json', 'Malformed JSON body.') from exc
            if method == 'POST' and self.path.split('?', 1)[0] == '/api/training/generate-stream':
                self.server.training.dispatch(method, self.path, body, on_event=self._stream_event)
                if not self._stream_finished:
                    raise TrainingApiError(500, 'incomplete_stream', 'Generation ended without a completion event.')
                return
            result = self.server.training.dispatch(method, self.path, body)
            self._json(200, result)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError):
            self.close_connection = True
        except TrainingApiError as exc:
            self._request_error(exc.status, exc.code, exc.message)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            self._request_error(503, 'gpu_memory_unavailable',
                'GPU memory is temporarily unavailable. Training stopped; no CPU fallback was used. Inspect current state before continuing.')
        except Exception as exc:
            print(f'Training request failed: {type(exc).__name__}', file=sys.stderr, flush=True)
            self._request_error(500, 'training_failed',
                                'The GPU training request failed. Check the owned training service log.')

    def do_GET(self):
        self._run('GET')

    def do_POST(self):
        self._run('POST')

    def do_OPTIONS(self):
        self._json(405, {'error': {'code': 'method_not_allowed', 'message': 'Use the local proxy.'}})


class GPUTrainingServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=18768)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; this service never silently falls back to CPU.')
    torch.cuda.set_device(0)
    # Limit allocator-owned storage while sharing the GPU with other
    # service. CUDA context and driver/library storage are outside this cap.
    torch.cuda.set_per_process_memory_fraction(0.08, 0)
    free, total = torch.cuda.mem_get_info(0)
    if free < 800 * 1024 * 1024:
        raise RuntimeError('Less than 800 MiB of GPU memory is free. Existing services were left unchanged.')
    started = time.time()
    configure_compilation_environment(ROOT)
    lab = GPUTrainingLab(ROOT / 'data', ROOT / '.training', seed=42, device='cuda:0',
                         compile_training=True, training_precision='fp16')
    restored_from = restore_startup_session(lab, ROOT / '.training')
    if not all(p.device.type == 'cuda' for p in lab.model.parameters()):
        raise RuntimeError('The model was not placed entirely on CUDA.')
    server = GPUTrainingServer(('127.0.0.1', args.port), GPUTrainingHandler)
    server.training = TrainingService(ROOT, factory=lambda **_: lab)
    server.health = {'app': APP_ID, 'ok': True, 'hostname': platform.node(),
                     'device': str(lab.device), 'gpuName': torch.cuda.get_device_name(0),
                     'torchVersion': torch.__version__, 'cudaVersion': torch.version.cuda,
                     'pid': os.getpid(), 'initialStep': lab.state()['step'],
                     'restoredFrom': restored_from, 'persistence': PERSISTENCE_DESCRIPTION,
                     'execution': lab.metadata()['execution'],
                     'weightsDevice': str(next(lab.model.parameters()).device),
                     'startedAtUnix': started, 'allocatorFractionLimit': 0.08}
    (ROOT / '.runtime').mkdir(exist_ok=True)
    (ROOT / '.runtime' / 'gpu-server.pid').write_text(str(os.getpid()), encoding='ascii')
    print(json.dumps(server.health), flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
