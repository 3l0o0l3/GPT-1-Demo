"""Independent cached hardware monitor; never imports or locks the training engine."""
from copy import deepcopy
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import argparse
import ctypes
import json
import os
import re
import platform
import subprocess
import threading
import time
from urllib.parse import urlsplit

from telemetry_probe import empty_source, utc_now, cpu_percent

ROOT = Path(__file__).resolve().parent
APP_ID = 'gpt-training-telemetry-v1'
PORT = 8769
INTERVAL = 1.0
ORIGINS = {'http://127.0.0.1:8768', 'http://localhost:8768'}


class FileTime(ctypes.Structure):
    _fields_ = [('low', ctypes.c_uint32), ('high', ctypes.c_uint32)]

    def ticks(self):
        return self.low + (self.high << 32)


class MemoryStatus(ctypes.Structure):
    _fields_ = [('length', ctypes.c_uint32), ('load', ctypes.c_uint32),
                ('totalPhys', ctypes.c_uint64), ('availPhys', ctypes.c_uint64),
                ('totalPageFile', ctypes.c_uint64), ('availPageFile', ctypes.c_uint64),
                ('totalVirtual', ctypes.c_uint64), ('availVirtual', ctypes.c_uint64),
                ('availExtendedVirtual', ctypes.c_uint64)]


def windows_times():
    idle, kernel, user = FileTime(), FileTime(), FileTime()
    if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
        raise OSError('Windows CPU counters unavailable.')
    # Windows kernel time includes idle time.
    return kernel.ticks() + user.ticks(), idle.ticks()


def sample_local(interval=0.3):
    sample = empty_source('local', 'Windows · actual CPU training host', platform.system(), True)
    sample['hostname'] = platform.node()
    sample['cpu'].update(name=platform.processor() or None, logicalCores=os.cpu_count(),
                         powerReason='CPU package power is not exposed by the standard Windows counters used here.')
    if os.name != 'nt':
        raise RuntimeError('The local sampler is for this Windows training host.')
    first = windows_times()
    time.sleep(interval)
    sample['cpu']['utilizationPercent'] = cpu_percent(first, windows_times())
    memory = MemoryStatus()
    memory.length = ctypes.sizeof(memory)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
        raise OSError('Windows memory counters unavailable.')
    used = memory.totalPhys - memory.availPhys
    sample['memory'] = {'usedBytes': used, 'totalBytes': memory.totalPhys,
                        'utilizationPercent': 100.0 * used / memory.totalPhys}
    sample['notes'] = ['Whole-machine CPU and RAM, including other applications; this panel does not attribute all host load to the demo.']
    sample['status'], sample['sampledAt'], sample['ageSeconds'] = 'ok', utc_now(), 0.0
    return sample


def sample_home():
    # Keep the legacy source ID; the SSH destination is explicitly configured.
    host = os.environ.get('GPT_TRAINING_SSH_HOST', '').strip()
    if not host:
        raise RuntimeError('Set GPT_TRAINING_SSH_HOST before sampling remote telemetry.')
    if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@:\[\]-]{0,254}', host):
        raise RuntimeError('GPT_TRAINING_SSH_HOST must be a single SSH destination, without options or whitespace.')
    script = (ROOT / 'telemetry_probe.py').read_text(encoding='utf-8')
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    result = subprocess.run(
        ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=4',
         '-o', 'ServerAliveInterval=3', '-o', 'ServerAliveCountMax=1',
         host, 'python3', '-'],
        input=script, capture_output=True, encoding='utf-8', errors='replace',
        timeout=8, creationflags=flags)
    if result.returncode != 0:
        raise RuntimeError('Configured remote telemetry is unavailable.')
    if len(result.stdout) > 65536:
        raise ValueError('Unexpected telemetry response size.')
    sample = json.loads(result.stdout, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Non-finite value.')))
    if not isinstance(sample, dict) or sample.get('id') != 'home' or sample.get('isTrainingHost') is not False or sample.get('status') != 'ok':
        raise ValueError('Unexpected telemetry response.')
    # Ensures malformed records fail here rather than breaking cache-only reads.
    datetime.fromisoformat(sample['sampledAt'])
    for field in ('cpu', 'memory', 'gpus', 'notes'):
        if field not in sample:
            raise ValueError('Incomplete telemetry response.')
    return sample


class TelemetryCache:
    def __init__(self, local_sampler=sample_local, remote_sampler=sample_home, interval=INTERVAL, training_host='local'):
        if training_host not in ('local', 'home'):
            raise ValueError('Training host must be local or home.')
        self.training_host = training_host
        self.samplers = {'local': local_sampler, 'home': remote_sampler}
        self.interval = interval
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.samples = {
            'local': empty_source('local', 'Windows · actual CPU training host', 'Windows', True),
            'home': empty_source('home', 'Linux · remote host', 'Linux', False),
        }
        self.failures = {}
        self.monotonic_samples = {}
        self.threads = []

    def sample_once(self, source_id):
        try:
            sample = self.samplers[source_id]()
            json.dumps(sample, allow_nan=False)
            with self.lock:
                self.samples[source_id] = sample
                self.monotonic_samples[source_id] = time.monotonic()
                self.failures.pop(source_id, None)
        except Exception:
            # Never send command lines, stderr or private SSH configuration to UI.
            reason = ('The configured remote host could not provide a fresh sample (connection, timeout or sensor error).'
                      if source_id == 'home' else 'Local Windows counters could not provide a fresh sample.')
            with self.lock:
                self.failures[source_id] = reason

    def _worker(self, source_id):
        while not self.stop_event.is_set():
            start = time.monotonic()
            self.sample_once(source_id)
            self.stop_event.wait(max(0.2, self.interval - (time.monotonic() - start)))

    def start(self):
        if self.threads:
            return
        # Separate sources keep Windows metrics fresh if remote SSH is slow.
        for source_id in self.samplers:
            worker = threading.Thread(target=self._worker, args=(source_id,), daemon=True,
                                      name=f'telemetry-{source_id}')
            self.threads.append(worker)
            worker.start()

    def stop(self):
        self.stop_event.set()

    def snapshot(self):
        with self.lock:
            sources = deepcopy(list(self.samples.values()))
            failures = dict(self.failures)
            measured = dict(self.monotonic_samples)
        now = datetime.now(timezone.utc)
        for source in sources:
            source_id = source['id']
            source['isTrainingHost'] = source_id == self.training_host
            source['label'] = (
                ('Windows · actual CPU training host' if source_id == 'local' else 'Linux · GPU training')
                if source['isTrainingHost'] else
                ('Windows · browser / gateway' if source_id == 'local' else 'Linux · remote host'))
            if source.get('sampledAt'):
                stamp = datetime.fromisoformat(source['sampledAt'])
                source['ageSeconds'] = max(0.0, (now - stamp).total_seconds())
                timed_out = time.monotonic() - measured.get(source_id, 0) > max(15, self.interval * 3)
                source['status'] = 'stale' if source_id in failures or timed_out else 'ok'
                if source_id in failures:
                    source['notes'].append(failures[source_id] + ' Showing the last successful measurement.')
            elif source_id in failures:
                source['status'] = 'unavailable'
                source['notes'].append(failures[source_id])
        return {'schemaVersion': 1, 'pollIntervalMs': int(self.interval * 1000), 'sources': sources}


class TelemetryHandler(BaseHTTPRequestHandler):
    server_version = 'TrainingTelemetry/1'

    def log_message(self, fmt, *args):
        pass  # Periodic samples should not create unbounded logs.

    def _boundary(self):
        allowed_hosts = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
        hosts = self.headers.get_all('Host', [])
        if len(hosts) != 1 or hosts[0] not in allowed_hosts:
            return False
        origins = self.headers.get_all('Origin', [])
        return not origins or (len(origins) == 1 and origins[0] in ORIGINS)

    def _json(self, status, payload):
        data = json.dumps(payload, allow_nan=False).encode('utf-8')
        self.send_response(status)
        origin = self.headers.get('Origin')
        if origin in ORIGINS:
            self.send_header('Access-Control-Allow-Origin', origin)
        self.send_header('Vary', 'Origin')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if not self._boundary():
            return self._json(403, {'error': {'code': 'forbidden', 'message': 'Loopback requests from the local demo only.'}})
        parts = urlsplit(self.path)
        if parts.query or parts.fragment:
            return self._json(400, {'error': {'code': 'invalid_input', 'message': 'No query parameters are accepted.'}})
        if parts.path == '/api/health':
            return self._json(200, {'app': APP_ID, 'ok': True})
        if parts.path == '/api/telemetry':
            return self._json(200, self.server.telemetry.snapshot())
        self._json(404, {'error': {'code': 'not_found', 'message': 'Unknown telemetry route.'}})

    def do_POST(self):
        self._json(405, {'error': {'code': 'method_not_allowed', 'message': 'Use GET.'}})

    do_PUT = do_POST
    do_DELETE = do_POST
    do_PATCH = do_POST
    do_OPTIONS = do_POST
    do_HEAD = do_POST


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-host', choices=('local', 'home'), default='local')
    args = parser.parse_args()
    cache = TelemetryCache(training_host=args.training_host)
    server = ThreadingHTTPServer(('127.0.0.1', PORT), TelemetryHandler)
    server.daemon_threads = True
    server.telemetry = cache
    cache.start()
    try:
        print(f'{APP_ID} listening on http://127.0.0.1:{PORT}', flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        cache.stop()
        server.server_close()


if __name__ == '__main__':
    main()
