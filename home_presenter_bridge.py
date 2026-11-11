"""Loopback presenter bridge; reuses the live trainer and never imports its model."""
from copy import deepcopy
from datetime import datetime, timezone
import signal
import threading
import time

from runtime_service import GpuUpstream, RuntimeFailure, RuntimeHandler, RuntimeServer, REMOTE_APP
from telemetry_probe import empty_source, sample_linux

APP_ID = 'gpt-home-presenter-bridge-v1'
PORT = 18769


class ExistingTrainer:
    def ensure_running(self):
        pass  # The existing trainer owns its own lifecycle; never start/restart it.


class HomeTelemetry:
    def __init__(self, sampler=sample_linux, interval=1.0):
        self.sampler, self.interval = sampler, interval
        self.lock, self.stopping = threading.Lock(), threading.Event()
        self.sample = empty_source('home', 'Linux · GPU training', 'Linux', True)
        self.sample_time, self.failed, self.thread = None, False, None

    def sample_once(self):
        try:
            value = self.sampler()
            if not isinstance(value, dict) or value.get('id') != 'home':
                raise ValueError('Invalid Linux telemetry sample')
            value = deepcopy(value)
            value['isTrainingHost'] = True
            value['label'] = 'Linux · GPU training'
            with self.lock:
                self.sample, self.sample_time, self.failed = value, time.monotonic(), False
        except Exception:
            with self.lock:
                self.failed = True

    def _run(self):
        while not self.stopping.is_set():
            started = time.monotonic()
            self.sample_once()
            self.stopping.wait(max(.1, self.interval-(time.monotonic()-started)))

    def start(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self._run, daemon=True, name='gpu-telemetry')
            self.thread.start()

    def stop(self):
        self.stopping.set()

    def snapshot(self):
        with self.lock:
            value, measured, failed = deepcopy(self.sample), self.sample_time, self.failed
        if measured is None:
            value['status'] = 'unavailable' if failed else 'loading'
        else:
            value['ageSeconds'] = max(0., time.monotonic()-measured)
            value['status'] = 'stale' if failed or value['ageSeconds'] > 15 else 'ok'
        if failed:
            value['notes'].append('Linux sensors could not provide a fresh sample.')
        return {'schemaVersion':1, 'pollIntervalMs':int(self.interval*1000), 'sources':[value]}


class HomePresenterHandler(RuntimeHandler):
    server_version = 'HomePresenterBridge/1'

    def _boundary(self):
        return (self.headers.get_all('Host', []) == [f'127.0.0.1:{self.server.server_port}'] and
                not self.headers.get_all('Origin', []) and
                self.headers.get('Sec-Fetch-Site') != 'cross-site')

    def do_GET(self):
        if self._reject_boundary():
            return
        if self.path == '/api/health':
            return self._json(200, {'app':APP_ID,'ok':True,'trainingTarget':'remote / CUDA'})
        super().do_GET()

    def do_OPTIONS(self):
        self._json(403, {'error':{'code':'forbidden','message':'Use the authenticated presenter gateway.'}})


def main():
    server = RuntimeServer(('127.0.0.1',PORT), HomePresenterHandler)
    server.gpu, server.telemetry = GpuUpstream(ExistingTrainer(), port=18768), HomeTelemetry()
    try:
        code, health = server.gpu.request('GET','/api/health',timeout=8)
        if code != 200 or health.get('app') != REMOTE_APP:
            raise RuntimeFailure('The configured GPU trainer is unavailable.')
        server.telemetry.start()
        def stop(*_):
            threading.Thread(target=server.shutdown,daemon=True).start()
        signal.signal(signal.SIGTERM,stop)
        signal.signal(signal.SIGINT,stop)
        print(f'{APP_ID} ready on loopback:{PORT}',flush=True)
        server.serve_forever(poll_interval=.25)
    finally:
        server.telemetry.stop()
        server.server_close()


if __name__ == '__main__':
    main()
