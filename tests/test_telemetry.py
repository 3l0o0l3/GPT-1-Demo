import http.client
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from telemetry_probe import cpu_times, cpu_percent, memory_info, gpu_rows, package_energy, package_watts, empty_source, utc_now
from telemetry_service import TelemetryCache, TelemetryHandler, ThreadingHTTPServer, APP_ID


def measured(source_id='home'):
    value = empty_source(source_id, 'Fixture', 'Linux', False)
    value.update(status='ok', sampledAt=utc_now(), ageSeconds=0)
    value['cpu']['utilizationPercent'] = 25.0
    return value


class ProbeTests(unittest.TestCase):
    def test_cpu_delta_excludes_duplicate_guest_time_and_treats_iowait_as_idle(self):
        first = cpu_times('cpu 100 0 50 850 10 0 0 0 70 0\ncpu0 0 0 0 0')
        second = cpu_times('cpu 180 0 50 870 10 0 0 0 120 0')
        self.assertEqual(first, (1010, 860))
        self.assertEqual(cpu_percent(first, second), 80)
        self.assertIsNone(cpu_percent(first, first))

    def test_linux_memory_uses_available_and_binary_units(self):
        result = memory_info('MemTotal: 1000 kB\nMemAvailable: 600 kB\nMemFree: 100 kB')
        self.assertEqual(result, {'usedBytes': 409600, 'totalBytes': 1024000, 'utilizationPercent': 40})
        self.assertIsNone(memory_info('MemTotal: 1000 kB')['usedBytes'])

    def test_gpu_missing_sensors_never_become_zero(self):
        rows = gpu_rows('0, NVIDIA RTX 4090, 0, 1024, 24576, 12.5, 450, 42\n1, Other GPU, [N/A], N/A, 1024, N/A, N/A, N/A')
        self.assertEqual(rows[0]['memoryUsedBytes'], 1024 ** 3)
        self.assertEqual(rows[0]['utilizationPercent'], 0)
        self.assertEqual(rows[0]['powerWatts'], 12.5)
        self.assertIsNone(rows[1]['utilizationPercent'])
        self.assertIsNone(rows[1]['powerWatts'])
        self.assertIsNone(gpu_rows('0, GPU, nan, 1, 2, inf, 400, -999')[0]['powerWatts'])

    def test_package_power_handles_wrap_and_ignores_overlapping_subdomains(self):
        self.assertAlmostEqual(package_watts({'p': (900000, 1000000)}, {'p': (100000, 1000000)}, .5), .4)
        self.assertIsNone(package_watts({}, {}, .5))
        self.assertIsNone(package_watts({'p': (90, 100)}, {'p': (1, 200)}, .5))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, label in (('zone0', 'package-0'), ('zone1', 'core'), ('zone2', 'psys')):
                zone = root / name
                zone.mkdir()
                (zone / 'name').write_text(label)
                (zone / 'energy_uj').write_text('1000')
                (zone / 'max_energy_range_uj').write_text('10000')
            self.assertEqual(len(package_energy(root)), 1)
            with patch.object(Path, 'read_text', side_effect=PermissionError()):
                self.assertEqual(package_energy(root), {})


class CacheTests(unittest.TestCase):
    def test_initial_failure_and_last_good_sample_are_distinct(self):
        def fail():
            raise RuntimeError('PRIVATE SSH DETAIL MUST NOT LEAK')
        cache = TelemetryCache(remote_sampler=fail)
        self.assertEqual(cache.snapshot()['sources'][1]['status'], 'warming')
        cache.sample_once('home')
        failed = cache.snapshot()['sources'][1]
        self.assertEqual(failed['status'], 'unavailable')
        self.assertIsNone(failed['cpu']['utilizationPercent'])
        self.assertNotIn('PRIVATE', json.dumps(failed))
        sample = measured()
        cache.samplers['home'] = lambda: sample
        cache.sample_once('home')
        cache.samplers['home'] = fail
        cache.sample_once('home')
        stale = cache.snapshot()['sources'][1]
        self.assertEqual(stale['status'], 'stale')
        self.assertEqual(stale['sampledAt'], sample['sampledAt'])
        self.assertEqual(stale['cpu']['utilizationPercent'], 25)
        self.assertGreaterEqual(stale['ageSeconds'], 0)
        self.assertEqual(len(sample['notes']), 0)  # Snapshot does not mutate stored source.

    def test_cached_reads_do_not_wait_for_ssh_and_start_is_idempotent(self):
        entered, release = threading.Event(), threading.Event()
        def slow():
            entered.set()
            release.wait(1)
            return measured()
        cache = TelemetryCache(local_sampler=lambda: measured('local'), remote_sampler=slow)
        cache.start()
        cache.start()
        self.assertTrue(entered.wait(1))
        try:
            start = time.monotonic()
            data = cache.snapshot()
            self.assertLess(time.monotonic() - start, .1)
            self.assertEqual(len(cache.threads), 2)
            self.assertEqual(data['sources'][0]['status'], 'ok')
            self.assertEqual(data['sources'][1]['status'], 'warming')
        finally:
            cache.stop()
            release.set()
            for worker in cache.threads:
                worker.join(1)

    def test_selected_training_host_is_explicit_not_inferred_from_load(self):
        cache = TelemetryCache(training_host='home')
        local, remote = cache.snapshot()['sources']
        self.assertFalse(local['isTrainingHost'])
        self.assertTrue(remote['isTrainingHost'])
        self.assertIn('GPU training', remote['label'])
        with self.assertRaises(ValueError):
            TelemetryCache(training_host='arbitrary-host')


class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), TelemetryHandler)
        cls.server.telemetry = TelemetryCache()
        cls.worker = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.worker.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.worker.join(1)

    def request(self, path, headers=None, method='GET'):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=2)
        connection.request(method, path, headers=headers or {})
        response = connection.getresponse()
        body = json.loads(response.read())
        result = response.status, dict(response.getheaders()), body
        connection.close()
        return result

    def test_loopback_origin_receives_cached_snapshot(self):
        status, headers, result = self.request('/api/telemetry', {'Origin': 'http://127.0.0.1:8768'})
        self.assertEqual(status, 200)
        self.assertEqual(headers['Access-Control-Allow-Origin'], 'http://127.0.0.1:8768')
        self.assertEqual(headers['Cache-Control'], 'no-store')
        self.assertEqual(result['schemaVersion'], 1)
        self.assertEqual(len(result['sources']), 2)
        self.assertEqual(self.request('/api/health')[2]['app'], APP_ID)

    def test_host_origin_and_input_boundaries(self):
        self.assertEqual(self.request('/api/telemetry', {'Origin': 'https://unrelated.example'})[0], 403)
        self.assertEqual(self.request('/api/telemetry', {'Host': 'unrelated.example'})[0], 403)
        self.assertEqual(self.request('/api/telemetry?host=other')[0], 400)
        self.assertEqual(self.request('/arbitrary')[0], 404)
        self.assertEqual(self.request('/api/telemetry', method='POST')[0], 405)


if __name__ == '__main__':
    unittest.main()


class ConfiguredRemoteTelemetryTests(unittest.TestCase):
    def test_missing_or_invalid_destination_never_spawns_a_sampler(self):
        from telemetry_service import sample_home
        for value in ('','-oProxyCommand=bad','host name'):
            with self.subTest(value=value), patch.dict(os.environ,{'GPT_TRAINING_SSH_HOST':value}), patch('telemetry_service.subprocess.run') as run:
                with self.assertRaisesRegex(RuntimeError,'GPT_TRAINING_SSH_HOST'):
                    sample_home()
                run.assert_not_called()

    def test_configured_remote_sampler_preserves_the_legacy_source_id(self):
        from telemetry_service import sample_home
        with patch.dict(os.environ,{'GPT_TRAINING_SSH_HOST':'test-user@gpu.example.com'}), patch('telemetry_service.subprocess.run',return_value=Mock(returncode=0,stdout=json.dumps(measured()))) as run:
            sample=sample_home()
            self.assertEqual(sample['id'],'home')
            self.assertEqual(run.call_args.args[0][-3: ],['test-user@gpu.example.com','python3','-'])
        for source in TelemetryCache(training_host='home').snapshot()['sources']:
            self.assertNotIn('home',source['label']);self.assertNotIn('Qwen',source['label'])
