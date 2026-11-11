"""Independent diagnostic contracts, without torch or a CUDA device."""
import json
import copy
import threading
import unittest

from training_timing import NULL_SAMPLE, TrainingTiming


class Clock:
    def __init__(self):
        self.value = 0.
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.value

    def advance(self, milliseconds):
        self.value += milliseconds / 1000.


class Events:
    def __init__(self, clock, failure=None):
        self.clock, self.failure = clock, failure
        self.ready = False
        self.created = self.queries = self.reads = 0
        self.query_entered = self.query_release = None

    def __call__(self):
        if self.failure == 'create':
            raise RuntimeError('event creation failed')
        self.created += 1
        owner = self

        class Event:
            def record(self):
                if owner.failure == 'record':
                    raise RuntimeError('event recording failed')
                self.timestamp = owner.clock.value

            def query(self):
                owner.queries += 1
                if owner.query_entered is not None:
                    owner.query_entered.set()
                    if not owner.query_release.wait(2):
                        raise RuntimeError('test query gate timed out')
                if owner.failure == 'query':
                    raise RuntimeError('event query failed')
                return owner.ready

            def elapsed_time(self, other):
                owner.reads += 1
                if owner.failure == 'elapsed':
                    raise RuntimeError('event read failed')
                if owner.failure == 'nonfinite':
                    return float('nan')
                return (other.timestamp - self.timestamp) * 500.

        return Event()


class TrainingTimingTests(unittest.TestCase):
    def make(self, **kwargs):
        clock = Clock()
        events = Events(clock)
        timing = TrainingTiming('cuda', clock=clock, event_factory=events, **kwargs)
        return timing, clock, events

    def test_first_observed_then_absolute_cadence_and_independent_evaluation(self):
        timing, clock, _ = self.make()
        for step in (97, 98, 99, 100, 101):
            sample = timing.begin_step('lm', step)
            self.assertEqual(sample.sampled, step in (97, 100))
            clock.advance(1)
            sample.end()
        self.assertTrue(timing.begin_step('transfer', 731).sampled)
        evaluation = timing.begin_evaluation('lm', 101)
        clock.advance(3)
        evaluation.end()
        view = timing.snapshot('lm')
        self.assertEqual([97, 100], [row['step'] for row in view['steps']['trend']])
        self.assertEqual(5, view['counters']['observedSteps'])
        self.assertEqual(2, view['counters']['sampledSteps'])
        self.assertEqual(1, view['counters']['evaluations'])
        self.assertEqual(3, view['counters']['successfulSamples'])
        self.assertEqual(101, view['evaluations']['trend'][0]['step'])

    def test_cpu_repeated_and_nested_spans_are_not_summed_into_total(self):
        clock = Clock()
        timing = TrainingTiming('cpu', clock=clock)
        sample = timing.begin_step('lm', 1)
        clock.advance(1)
        with sample.phase('outer'):
            clock.advance(2)
            with sample.phase('inner'):
                clock.advance(3)
            clock.advance(2)
        with sample.phase('inner'):
            clock.advance(1)
        sample.end()
        group = timing.snapshot('lm')['steps']
        row = group['trend'][0]
        self.assertAlmostEqual(9., row['cpuMs'])
        self.assertAlmostEqual(7., group['phases']['outer']['cpuMs']['mean'])
        self.assertAlmostEqual(4., group['phases']['inner']['cpuMs']['mean'])
        self.assertIsNone(row['gpuMs'])
        self.assertGreater(sum(p['cpuMs']['mean'] for p in group['phases'].values()), row['cpuMs'])

    def test_window_statistics_exclude_failures_but_keep_failure_trend(self):
        clock = Clock()
        timing = TrainingTiming('cpu', sample_every=1, capacity=3, clock=clock)
        for step, elapsed, success in ((1, 100., True), (2, 2., True),
                                       (3, 99., False), (4, 4., True)):
            sample = timing.begin_step('lm', step)
            with sample.phase('work'):
                clock.advance(elapsed)
            sample.end(success=success)
            sample.end(success=success)  # idempotent
        view = timing.snapshot('lm')
        group = view['steps']
        self.assertEqual((3, 2, 1), (group['count'], group['successfulCount'], group['failedCount']))
        self.assertEqual([2, 3, 4], [row['step'] for row in group['trend']])
        self.assertFalse(group['trend'][1]['success'])
        for stats in (group['total']['cpuMs'], group['phases']['work']['cpuMs']):
            self.assertEqual(2, stats['count'])
            self.assertAlmostEqual(3., stats['mean'])
            self.assertAlmostEqual(4., stats['p95'])
            self.assertAlmostEqual(2., stats['min'])
            self.assertAlmostEqual(4., stats['max'])
        self.assertEqual(3, view['counters']['successfulSamples'])
        self.assertEqual(1, view['counters']['failedSamples'])

    def test_gpu_resolution_is_delayed_and_snapshot_never_calls_events(self):
        timing, clock, events = self.make()
        sample = timing.begin_step('lm', 1)
        with sample.phase('forward'):
            clock.advance(4)
        with sample.phase('forward'):
            clock.advance(2)
        clock.advance(4)
        sample.end()
        view = timing.snapshot('lm')
        self.assertEqual(0, events.queries)
        self.assertEqual(1, view['pendingGpuSamples'])
        self.assertEqual(0, view['steps']['total']['gpuMs']['count'])
        timing.drain()
        self.assertEqual(1, events.queries)
        self.assertEqual(0, events.reads)
        events.ready = True
        timing.drain()
        view = timing.snapshot('lm')
        self.assertEqual(0, view['pendingGpuSamples'])
        self.assertAlmostEqual(10., view['steps']['total']['cpuMs']['mean'])
        self.assertAlmostEqual(5., view['steps']['total']['gpuMs']['mean'])
        self.assertAlmostEqual(3., view['steps']['phases']['forward']['gpuMs']['mean'])
        self.assertEqual(1, view['steps']['total']['gpuMs']['count'])

    def test_pending_capacity_is_global_and_cpu_samples_survive_overflow(self):
        timing, clock, events = self.make(max_pending=1, sample_every=1)
        first = timing.begin_step('lm', 1)
        second = timing.begin_step('transfer', 1)
        clock.advance(2)
        first.end()
        second.end()
        self.assertEqual(2, events.created)  # one total start/end pair only
        view = timing.snapshot('transfer')
        self.assertEqual(1, view['steps']['count'])
        self.assertEqual(1, view['counters']['gpuDroppedSamples'])
        self.assertEqual(0, view['pendingGpuSamples'])
        self.assertIsNone(view['steps']['trend'][0]['gpuMs'])
        events.ready = True
        timing.drain()
        third = timing.begin_step('transfer', 2)
        third.end()
        timing.drain()
        self.assertEqual(1, timing.snapshot('transfer')['steps']['total']['gpuMs']['count'])

    def test_event_failures_never_fail_training_or_fabricate_gpu_numbers(self):
        for failure in ('create', 'record', 'query', 'elapsed', 'nonfinite'):
            with self.subTest(failure=failure):
                clock = Clock()
                events = Events(clock, failure=failure)
                timing = TrainingTiming('cuda', clock=clock, event_factory=events)
                sample = timing.begin_step('lm', 1)
                with sample.phase('work'):
                    clock.advance(2)
                sample.end()
                events.ready = True
                timing.drain()
                timing.drain()
                view = timing.snapshot('lm')
                self.assertEqual(1, view['counters']['eventErrors'])
                self.assertEqual(0, view['pendingGpuSamples'])
                self.assertEqual(1, view['steps']['total']['cpuMs']['count'])
                self.assertEqual(0, view['steps']['total']['gpuMs']['count'])

    def test_original_phase_error_is_preserved_and_failed_sample_is_visible(self):
        timing, clock, events = self.make()
        events.failure = 'record'
        sample = timing.begin_step('lm', 1)
        original = RuntimeError('model failed')
        try:
            with sample.phase('forward'):
                clock.advance(1)
                raise original
        except RuntimeError as actual:
            self.assertIs(original, actual)
        finally:
            sample.end(success=False)
        view = timing.snapshot('lm')
        self.assertEqual(1, view['steps']['failedCount'])
        self.assertEqual(0, view['steps']['total']['cpuMs']['count'])
        self.assertFalse(view['steps']['trend'][0]['success'])

    def test_snapshot_is_detached_and_strict_json_serializable(self):
        timing, _, _ = self.make()
        sample = timing.begin_step('lm', 1)
        with sample.phase('work'):
            pass
        sample.end()
        view = timing.snapshot('lm')
        json.dumps(view, allow_nan=False)
        view['counters']['sampledSteps'] = 999
        view['steps']['phases']['work']['cpuMs']['mean'] = 999
        view['steps']['total']['cpuMs']['mean'] = 999
        untouched = timing.snapshot('lm')
        self.assertEqual(1, untouched['counters']['sampledSteps'])
        self.assertEqual(0., untouched['steps']['phases']['work']['cpuMs']['mean'])
        self.assertEqual({'step','success','cpuMs','gpuMs'}, set(untouched['steps']['trend'][0]))
        self.assertEqual(0., untouched['steps']['total']['cpuMs']['mean'])

    def test_reset_discards_old_pending_and_late_handles_only_for_selected_kind(self):
        timing, _, events = self.make()
        ended = timing.begin_step('lm', 1)
        ended.end()
        late = timing.begin_evaluation('lm', 1)
        transfer = timing.begin_step('transfer', 5)
        transfer.end()
        timing.reset('lm')
        late.end()
        events.ready = True
        timing.drain()
        self.assertEqual(1, events.queries)  # only the unaffected Transfer record
        self.assertEqual(0, timing.snapshot('lm')['steps']['count'])
        self.assertEqual(0, timing.snapshot('lm')['evaluations']['count'])
        self.assertEqual(0, timing.snapshot('lm')['pendingGpuSamples'])
        self.assertEqual(1, timing.snapshot('transfer')['steps']['count'])
        self.assertTrue(timing.begin_step('lm', 177).sampled)
        timing.reset()
        self.assertEqual(0, timing.snapshot('transfer')['steps']['count'])

    def test_disabled_and_null_samples_do_not_touch_clocks_or_events(self):
        def forbidden():
            raise AssertionError('disabled instrumentation touched dependency')
        timing = TrainingTiming('cuda', enabled=False, clock=forbidden, event_factory=forbidden)
        for sample in (timing.begin_step('lm', 1), timing.begin_evaluation('transfer', 0), NULL_SAMPLE):
            self.assertIs(sample, NULL_SAMPLE)
            with sample.phase('ignored'):
                pass
            sample.end()
        timing.drain()
        view = timing.snapshot('lm')
        self.assertFalse(view['enabled'])
        self.assertFalse(view['gpuSupported'])
        self.assertEqual(0, view['counters']['observedSteps'])

    def test_event_query_never_holds_the_snapshot_publication_lock(self):
        timing, _, events = self.make()
        timing.begin_step('lm', 1).end()
        events.query_entered, events.query_release = threading.Event(), threading.Event()
        worker = threading.Thread(target=timing.drain)
        worker.start()
        try:
            self.assertTrue(events.query_entered.wait(1))
            finished = threading.Event()
            reader = threading.Thread(target=lambda: (timing.snapshot('lm'), finished.set()))
            reader.start()
            self.assertTrue(finished.wait(1), 'snapshot blocked on CUDA query')
            reader.join(1)
        finally:
            events.query_release.set()
            worker.join(2)
        self.assertFalse(worker.is_alive())

    def test_invalid_api_values_are_explicit(self):
        for kwargs in ({'sample_every': 0}, {'capacity': True}, {'max_pending': -1}, {'enabled': 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TrainingTiming('cpu', **kwargs)
        timing = TrainingTiming('cpu')
        for kind, step in (('bad', 1), ('lm', -1), ('transfer', True)):
            with self.assertRaises(ValueError):
                timing.begin_step(kind, step)
        with self.assertRaises(ValueError):
            timing.snapshot('bad')
        sample = timing.begin_step('lm', 1)
        with self.assertRaises(ValueError):
            sample.phase('')
        with self.assertRaises(ValueError):
            sample.end(success=1)

    def test_failed_clock_is_missing_data_not_a_zero_duration(self):
        def broken():
            raise RuntimeError('clock unavailable')
        timing = TrainingTiming('cpu', clock=broken)
        sample = timing.begin_step('lm', 1)
        with sample.phase('work'):
            pass
        sample.end()
        view = timing.snapshot('lm')
        self.assertGreater(view['counters']['clockErrors'], 0)
        self.assertIsNone(view['steps']['trend'][0]['cpuMs'])
        self.assertIsNone(view['steps']['phases']['work']['cpuMs']['mean'])
        self.assertEqual(0, view['steps']['total']['cpuMs']['count'])

    def test_published_values_are_deeply_immutable_json_and_copy_safe(self):
        timing, clock, events = self.make(sample_every=1)
        sample=timing.begin_step('lm',1)
        with sample.phase('work'):clock.advance(2)
        sample.end()
        published=timing.published_snapshot('lm')
        frozen_json=json.dumps(published,sort_keys=True)
        self.assertEqual(json.dumps(timing.snapshot('lm'),sort_keys=True),frozen_json)
        self.assertIs(copy.deepcopy(published),published)
        self.assertIs(copy.deepcopy({'nested':published})['nested'],published)
        self.assertIs(copy.copy(published),published)
        self.assertIsInstance(published['steps']['trend'],tuple)
        for node in (published,published['counters'],published['steps'],
                     published['steps']['trend'][0],published['steps']['total']['cpuMs'],
                     published['steps']['phases']['work']):
            for operation in (lambda d:d.__setitem__('new',1),lambda d:d.__delitem__(next(iter(d))),
                              lambda d:d.clear(),lambda d:d.pop(next(iter(d))),lambda d:d.popitem(),
                              lambda d:d.setdefault('new',1),lambda d:d.update({'new':1}),
                              lambda d:d.__ior__({'new':1}),lambda d:d.__init__({'new':1})):
                with self.assertRaises(TypeError):operation(node)
        mutable=timing.snapshot('lm')
        mutable['steps']['trend'].append({'step':999})
        mutable['steps']['trend'][0]['step']=999
        self.assertEqual(frozen_json,json.dumps(published,sort_keys=True))
        events.ready=True;timing.drain()
        self.assertIsNone(published['steps']['trend'][0]['gpuMs'])
        self.assertIsNotNone(timing.published_snapshot('lm')['steps']['trend'][0]['gpuMs'])
        timing.begin_step('lm',2).end()
        timing.reset('lm')
        self.assertEqual(frozen_json,json.dumps(published,sort_keys=True))
        self.assertEqual(0,timing.published_snapshot('lm')['steps']['count'])


if __name__ == '__main__':
    unittest.main()
