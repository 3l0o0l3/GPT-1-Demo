"""Bounded, runtime-only CPU/CUDA diagnostics; see TRAINING-TIMING-CONTRACT.md."""
from __future__ import annotations

from collections import deque
from contextlib import nullcontext
import copy
import math
import threading
import time


class _NullSample:
    sampled = False

    def phase(self, name):
        return nullcontext()

    def end(self, success=True):
        return None


NULL_SAMPLE = _NullSample()


class _FrozenDict(dict):
    """JSON-compatible read-only value; copies safely share immutable children."""
    __slots__ = ()

    def __init__(self, *args, **kwargs):
        raise TypeError('Published timing values are immutable.')

    @classmethod
    def _from_items(cls, items):
        result = dict.__new__(cls)
        dict.__init__(result, items)
        return result

    def _reject(self, *args, **kwargs):
        raise TypeError('Published timing values are immutable.')

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _reject

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


def _freeze(value):
    if isinstance(value, _FrozenDict):
        return value
    if isinstance(value, dict):
        return _FrozenDict._from_items((key, _freeze(item)) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _mutable(value):
    if isinstance(value, dict):
        return {key: _mutable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_mutable(item) for item in value]
    return value


class _CudaEvent:
    """The only torch dependency; construction never reads model state."""
    def __init__(self, device):
        import torch
        self._torch, self._device = torch, device
        self._event = torch.cuda.Event(enable_timing=True)

    def record(self):
        self._event.record(self._torch.cuda.current_stream(device=self._device))

    def query(self):
        return self._event.query()

    def elapsed_time(self, other):
        return self._event.elapsed_time(other._event)


def _stats(values):
    if not values:
        return {'count': 0, 'mean': None, 'p95': None, 'min': None, 'max': None}
    values = sorted(values)
    return {'count': len(values), 'mean': math.fsum(values) / len(values),
            'p95': values[math.ceil(len(values) * .95) - 1],
            'min': values[0], 'max': values[-1]}


def _summary(records):
    successful = [row for row in records if row['success']]
    names = sorted({name for row in successful for name in row['phases']})
    total = {key: _stats([row[key] for row in successful if row[key] is not None])
             for key in ('cpuMs', 'gpuMs')}
    phases = {name: {key: _stats([row['phases'][name][key] for row in successful
                                if name in row['phases'] and row['phases'][name][key] is not None])
                     for key in ('cpuMs', 'gpuMs')} for name in names}
    return {'count': len(records), 'successfulCount': len(successful),
            'failedCount': len(records) - len(successful), 'total': total,
            'phases': phases,
            'trend': [{key: row[key] for key in ('step', 'success', 'cpuMs', 'gpuMs')}
                      for row in records]}


class _Phase:
    def __init__(self, sample, name):
        self._sample, self._name = sample, name

    def __enter__(self):
        self._start = self._sample._owner._now(self._sample._kind)
        self._event = self._sample._event()
        return self

    def __exit__(self, exc_type, exc, traceback):
        sample = self._sample
        end_event = sample._event()
        elapsed = sample._owner._elapsed(self._start, sample._kind)
        if not sample._ended:
            previous = sample._phases.get(self._name, 0.)
            sample._phases[self._name] = (previous + elapsed
                if previous is not None and elapsed is not None else None)
            if self._event is not None and end_event is not None:
                sample._events.append((self._name, self._event, end_event))
        return False


class _Sample:
    sampled = True

    def __init__(self, owner, kind, step, family, epoch, gpu_reserved):
        self._owner, self._kind, self._step = owner, kind, step
        self._family, self._epoch = family, epoch
        self._gpu_reserved, self._gpu_failed = gpu_reserved, False
        self._ended = False
        self._phases, self._events = {}, []
        self._names = set()
        self._start = owner._now(kind)
        self._start_event = self._event()
        self._end_event = None

    def _event(self):
        if not self._gpu_reserved or self._gpu_failed:
            return None
        try:
            event = self._owner._event_factory()
            event.record()
            return event
        except Exception:
            self._fail_gpu()
            return None

    def _fail_gpu(self):
        if not self._gpu_failed:
            self._gpu_failed = True
            self._events.clear()
            self._owner._gpu_failure(self)

    def phase(self, name):
        if self._ended:
            return nullcontext()
        if not isinstance(name, str) or not name or len(name) > 80:
            raise ValueError('Timing phase names must contain 1-80 characters.')
        if name not in self._names and len(self._names) >= 32:
            raise ValueError('A timing sample supports at most 32 distinct phases.')
        self._names.add(name)
        return _Phase(self, name)

    def end(self, success=True):
        if self._ended:
            return None
        if type(success) is not bool:
            raise ValueError('Timing success must be a boolean.')
        self._ended = True
        self._end_event = self._event()
        row = {'step': self._step, 'success': success,
               'cpuMs': self._owner._elapsed(self._start, self._kind), 'gpuMs': None,
               'phases': {name: {'cpuMs': elapsed, 'gpuMs': None}
                          for name, elapsed in self._phases.items()}}
        self._owner._finish(self, row)

    def _gpu_values(self):
        # Called only by the serialized owner, outside the CPU publication lock.
        if not self._end_event.query():
            return None
        total = float(self._start_event.elapsed_time(self._end_event))
        phases = {name: 0. for name in self._phases}
        for name, start, end in self._events:
            phases[name] += float(start.elapsed_time(end))
        if any(not math.isfinite(value) or value < 0 for value in [total, *phases.values()]):
            raise ValueError('CUDA timing returned an invalid duration.')
        return total, phases


class TrainingTiming:
    """One serialized writer, concurrent CPU-only snapshot readers."""
    def __init__(self, device, sample_every=20, capacity=200, max_pending=32, *,
                 enabled=True, clock=time.perf_counter, event_factory=None):
        for name, value in (('sample_every', sample_every), ('capacity', capacity),
                            ('max_pending', max_pending)):
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer.')
        if type(enabled) is not bool:
            raise ValueError('enabled must be a boolean.')
        self.device, self.sample_every = device, sample_every
        self.capacity, self.max_pending, self.enabled = capacity, max_pending, enabled
        self._clock, self._lock = clock, threading.Lock()
        device_type = getattr(device, 'type', str(device).split(':')[0])
        self._gpu_supported = enabled and device_type == 'cuda'
        self._event_factory = event_factory or (lambda: _CudaEvent(device))
        self._states = {}
        self.reset()

    @staticmethod
    def _kind(kind):
        if kind not in ('lm', 'transfer'):
            raise ValueError('Timing kind must be lm or transfer.')

    def _clock_failure(self, kind):
        with self._lock:
            self._states[kind]['counters']['clockErrors'] += 1
            self._metadata_locked(kind)

    def _now(self, kind):
        try:
            value = float(self._clock())
            if math.isfinite(value):
                return value
        except Exception:
            pass
        self._clock_failure(kind)
        return None

    def _elapsed(self, start, kind):
        end = self._now(kind)
        if start is None or end is None:
            return None
        elapsed = (end - start) * 1000.
        if math.isfinite(elapsed) and elapsed >= 0:
            return elapsed
        self._clock_failure(kind)
        return None

    def _metadata_locked(self, kind):
        state = self._states[kind]
        # Only small outer metadata changes per observed update. The bounded
        # phase/trend trees were frozen once when their records last changed.
        state['published'] = _FrozenDict._from_items({
            'version': 1, 'kind': kind, 'enabled': self.enabled,
            'sampleEvery': self.sample_every, 'capacity': self.capacity,
            'window': 'last-capacity-records', 'summaryScope': 'successful-records-only',
            'gpuSupported': self._gpu_supported, 'pendingGpuSamples': state['reserved'],
            'counters': _freeze(state['counters']), **state['groups']})

    def _publish_locked(self, kind):
        state = self._states[kind]
        state['groups'] = {family: _freeze(_summary(state[family]))
                           for family in ('steps', 'evaluations')}
        self._metadata_locked(kind)

    def reset(self, kind=None):
        kinds = ('lm', 'transfer') if kind is None else (kind,)
        for selected in kinds:
            self._kind(selected)
        with self._lock:
            for selected in kinds:
                epoch = self._states.get(selected, {}).get('epoch', 0) + 1
                self._states[selected] = {
                    'epoch': epoch, 'steps': deque(maxlen=self.capacity),
                    'evaluations': deque(maxlen=self.capacity), 'pending': [], 'reserved': 0,
                    'counters': {key: 0 for key in ('observedSteps', 'sampledSteps',
                        'successfulSamples', 'failedSamples', 'evaluations',
                        'gpuDroppedSamples', 'eventErrors', 'clockErrors')}}
                self._publish_locked(selected)

    def begin_step(self, kind, step):
        return self._begin(kind, step, 'steps')

    def begin_evaluation(self, kind, step):
        return self._begin(kind, step, 'evaluations')

    def _begin(self, kind, step, family):
        self._kind(kind)
        if type(step) is not int or step < 0:
            raise ValueError('Timing step must be a nonnegative integer.')
        if not self.enabled:
            return NULL_SAMPLE
        with self._lock:
            state = self._states[kind]
            counters = state['counters']
            if family == 'steps':
                counters['observedSteps'] += 1
                sampled = counters['observedSteps'] == 1 or step % self.sample_every == 0
                counters['sampledSteps'] += int(sampled)
            else:
                counters['evaluations'] += 1
                sampled = True
            reserved = False
            if sampled and self._gpu_supported:
                if sum(item['reserved'] for item in self._states.values()) < self.max_pending:
                    state['reserved'] += 1
                    reserved = True
                else:
                    counters['gpuDroppedSamples'] += 1
            epoch = state['epoch']
            self._metadata_locked(kind)
        return _Sample(self, kind, step, family, epoch, reserved) if sampled else NULL_SAMPLE

    def _gpu_failure(self, sample):
        with self._lock:
            state = self._states[sample._kind]
            if sample._epoch == state['epoch']:
                state['counters']['eventErrors'] += 1
                if sample._gpu_reserved:
                    state['reserved'] -= 1
                self._metadata_locked(sample._kind)
            sample._gpu_reserved = False

    def _finish(self, sample, row):
        with self._lock:
            state = self._states[sample._kind]
            if sample._epoch != state['epoch']:
                return
            state[sample._family].append(row)
            state['counters']['successfulSamples' if row['success'] else 'failedSamples'] += 1
            if sample._gpu_reserved and not sample._gpu_failed:
                state['pending'].append((sample, row))
            self._publish_locked(sample._kind)

    def drain(self):
        with self._lock:
            pending = [item for state in self._states.values() for item in state['pending']]
        changed = set()
        for sample, row in pending:
            try:
                values = sample._gpu_values()
                if values is None:
                    continue
            except Exception:
                sample._fail_gpu()
                values = None
            with self._lock:
                state = self._states[sample._kind]
                if sample._epoch != state['epoch']:
                    continue
                state['pending'].remove((sample, row))
                if sample._gpu_reserved:
                    state['reserved'] -= 1
                    sample._gpu_reserved = False
                if values is not None:
                    row['gpuMs'], phases = values
                    for name, value in phases.items():
                        row['phases'][name]['gpuMs'] = value
                changed.add(sample._kind)
        if changed:
            with self._lock:
                for kind in changed:
                    self._publish_locked(kind)

    def snapshot(self, kind):
        self._kind(kind)
        with self._lock:
            return _mutable(self._states[kind]['published'])

    def published_snapshot(self, kind):
        """Stable, deeply read-only JSON value; deep-copying may share it safely."""
        self._kind(kind)
        with self._lock:
            return self._states[kind]['published']
