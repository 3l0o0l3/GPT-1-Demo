"""One bounded training worker with lock-independent immutable CPU snapshots.

Only this coordinator owns background-job state. TrainingLab remains the owner
of model/optimizer/RNG state, accessed exclusively through its public methods.
"""
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime, timezone
import math
import threading
import time
import uuid


ACTIVE = frozenset(('running', 'pause_requested'))
MAX_REQUESTS = 64


class TrainingJobError(Exception):
    def __init__(self, status, code, message):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def _date(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def _step(state):
    value = state.get('step') if isinstance(state, dict) else None
    if type(value) is not int or value < 0:
        raise RuntimeError('The engine returned an invalid optimizer step.')
    return value


class TrainingJobs:
    def __init__(self, model_lock, get_lab, clock=time.time,
                 monotonic=time.monotonic, thread_factory=threading.Thread):
        self.model_lock, self.get_lab = model_lock, get_lab
        self.clock, self.monotonic, self.thread_factory = clock, monotonic, thread_factory
        self._start_lock = threading.Lock()
        self._snapshot_lock = threading.Lock()
        self._snapshot = None
        self._run = None
        self._requests = OrderedDict()

    def _reference(self):
        with self._snapshot_lock:
            return self._snapshot, self._run

    def active(self):
        with self._snapshot_lock:
            return self._run is not None

    def _copy(self, snapshot, run):
        # No model lock or snapshot lock is held while a potentially large
        # history is copied. Published dictionaries are never mutated again.
        value = deepcopy(snapshot)
        if value is not None and run is not None and value['id'] == run['id']:
            self._timing(value, run)
        return {'job': value}

    def get(self):
        return self._copy(*self._reference())

    def cached_state(self, action):
        with self._snapshot_lock:
            snapshot = self._snapshot if self._run is not None else None
        if snapshot is None:
            return None
        key = 'liveState' if action == 'state' else 'transferState'
        return deepcopy(snapshot[key])

    def _timing(self, snapshot, run):
        elapsed = max(0.0, self.monotonic() - run['started'])
        if not math.isfinite(elapsed):
            elapsed = 0.0
        rate = snapshot['completedSteps'] / elapsed if elapsed > 0 else 0.0
        snapshot['elapsedSeconds'] = elapsed
        snapshot['stepsPerSecond'] = rate if math.isfinite(rate) else 0.0

    def start(self, kind, steps, learning_rate, request_id):
        fingerprint = (kind, steps, learning_rate)
        with self._start_lock:
            previous, previous_run = self._reference()
            if request_id in self._requests:
                if self._requests[request_id] != fingerprint:
                    raise TrainingJobError(409, 'request_conflict', 'This request ID was already used with different training settings.')
                if previous is not None and previous['requestId'] == request_id:
                    return self._copy(previous, previous_run)
                raise TrainingJobError(409, 'stale_request', 'This training request already finished. Refresh progress before starting a new request.')
            if not self.model_lock.acquire(blocking=False):
                raise TrainingJobError(409, 'training_busy', 'A model operation is running. Follow its progress or pause it before starting another.')
            launched = False
            published = False
            try:
                # A predecessor may have finished between our deduplication
                # lookup and lock acquisition. Rollback must restore that
                # terminal snapshot, never resurrect its old running state.
                previous, previous_run = self._reference()
                lab = self.get_lab()
                live = lab.state()
                transfer = lab.transfer_state()
                if kind == 'transfer' and not transfer.get('available'):
                    raise TrainingJobError(409, 'invalid_training_state', 'Start a live transfer pair before fine-tuning.')
                initial = _step(live if kind == 'lm' else transfer)
                now = _date(self.clock())
                job_id = str(uuid.uuid4())
                run = {'id': job_id, 'started': self.monotonic(), 'pause': threading.Event()}
                snapshot = deepcopy({
                    'id': job_id, 'requestId': request_id, 'kind': kind, 'status': 'running',
                    'requestedSteps': steps, 'completedSteps': 0,
                    'initialStep': initial, 'currentStep': initial, 'learningRate': learning_rate,
                    'startedAt': now, 'updatedAt': now, 'elapsedSeconds': 0.0, 'stepsPerSecond': 0.0,
                    'liveState': live, 'transferState': transfer, 'error': None,
                })
                worker = self.thread_factory(target=self._work, args=(lab, run),
                                             name='gpt-training-job-' + job_id[:8], daemon=True)
                with self._snapshot_lock:
                    self._snapshot, self._run = snapshot, run
                published = True
                self._requests[request_id] = fingerprint
                worker.start()
                launched = True
                while len(self._requests) > MAX_REQUESTS:
                    self._requests.popitem(last=False)
                # Holding the start lock prevents another completed job being
                # replaced before this caller receives its own job identity.
                return self.get()
            except TrainingJobError:
                raise
            except (ImportError, ModuleNotFoundError):
                raise TrainingJobError(503, 'training_not_ready', 'Training dependencies are unavailable.') from None
            except FileNotFoundError:
                raise TrainingJobError(404, 'artifact_unavailable', 'The required training data or checkpoint is unavailable.') from None
            except Exception:
                raise TrainingJobError(503, 'job_start_failed', 'The training task could not start. Refresh model state before trying again.') from None
            finally:
                if not launched:
                    with self._snapshot_lock:
                        if published:
                            self._snapshot, self._run = previous, previous_run
                            self._requests.pop(request_id, None)
                        self.model_lock.release()

    def pause(self, job_id):
        with self._snapshot_lock:
            snapshot, run = self._snapshot, self._run
            if snapshot is None or snapshot['id'] != job_id:
                raise TrainingJobError(409, 'job_conflict', 'The requested training task is no longer current. Refresh progress.')
            if run is not None:
                run['pause'].set()
                # Shallow replacement is safe: nested states are immutable.
                snapshot = dict(snapshot, status='pause_requested', updatedAt=_date(self.clock()))
                self._snapshot = snapshot
        return self._copy(snapshot, run)

    def _publish(self, run, live, transfer, status=None, error=None):
        prior, _ = self._reference()
        snapshot = dict(prior)
        snapshot['liveState'], snapshot['transferState'] = deepcopy(live), deepcopy(transfer)
        snapshot['currentStep'] = _step(live if prior['kind'] == 'lm' else transfer)
        snapshot['completedSteps'] = snapshot['currentStep'] - snapshot['initialStep']
        if snapshot['completedSteps'] < 0:
            raise RuntimeError('The engine optimizer step moved backwards.')
        snapshot['updatedAt'] = _date(self.clock())
        snapshot['error'] = error
        self._timing(snapshot, run)
        with self._snapshot_lock:
            snapshot['status'] = status or ('pause_requested' if run['pause'].is_set() else 'running')
            self._snapshot = snapshot
            if status is not None:
                # Terminal snapshots become visible only once model operations
                # may safely resume; no reader sees an early completed flag.
                self.model_lock.release()
                self._run = None

    def _work(self, lab, run):
        initial, _ = self._reference()
        current = initial
        terminal = False
        try:
            while current['completedSteps'] < current['requestedSteps'] and not run['pause'].is_set():
                count = min(10, current['requestedSteps'] - current['completedSteps'])
                before = current['currentStep']
                method = lab.train if current['kind'] == 'lm' else lab.train_transfer
                method(steps=count, learning_rate=current['learningRate'])
                live, transfer = lab.state(), lab.transfer_state()
                actual = _step(live if current['kind'] == 'lm' else transfer)
                if actual != before + count:
                    raise RuntimeError('The engine did not complete the requested optimizer steps.')
                self._publish(run, live, transfer)
                current, _ = self._reference()
            status = 'completed' if current['completedSteps'] == current['requestedSteps'] else 'paused'
            self._publish(run, lab.state(), lab.transfer_state(), status=status)
            terminal = True
        except Exception:
            error = {'code': 'training_failed',
                     'message': 'Training stopped after an error. The latest confirmed model state is shown; refresh before starting another task.'}
            try:
                self._publish(run, lab.state(), lab.transfer_state(), status='failed', error=error)
                terminal = True
            except Exception:
                # A broken state accessor cannot strand the model lock. Retain
                # the last confirmed snapshot rather than fabricate progress.
                with self._snapshot_lock:
                    self._snapshot = dict(self._snapshot, status='failed', error={
                        'code': 'training_state_unavailable',
                        'message': 'Training failed and its final state could not be read. Refresh model state before further work.'})
                    self.model_lock.release()
                    self._run = None
                    terminal = True
        finally:
            if not terminal:
                with self._snapshot_lock:
                    self._snapshot = dict(self._snapshot, status='failed', error={
                        'code': 'training_interrupted', 'message': 'The background training task was interrupted.'})
                    self.model_lock.release()
                    self._run = None
