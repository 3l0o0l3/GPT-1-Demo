"""Worker/control concurrency with deterministic gates; no GPU or live state."""
from copy import deepcopy
import threading
import time
import unittest
from unittest.mock import patch
import uuid

from training_jobs import TrainingJobs, TrainingJobError
from training_service import TrainingService, TrainingApiError


def identifier():
    return str(uuid.uuid4())


class JobLab:
    def __init__(self, **kwargs):
        self.step = 20
        self.transfer_step = 4
        self.history = [{'step':20,'trainLoss':2.0,'valLoss':2.5}]
        self.transfer_history = [{'step':4,'trainLoss':.7,'valLoss':.8,'accuracy':.5}]
        self.calls = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.fail_after = None
        self.available = True
        self.state_reads = 0

    def state(self):
        self.state_reads += 1
        return {'step':self.step, 'mode':'live', 'source':'random',
                'history':self.history, 'metrics':dict(self.history[-1])}

    def transfer_state(self):
        return {'available':self.available, 'step':self.transfer_step, 'runId':'fixture-pair',
                'branches':[{'name':name,'history':self.transfer_history,'metrics':dict(self.transfer_history[-1])}
                            for name in ('pretrained','scratch')]}

    def metadata(self): return {'architecture':{'parameters':123}}
    def recordings(self): return {'pretraining':{'available':False}}
    def examples(self, **kwargs): return {'examples':[], 'requested':kwargs}
    def inspect(self, text, **kwargs): return {'text':text, **kwargs}
    def reset(self, **kwargs):
        self.step = 0
        self.history = [{'step':0,'trainLoss':8.0,'valLoss':8.0}]
        return self.state()

    def _train(self, kind, steps, learning_rate):
        self.calls.append((kind,steps,learning_rate))
        self.entered.set()
        if not self.release.wait(5):
            raise RuntimeError('Fixture gate timed out')
        field = 'step' if kind == 'lm' else 'transfer_step'
        for i in range(steps):
            setattr(self,field,getattr(self,field)+1)
            if self.fail_after is not None and i + 1 == self.fail_after:
                raise RuntimeError('private-engine-error fixture must not escape')
        if kind == 'lm':
            self.history.append({'step':self.step,'trainLoss':1.9,'valLoss':2.4})
            return self.state()
        self.transfer_history.append({'step':self.transfer_step,'trainLoss':.6,'valLoss':.7,'accuracy':.6})
        return self.transfer_state()

    def train(self, steps, learning_rate): return self._train('lm',steps,learning_rate)
    def train_transfer(self, steps, learning_rate): return self._train('transfer',steps,learning_rate)


class TrainingJobTests(unittest.TestCase):
    def setUp(self):
        self.lab = JobLab()
        self.lock = threading.Lock()
        self.jobs = TrainingJobs(self.lock, lambda:self.lab)

    def tearDown(self):
        self.lab.release.set()
        self.finish()

    def start(self, kind='lm', steps=1000, request_id=None):
        return self.jobs.start(kind,steps,.0003,request_id or identifier())['job']

    def finish(self):
        deadline = time.monotonic() + 5
        while self.jobs.active() and time.monotonic() < deadline:
            threading.Event().wait(.005)
        self.assertFalse(self.jobs.active(), 'Worker did not finish')
        return self.jobs.get()['job']

    def test_one_request_performs_exactly_1000_steps_and_preserves_eval_points(self):
        job = self.start()
        done = self.finish()
        self.assertEqual(done['id'],job['id'])
        self.assertEqual((done['status'],done['completedSteps'],done['currentStep']),('completed',1000,1020))
        self.assertEqual(len(self.lab.calls),100)
        self.assertTrue(all(call==('lm',10,.0003) for call in self.lab.calls))
        self.assertEqual([p['step'] for p in done['liveState']['history']],list(range(20,1021,10)))
        self.assertEqual(done['transferState']['step'],4)
        self.assertTrue(self.lock.acquire(blocking=False))
        self.lock.release()

    def test_one_request_performs_exactly_5000_steps_with_unchanged_internal_blocks(self):
        job = self.start(steps=5000)
        done = self.finish()
        self.assertEqual(done['id'],job['id'])
        self.assertEqual((done['status'],done['requestedSteps'],done['completedSteps'],done['currentStep']),
                         ('completed',5000,5000,5020))
        self.assertEqual(len(self.lab.calls),500)
        self.assertTrue(all(call==('lm',10,.0003) for call in self.lab.calls))
        self.assertEqual([point['step'] for point in done['liveState']['history']],list(range(20,5021,10)))
        self.assertEqual(done['transferState']['step'],4)
        self.assertTrue(self.lock.acquire(blocking=False));self.lock.release()

    def test_transfer_uses_paired_steps_and_leaves_language_model_unchanged(self):
        self.start('transfer',23)
        done = self.finish()
        self.assertEqual((done['completedSteps'],done['initialStep'],done['currentStep']),(23,4,27))
        self.assertEqual([call[1] for call in self.lab.calls],[10,10,3])
        self.assertTrue(all(call[0]=='transfer' for call in self.lab.calls))
        self.assertEqual(done['liveState']['step'],20)
        self.assertEqual(done['transferState']['branches'][0]['history'][-1]['step'],27)

    def test_progress_pause_and_cached_state_do_not_wait_for_blocked_training(self):
        self.lab.release.clear()
        request_id=identifier()
        job = self.start(steps=5000,request_id=request_id)
        self.assertTrue(self.lab.entered.wait(1))
        self.assertEqual(self.start(steps=5000,request_id=request_id)['id'],job['id'])
        returned = threading.Event()
        results = []
        def observe():
            results.append(self.jobs.get()['job'])
            results.append(self.jobs.cached_state('state'))
            results.append(self.jobs.pause(job['id'])['job'])
            returned.set()
        observer = threading.Thread(target=observe)
        observer.start()
        self.assertTrue(returned.wait(1),'Reading progress or requesting pause waited for training')
        observer.join()
        self.assertEqual(results[0]['completedSteps'],0)
        self.assertEqual(results[1]['step'],20)
        self.assertEqual(results[2]['status'],'pause_requested')
        self.assertFalse(self.lock.acquire(blocking=False))
        self.lab.release.set()
        done = self.finish()
        self.assertEqual((done['status'],done['completedSteps']),('paused',10))
        self.assertEqual(len(self.lab.calls),1)
        self.assertEqual(self.jobs.pause(job['id'])['job']['status'],'paused')

    def test_snapshot_consumer_cannot_mutate_worker_state(self):
        self.lab.release.clear()
        self.start(steps=10)
        value = self.jobs.get()['job']
        value['liveState']['history'][0]['trainLoss'] = -99
        value['transferState']['branches'].clear()
        self.assertEqual(self.jobs.get()['job']['liveState']['history'][0]['trainLoss'],2.0)
        self.assertEqual(len(self.jobs.get()['job']['transferState']['branches']),2)
        self.assertEqual(self.lab.history[0]['trainLoss'],2.0)

    def test_slow_snapshot_copy_does_not_hold_publisher_or_control_lock(self):
        self.lab.release.clear()
        job = self.start(steps=10)
        copying, allow_copy = threading.Event(), threading.Event()
        def controlled_copy(value):
            if threading.current_thread().name == 'slow-snapshot-reader':
                copying.set()
                if not allow_copy.wait(3):
                    raise RuntimeError('Slow reader fixture timed out')
            return deepcopy(value)
        with patch('training_jobs.deepcopy',side_effect=controlled_copy):
            reader = threading.Thread(target=self.jobs.get,name='slow-snapshot-reader')
            reader.start()
            self.assertTrue(copying.wait(1))
            self.assertEqual(self.jobs.pause(job['id'])['job']['status'],'pause_requested')
            self.lab.release.set()
            self.assertEqual(self.finish()['status'],'completed')
            allow_copy.set()
            reader.join(1)
            self.assertFalse(reader.is_alive())

    def test_duplicate_request_never_replays_and_changed_or_old_ids_conflict(self):
        request_id = identifier()
        self.lab.release.clear()
        first = self.start(steps=10,request_id=request_id)
        repeated = self.start(steps=10,request_id=request_id)
        self.assertEqual(first['id'],repeated['id'])
        with self.assertRaises(TrainingJobError) as changed:
            self.start(steps=11,request_id=request_id)
        self.assertEqual(changed.exception.code,'request_conflict')
        self.lab.release.set();done = self.finish()
        self.assertEqual(self.start(steps=10,request_id=request_id)['id'],done['id'])
        self.assertEqual(len(self.lab.calls),1)
        self.start(steps=1);self.finish()
        with self.assertRaises(TrainingJobError) as stale:
            self.start(steps=10,request_id=request_id)
        self.assertEqual(stale.exception.code,'stale_request')
        self.assertEqual(len(self.lab.calls),2)

    def test_competing_starts_reserve_exactly_one_worker(self):
        self.lab.release.clear()
        gate = threading.Barrier(3)
        replies, errors = [], []
        def contender():
            gate.wait()
            try:replies.append(self.start(steps=10))
            except TrainingJobError as error:errors.append(error.code)
        threads = [threading.Thread(target=contender) for _ in range(2)]
        for thread in threads:thread.start()
        gate.wait()
        for thread in threads:thread.join(1)
        self.assertEqual(len(replies),1)
        self.assertEqual(errors,['training_busy'])
        self.assertEqual(len(self.lab.calls),1)

    def test_failure_reports_partial_real_progress_and_hides_private_exception(self):
        self.lab.fail_after = 3
        self.start(steps=1000)
        failed = self.finish()
        self.assertEqual((failed['status'],failed['completedSteps'],failed['currentStep']),('failed',3,23))
        self.assertEqual(failed['liveState']['history'][-1]['step'],20)
        self.assertNotIn('private-engine-error',str(failed))
        self.lab.fail_after = None
        self.start(steps=1)
        self.assertEqual(self.finish()['currentStep'],24)

    def test_failed_initialization_and_thread_start_release_model_lock(self):
        def broken_factory(): raise ImportError('private import details')
        jobs = TrainingJobs(self.lock,broken_factory)
        with self.assertRaises(TrainingJobError) as failure:jobs.start('lm',10,.0003,identifier())
        self.assertEqual(failure.exception.code,'training_not_ready')
        self.assertIsNone(jobs.get()['job'])
        class BrokenThread:
            def __init__(self,**kwargs): pass
            def start(self):raise RuntimeError('private thread details')
        jobs = TrainingJobs(self.lock,lambda:self.lab,thread_factory=BrokenThread)
        with self.assertRaises(TrainingJobError) as failure:jobs.start('lm',10,.0003,identifier())
        self.assertEqual(failure.exception.code,'job_start_failed')
        self.assertIsNone(jobs.get()['job'])
        self.assertTrue(self.lock.acquire(blocking=False));self.lock.release()

    def test_failed_launch_restores_finished_predecessor_when_it_finishes_during_acquire(self):
        old_released = threading.Event()
        lab, underlying = self.lab, self.lock
        class InterleavedLock:
            finish_old_before_acquire = False
            def acquire(self, blocking=False):
                if self.finish_old_before_acquire:
                    self.finish_old_before_acquire = False
                    # start() has already read the old running snapshot. Allow
                    # that worker to finish before the new lock reservation.
                    lab.release.set()
                    if not old_released.wait(2):
                        raise RuntimeError('Old worker did not release its lock')
                return underlying.acquire(blocking=blocking)
            def release(self):
                underlying.release()
                old_released.set()
        class BrokenThread:
            def __init__(self, **kwargs):pass
            def start(self):raise RuntimeError('New worker could not start')
        lock = InterleavedLock()
        jobs = TrainingJobs(lock, lambda:lab)
        lab.release.clear()
        first = jobs.start('lm',10,.0003,identifier())['job']
        self.assertTrue(lab.entered.wait(1))
        lock.finish_old_before_acquire = True
        jobs.thread_factory = BrokenThread
        with self.assertRaises(TrainingJobError) as failure:
            jobs.start('lm',10,.0003,identifier())
        self.assertEqual(failure.exception.code,'job_start_failed')
        restored = jobs.get()['job']
        self.assertEqual(restored['id'],first['id'])
        self.assertEqual((restored['status'],restored['completedSteps'],restored['currentStep']),
                         ('completed',10,30))
        self.assertFalse(jobs.active())
        self.assertIsNone(jobs.cached_state('state'))
        self.assertTrue(self.lock.acquire(blocking=False));self.lock.release()

    def test_unknown_pause_and_missing_transfer_do_not_start_work(self):
        with self.assertRaises(TrainingJobError):self.jobs.pause(identifier())
        self.lab.available = False
        with self.assertRaises(TrainingJobError) as failure:self.start('transfer')
        self.assertEqual(failure.exception.status,409)
        self.assertEqual(self.lab.calls,[])
        self.assertTrue(self.lock.acquire(blocking=False));self.lock.release()

    def test_engine_stall_fails_instead_of_spinning_or_claiming_completion(self):
        self.lab.train = lambda **kwargs:self.lab.state()
        self.start(steps=1000)
        failed=self.finish()
        self.assertEqual((failed['status'],failed['completedSteps']),('failed',0))
        self.assertTrue(self.lock.acquire(blocking=False));self.lock.release()

    def test_broken_final_state_keeps_last_confirmed_snapshot_and_releases_lock(self):
        original=self.lab.state
        def unreliable_state():
            if self.lab.step>20:raise RuntimeError('private state details')
            return original()
        self.lab.state=unreliable_state
        self.start(steps=10)
        failed=self.finish()
        self.assertEqual(failed['status'],'failed')
        self.assertEqual(failed['currentStep'],20)
        self.assertEqual(failed['error']['code'],'training_state_unavailable')
        self.assertNotIn('private state details',str(failed))
        self.assertTrue(self.lock.acquire(blocking=False));self.lock.release()


class BackgroundServiceTests(unittest.TestCase):
    def setUp(self):
        self.lab = JobLab()
        self.lab.release.clear()
        self.service = TrainingService('.',factory=lambda **kwargs:self.lab)

    def tearDown(self):
        self.lab.release.set()
        deadline = time.monotonic()+5
        while self.service.jobs.active() and time.monotonic()<deadline:
            threading.Event().wait(.005)
        self.assertFalse(self.service.jobs.active())

    def dispatch(self,method,action,body=None):
        return self.service.dispatch(method,'/api/training/'+action,body)

    def payload(self,**kwargs):
        return dict({'kind':'lm','steps':1000,'learningRate':.0003,'requestId':identifier()},**kwargs)

    def test_boot_reads_and_job_pause_remain_available_without_touching_model(self):
        job = self.dispatch('POST','job-start',self.payload(steps=5000))['job']
        self.assertEqual(job['requestedSteps'],5000)
        self.assertTrue(self.lab.entered.wait(1))
        reads = self.lab.state_reads
        self.assertEqual(self.dispatch('GET','state')['step'],20)
        self.assertEqual(self.lab.state_reads,reads)
        self.assertEqual(self.dispatch('GET','transfer-state')['step'],4)
        self.assertEqual(self.dispatch('GET','meta')['architecture']['parameters'],123)
        self.assertEqual(self.dispatch('GET','examples?kind=lm&split=train&limit=3')['requested']['limit'],3)
        self.assertFalse(self.dispatch('GET','recordings')['pretraining']['available'])
        self.assertEqual(self.dispatch('POST','job-pause',{'jobId':job['id']})['job']['status'],'pause_requested')
        for action,body in (('inspect',{'text':'x'}),('train',{'steps':1}),('reset',{}),('save-live',{})):
            with self.assertRaises(TrainingApiError) as busy:self.dispatch('POST',action,body)
            self.assertEqual(busy.exception.code,'training_busy')

    def test_terminal_snapshot_does_not_mask_later_manual_reset(self):
        self.lab.release.set()
        self.dispatch('POST','job-start',self.payload(steps=1))
        deadline=time.monotonic()+2
        while self.dispatch('GET','job')['job']['status'] in ('running','pause_requested') and time.monotonic()<deadline:
            threading.Event().wait(.002)
        # Terminal visibility itself guarantees that the model lock is free.
        self.assertEqual(self.dispatch('POST','reset',{})['step'],0)
        self.assertEqual(self.dispatch('GET','state')['step'],0)
        self.assertEqual(self.dispatch('GET','job')['job']['currentStep'],21)

    def test_idle_job_reads_never_initialize_model_and_validation_is_strict(self):
        self.assertEqual(self.dispatch('GET','job'),{'job':None})
        for field,values in (('steps',(None,True,0,5001,1.5,'10')),
                             ('kind',(None,True,'other')),
                             ('learningRate',(None,True,float('nan'),0,.02)),
                             ('requestId',(None,True,'abc','0'*32))):
            for value in values:
                with self.subTest(field=field,value=value):
                    with self.assertRaises(TrainingApiError) as error:self.dispatch('POST','job-start',self.payload(**{field:value}))
                    self.assertEqual(error.exception.status,400)
        for method,route,body in (('GET','job?extra=1',None),('GET','job-start',None),
                                   ('POST','job',{}),('POST','job-pause',{}),
                                   ('POST','job-pause',{'jobId':identifier(),'extra':1})):
            with self.assertRaises(TrainingApiError):self.dispatch(method,route,body)
        self.assertIsNone(self.service.lab)


if __name__=='__main__':unittest.main()
