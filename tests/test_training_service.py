import unittest
from pathlib import Path
from training_service import TrainingService, TrainingApiError


class FakeLab:
    def __init__(self, **kwargs): self.calls=[]
    def metadata(self): return {'architecture':{'parameters':12}}
    def state(self): return {'step':0}
    def recordings(self): return {'pretraining':{'available':False}}
    def examples(self, **kwargs): return {'examples':[], 'requested':kwargs}
    def inspect(self, text, **kwargs): return {'text':text, **kwargs}
    def preview(self, **kwargs): return kwargs
    def train(self, **kwargs): self.calls.append(kwargs);return {'step':kwargs['steps']}
    def commit(self, preview_id): raise ValueError('Preview is stale or already applied.')
    def reset(self, **kwargs): return kwargs
    def predict(self, text, **kwargs): return {'text':text, **kwargs}
    def generate(self, text, **kwargs): return {'text':text, **kwargs}
    def transfer_state(self): return {'available':False}
    def start_transfer(self, **kwargs): self.calls.append(('start_transfer',kwargs));return kwargs
    def train_transfer(self, **kwargs): self.calls.append(('train_transfer',kwargs));return kwargs
    def predict_transfer(self, text, **kwargs): return {'text':text, **kwargs}
    def save_live(self): self.calls.append(('save_live',{}));return {'saved':True,'step':4000}
    def restore_live(self): self.calls.append(('restore_live',{}));return {'step':4000}


class ServiceTests(unittest.TestCase):
    def setUp(self): self.service=TrainingService(Path('.'), factory=FakeLab)
    def post(self, action, body): return self.service.dispatch('POST','/api/training/'+action,body)
    def reject(self, action, body, status=400):
        with self.assertRaises(TrainingApiError) as caught: self.post(action,body)
        self.assertEqual(caught.exception.status,status)
    def test_valid_requests_use_bounded_public_methods(self):
        self.assertEqual(self.post('train',{'steps':10,'learningRate':.0003})['step'],10)
        self.assertEqual(self.post('inspect',{'text':'A real sentence.'})['position'],0)
        self.assertEqual(self.post('predict',{'text':'Good film.','branch':'scratch'})['branch'],'scratch')
    def test_pretraining_and_seed_inputs_reject_invalid_types(self):
        for value in (True,0,11,3.5,'5'):
            self.reject('train',{'steps':value})
        for rate in (float('nan'),float('inf'),0,-1,True,.02):
            self.reject('train',{'learningRate':rate})
        self.reject('reset',{'source':'/tmp/a.pt'})
        self.reject('reset',{'seed':True})
    def test_unsupported_fields_and_missing_text_are_rejected(self):
        self.reject('inspect',{'text':'a','checkpoint':'/etc/passwd'})
        self.reject('inspect',{'text':' '*4})
        self.reject('inspect',{'text':'a'*8193})
        self.reject('predict',{'text':'a','branch':'unknown'})
        self.reject('commit',{})
    def test_stale_commit_is_conflict_and_releases_lock(self):
        self.reject('commit',{'previewId':'old'},409)
        self.assertEqual(self.post('train',{'steps':1})['step'],1)
    def test_busy_operations_do_not_mutate_engine(self):
        self.service.lock.acquire()
        try: self.reject('train',{'steps':1},409)
        finally: self.service.lock.release()
        self.assertIsNone(self.service.lab)
    def test_examples_exclude_test_and_bound_limit(self):
        result=self.service.dispatch('GET','/api/training/examples?kind=sentiment&split=validation&limit=5')
        self.assertEqual(result['requested'],{'kind':'sentiment','split':'validation','limit':5})
        for query in ('split=test','limit=999','kind=lm&kind=sentiment','path=a','limit=a'):
            with self.assertRaises(TrainingApiError): self.service.dispatch('GET','/api/training/examples?'+query)
    def test_unknown_routes_are_not_engine_methods(self):
        with self.assertRaises(TrainingApiError) as caught:
            self.service.dispatch('POST','/api/training/__dict__',{})
        self.assertEqual(caught.exception.status,404)

    def test_live_generation_maps_defaults_and_explicit_sampling(self):
        self.assertEqual(self.post('generate', {'text':'A film'}), {
            'text':'A film', 'max_new_tokens':48, 'temperature':.8, 'seed':42, 'model':'live',
            'sliding_window':False, 'window_size':128})
        self.assertEqual(self.post('generate', {'text':'A film', 'maxNewTokens':64,
                                              'temperature':0, 'seed':0, 'model':'random'}), {
            'text':'A film', 'max_new_tokens':64, 'temperature':0, 'seed':0, 'model':'random',
            'sliding_window':False, 'window_size':128})

    def test_live_invalid_inputs_never_initialize_engine(self):
        invalid = [
            ('generate', {'text':'x', 'maxNewTokens':value}) for value in (0,1001,True,1.5,'2')
        ] + [
            ('generate', {'text':'x', 'temperature':value}) for value in (-1,1.51,True,float('nan'),float('inf'),'0.8')
        ] + [
            ('transfer-start', {'auxiliaryWeight':value}) for value in (True,False,.1,float('nan'),'.5',None)
        ] + [
            ('transfer-train', {'steps':value}) for value in (0,11,True,'10')
        ] + [
            ('transfer-train', {'learningRate':value}) for value in (0,True,float('inf'))
        ] + [
            ('generate', {'text':' '}), ('generate', {'text':'x'*8193}),
            ('generate', {'text':'x', 'model':'pretrained'}),
            ('generate', {'text':'x', 'seed':True}),
            ('transfer-start', {'seed':-1}),
            ('transfer-predict', {'text':'good', 'branch':'auxiliary'}),
            ('transfer-predict', {'text':''}),
        ]
        for action, body in invalid:
            with self.subTest(action=action, body=body):
                self.reject(action, body)
                self.assertIsNone(self.service.lab)

    def test_live_transfer_maps_public_methods_and_preserves_recorded_route(self):
        self.assertEqual(self.service.dispatch('GET','/api/training/transfer-state'), {'available':False})
        self.assertEqual(self.post('transfer-start', {}), {'seed':42,'auxiliary_weight':0})
        self.assertEqual(self.post('transfer-start', {'seed':7,'auxiliaryWeight':.5}),
                         {'seed':7,'auxiliary_weight':.5})
        self.assertEqual(self.post('transfer-train', {}), {'steps':10,'learning_rate':.0003})
        self.assertEqual(self.post('transfer-predict', {'text':'good'}), {'text':'good','branch':'pretrained'})
        self.assertEqual(self.post('transfer-predict', {'text':'good','branch':'scratch'})['branch'], 'scratch')
        self.assertEqual(self.post('predict', {'text':'good','branch':'auxiliary'})['branch'], 'auxiliary')

    def test_live_save_restore_reject_paths_and_accept_only_empty_objects(self):
        for action in ('save-live','restore-live'):
            for body in ({'path':'other.pt'},{'filename':'session.pt'},[],None):
                with self.subTest(action=action, body=body):
                    self.reject(action, body)
                    self.assertIsNone(self.service.lab)
        self.assertEqual(self.post('save-live', {}), {'saved':True,'step':4000})
        self.assertEqual(self.post('restore-live', {}), {'step':4000})
        self.assertEqual(self.service.lab.calls, [('save_live',{}),('restore_live',{})])

    def test_live_state_routes_reject_wrong_methods_queries_and_extra_fields(self):
        self.reject('transfer-state', {}, 404)
        for action in ('generate','transfer-start','transfer-train','transfer-predict','save-live','restore-live'):
            with self.subTest(action=action):
                with self.assertRaises(TrainingApiError) as caught:
                    self.service.dispatch('GET','/api/training/'+action)
                self.assertEqual(caught.exception.status,404)
                self.reject(action, {'unexpected':True})
        for url in ('/api/training/transfer-state?path=x','/api/training/transfer-state?seed=1'):
            with self.assertRaises(TrainingApiError) as caught:
                self.service.dispatch('GET',url)
            self.assertEqual(caught.exception.status,400)
        self.assertIsNone(self.service.lab)

    def test_live_missing_pair_conflicts_missing_file_404_and_releases_lock(self):
        class MissingStateLab(FakeLab):
            def train_transfer(self, **kwargs): raise ValueError('Start live transfer first.')
            def predict_transfer(self, *args, **kwargs): raise ValueError('Fine-tune at least one step first.')
            def restore_live(self): raise FileNotFoundError('session.pt')
        self.service = TrainingService('.',factory=MissingStateLab)
        self.reject('transfer-train',{},409)
        self.reject('transfer-predict',{'text':'good'},409)
        self.reject('restore-live',{},404)
        self.assertEqual(self.post('train',{'steps':1})['step'],1)

    def test_live_busy_requests_do_not_initialize_or_mutate_engine(self):
        self.service.lock.acquire()
        try:
            for action, body in (('generate',{'text':'film'}),('transfer-start',{}),
                                 ('transfer-train',{}),('transfer-predict',{'text':'good'}),
                                 ('save-live',{}),('restore-live',{})):
                with self.subTest(action=action): self.reject(action,body,409)
            with self.assertRaises(TrainingApiError) as caught:
                self.service.dispatch('GET','/api/training/transfer-state')
            self.assertEqual(caught.exception.code,'training_busy')
        finally:
            self.service.lock.release()
        self.assertIsNone(self.service.lab)


if __name__ == '__main__': unittest.main()
