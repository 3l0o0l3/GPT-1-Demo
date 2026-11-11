"""Transfer objective parity and warmup contracts; CUDA compilation is separate."""
from contextlib import ExitStack, nullcontext
import copy
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from training_engine import GPT, sequence_loss
from training_transfer_execution import PreparedTransferObjective


class TransferExecutionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.model = GPT(19)

    def batch(self, length, size=8):
        inputs = torch.arange(size*length).reshape(size,length).remainder(19)
        ends = length-1-torch.arange(size).remainder(min(3,length-1))
        inputs[torch.arange(size),ends] = 3
        inputs.masked_fill_(torch.arange(length)[None]>ends[:,None],0)
        return inputs, ends, torch.arange(size).remainder(2)

    def snapshot(self):
        return {'weights':{k:v.detach().clone() for k,v in self.model.state_dict().items()},
                'grads':[(p.grad, None if p.grad is None else p.grad.clone()) for p in self.model.parameters()],
                'modes':[m.training for m in self.model.modules()],
                'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.random.get_rng_state().clone()}

    def assert_snapshot(self, before):
        self.assertEqual(before['modes'],[m.training for m in self.model.modules()])
        self.assertEqual(before['python'],random.getstate())
        self.assertTrue(np.array_equal(before['numpy'][1],np.random.get_state()[1]))
        self.assertEqual(before['numpy'][0],np.random.get_state()[0])
        self.assertEqual(before['numpy'][2:],np.random.get_state()[2:])
        self.assertTrue(torch.equal(before['torch'],torch.random.get_rng_state()))
        for name,value in self.model.state_dict().items():
            self.assertTrue(torch.equal(before['weights'][name],value))
        for parameter,(original,saved) in zip(self.model.parameters(),before['grads']):
            self.assertIs(parameter.grad,original)
            if saved is not None:self.assertTrue(torch.equal(parameter.grad,saved))

    def fake_compile_context(self, objective, compiler):
        """Exercise preparation bookkeeping on CPU; never claim CUDA acceptance."""
        objective._enabled = True
        objective._precision = 'fp16'
        objective._device = torch.device('cuda:0')
        real_arange = torch.arange
        stack = ExitStack()
        def cpu_arange(*args, **kwargs):
            kwargs['device'] = 'cpu'
            return real_arange(*args, **kwargs)
        stack.enter_context(patch('training_transfer_execution.torch.arange',side_effect=cpu_arange))
        stack.enter_context(patch('training_transfer_execution.torch.compile',side_effect=compiler))
        stack.enter_context(patch.object(objective,'_autocast',side_effect=lambda:nullcontext()))
        stack.enter_context(patch('training_transfer_execution.torch.cuda.get_rng_state',return_value=torch.tensor([1,2],dtype=torch.uint8)))
        stack.enter_context(patch('training_transfer_execution.torch.cuda.set_rng_state'))
        stack.enter_context(patch('training_transfer_execution.torch.cuda.synchronize'))
        return stack

    def test_eager_values_gradients_and_actual_padding_match_original(self):
        for weight in (0.,.5):
            objective=PreparedTransferObjective(self.model,sequence_loss,auxiliary_weight=weight,precision='fp32')
            for length,size in ((7,8),(23,8),(64,8),(13,3)):
                x,ends,labels=self.batch(length,size)
                original=x.clone()
                lm,hidden,_=self.model(x,lm_head=weight>0)
                logits=self.model.classifier(hidden[torch.arange(size),ends])
                reference=torch.nn.functional.cross_entropy(logits,labels)
                if weight:reference=reference+weight*sequence_loss(lm[:,:-1],x[:,1:])
                value=objective(x,ends,labels)
                self.assertTrue(torch.equal(reference,value))
                expected=torch.autograd.grad(reference,tuple(self.model.parameters()),allow_unused=True)
                actual=torch.autograd.grad(value,tuple(self.model.parameters()),allow_unused=True)
                for a,b in zip(expected,actual):
                    self.assertTrue((a is None and b is None) or torch.equal(a,b))
                self.assertTrue(torch.equal(original,x))
                self.assertEqual(value.ndim,0)

    def test_disabled_metadata_no_compile_and_immutable_binding(self):
        objective=PreparedTransferObjective(self.model,sequence_loss,precision='fp32')
        before=self.snapshot()
        with patch('training_transfer_execution.torch.compile',side_effect=AssertionError('Unexpected compilation')):
            self.assertEqual(objective.prepare(),{'backend':'eager','ready':False,'enabled':False,
                                                  'warmupSeconds':None,'dynamicLength':False})
        self.assert_snapshot(before)
        with self.assertRaises(AttributeError):objective.model=GPT(19)
        for options in ({'enabled':1},{'precision':'bf16'},{'auxiliary_weight':True},{'auxiliary_weight':.2},
                        {'enabled':True,'precision':'fp16'}):
            with self.subTest(options=options),self.assertRaises(ValueError):
                PreparedTransferObjective(self.model,sequence_loss,**options)

    def test_prepare_restores_rng_gradient_objects_modes_and_is_idempotent(self):
        self.model.blocks[0].eval()
        for index,p in enumerate(self.model.parameters()):
            if index%2:p.grad=torch.full_like(p,.125)
        objective=PreparedTransferObjective(self.model,sequence_loss,auxiliary_weight=.5,precision='fp32')
        calls=[]
        def compiler(module,**options):
            self.assertEqual(options,{'dynamic':True,'mode':'default'})
            random.random();np.random.random();torch.rand(1)
            def compiled(x,ends,labels):
                calls.append(tuple(x.shape))
                random.random();np.random.random();torch.rand(1)
                return module(x,ends,labels)
            return compiled
        before=self.snapshot()
        with self.fake_compile_context(objective,compiler):
            result=objective.prepare()
            self.assertEqual(result,objective.prepare())
        self.assert_snapshot(before)
        self.assertEqual(calls,[(8,40),(8,40),(8,41),(8,41),(8,64),(8,64)])
        self.assertTrue(result['ready'] and result['dynamicLength'])
        self.assertEqual(result['backend'],'compiled-dynamic')

    def test_failed_warmup_restores_state_and_rejects_unprepared_eligible_call(self):
        for p in self.model.parameters():p.grad=torch.full_like(p,.25)
        objective=PreparedTransferObjective(self.model,sequence_loss,precision='fp32')
        def compiler(module,**kwargs):
            def invalid(*args):
                random.random();np.random.random();torch.rand(1)
                return self.model.token.weight.sum()*float('nan')
            return invalid
        before=self.snapshot()
        with self.fake_compile_context(objective,compiler):
            with self.assertRaisesRegex(RuntimeError,'no optimizer update'):
                objective.prepare()
        self.assert_snapshot(before)
        self.assertFalse(objective.metadata()['ready'])
        fake=SimpleNamespace(device=torch.device('cuda:0'),shape=(8,20))
        with self.assertRaisesRegex(RuntimeError,'Prepare'):
            objective(fake,None,None)


if __name__ == '__main__':unittest.main()
