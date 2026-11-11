"""Execution-cache contracts with a controlled compiler; real CUDA is separate."""
import copy
from contextlib import contextmanager
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from training_engine import TrainingLab
from test_training_engine import make_fixture


@contextmanager
def compiler_fixture(lab, compiler):
    """Exercise preparation on CPU without claiming CUDA/compiler acceptance."""
    original_device, original_arange = lab.device, torch.arange
    lab.device = torch.device('cuda:0')
    def cpu_arange(*args, **kwargs):
        return original_arange(*args, **dict(kwargs, device='cpu'))
    try:
        with patch('training_engine.torch.compile', compiler), \
             patch('training_engine.torch.arange', cpu_arange), \
             patch('training_engine._isolated_rng', lambda _: torch.random.fork_rng(devices=[])), \
             patch('training_engine.torch.cuda.synchronize'), \
             patch('training_engine.torch.compiler.cudagraph_mark_step_begin'):
            yield
    finally:
        lab.device = original_device


class TrainingCompileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / 'data'
        self.data.mkdir()
        make_fixture(self.data)
        self.lab = TrainingLab(self.data, self.root / 'artifacts')

    def tearDown(self):
        self.temp.cleanup()

    def assert_exact(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, np.ndarray):
            self.assertTrue(np.array_equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left: self.assert_exact(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(type(left), type(right))
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right): self.assert_exact(a, b)
        else:
            self.assertEqual(left, right)

    def test_default_eager_metadata_and_explicit_cuda_requirement(self):
        lab = self.lab
        with patch('training_engine.torch.compile') as compile:
            execution = lab.prepare_training_execution()
            self.assertEqual(execution, lab.metadata()['execution'])
            self.assertFalse(execution['compileEnabled'])
            self.assertFalse(execution['compileReady'])
            self.assertEqual(execution['lmBackend'], 'eager')
            self.assertEqual(execution['compiledShape'], [8, 64])
            lab.compile_training = True
            with self.assertRaisesRegex(RuntimeError, 'requires CUDA'):
                lab.prepare_training_execution()
            compile.assert_not_called()

    def test_warmup_is_idempotent_and_preserves_full_state_gradients_modes_rng(self):
        lab = self.lab
        lab.train(2)
        lab.model.eval()
        lab.model.blocks[0].train()
        parameters = list(lab.model.parameters())
        parameters[0].grad = torch.full_like(parameters[0], .125)
        original_grads = [parameter.grad for parameter in parameters]
        grad_values = [None if gradient is None else gradient.clone() for gradient in original_grads]
        modes = [module.training for module in lab.model.modules()]
        before = lab._capture()
        global_rng = (random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone())
        lab.compile_training = True
        def compile(model, **kwargs):
            self.assertEqual(kwargs, {'mode': 'reduce-overhead', 'dynamic': False})
            def call(x):
                random.random(); np.random.random(); torch.rand(1)
                return model(x)
            return call
        compiler = Mock(side_effect=compile)
        with compiler_fixture(lab, compiler), \
             patch.object(lab.optimizer, 'step', side_effect=AssertionError('warmup must not optimize')), \
             patch.object(lab, '_update', side_effect=AssertionError('warmup must not update')):
            first = lab.prepare_training_execution()
            second = lab.prepare_training_execution()
            self.assertEqual(first, second)
            self.assertTrue(first['compileReady'])
            self.assertGreaterEqual(first['warmupSeconds'], 0)
        self.assertEqual(compiler.call_count, 1)
        self.assert_exact(before, lab._capture())
        self.assert_exact(global_rng, (random.getstate(), np.random.get_state(), torch.random.get_rng_state()))
        self.assertEqual(modes, [module.training for module in lab.model.modules()])
        for parameter, original, value in zip(parameters, original_grads, grad_values):
            self.assertIs(parameter.grad, original)
            self.assert_exact(parameter.grad, value)

    def test_compile_failure_preserves_state_and_does_not_publish_callable(self):
        lab = self.lab
        lab.train(1)
        before = lab._capture()
        modes = [module.training for module in lab.model.modules()]
        gradients = [parameter.grad for parameter in lab.model.parameters()]
        lab.compile_training = True
        calls = 0
        def bad_call(x):
            nonlocal calls
            calls += 1
            if calls == 2: raise RuntimeError('compiler fixture failed')
            return lab.model(x)
        with compiler_fixture(lab, Mock(return_value=bad_call)):
            with self.assertRaisesRegex(RuntimeError, 'Training compilation failed'):
                lab.prepare_training_execution()
        self.assertFalse(lab.metadata()['execution']['compileReady'])
        self.assert_exact(before, lab._capture())
        self.assertEqual(modes, [module.training for module in lab.model.modules()])
        for parameter, gradient in zip(lab.model.parameters(), gradients):
            self.assertIs(parameter.grad, gradient)

    def test_only_eligible_non_trace_updates_call_compiled_model(self):
        lab = self.lab
        lab.compile_training = True
        compiled = Mock(side_effect=lab.model)
        lab._compiled_training_model = lab.model
        lab._compiled_training_call = compiled
        with compiler_fixture(lab, Mock(side_effect=AssertionError('already prepared'))):
            x = torch.full((8, 64), 5, dtype=torch.long)
            lab._update(x, x, .0003)
            self.assertEqual(compiled.call_count, 1)
            lab._update(x[:1, :6], x[:1, :6], .0003)
            lab._update(x, x, .0003, trace=True)
            self.assertEqual(compiled.call_count, 1)
            before_step = lab.step_count
            compiled.side_effect = RuntimeError('execution failed')
            with self.assertRaisesRegex(RuntimeError, 'execution failed'), patch.object(lab.optimizer, 'step') as step:
                lab._update(x, x, .0003)
            step.assert_not_called()
            self.assertEqual(lab.step_count, before_step)

    def test_inplace_restore_keeps_cache_reset_invalidates_and_restore_is_transactional(self):
        lab = self.lab
        lab.train(1)
        lab.save_live()
        lab.compile_training = True
        model = lab.model
        call = Mock(side_effect=model)
        lab._compiled_training_model, lab._compiled_training_call = model, call
        state = lab._capture()
        lab._restore(copy.deepcopy(state))
        self.assertIs(lab._compiled_training_call, call)
        with patch.object(TrainingLab, 'prepare_training_execution', side_effect=RuntimeError('compile failed')):
            with self.assertRaisesRegex(RuntimeError, 'compile failed'):
                lab.restore_live()
        self.assertIs(lab.model, model)
        self.assertIs(lab._compiled_training_call, call)
        self.assert_exact(state, lab._capture())
        with patch.object(lab, 'prepare_training_execution') as prepare:
            lab.reset(seed=7)
            prepare.assert_called_once()
        self.assertIsNot(lab.model, model)
        self.assertIsNone(lab._compiled_training_call)
        self.assertFalse(lab.metadata()['execution']['compileReady'])


if __name__ == '__main__':
    unittest.main()
