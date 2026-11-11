"""AMP accounting and checkpoint contracts using real CPU GradScaler, fake autocast."""
import copy
from contextlib import contextmanager, nullcontext
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import training_engine as engine
from test_training_engine import make_fixture


@contextmanager
def cpu_amp(lab):
    real_scaler = torch.amp.GradScaler
    with patch('training_engine.torch.amp.GradScaler', side_effect=lambda *a, **k: real_scaler('cpu')), \
         patch('training_engine.torch.autocast', side_effect=lambda *a, **k: nullcontext()):
        lab.training_precision = 'fp16'
        lab._grad_scaler, lab._dormant_grad_scaler = engine._new_grad_scaler('fp16')
        yield


def exact(left, right):
    if isinstance(left, torch.Tensor): return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict): return left.keys() == right.keys() and all(exact(value, right[key]) for key, value in left.items())
    if isinstance(left, (list, tuple)): return type(left) == type(right) and len(left) == len(right) and all(exact(a, b) for a, b in zip(left, right))
    return left == right


class AMPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / 'data'; self.data.mkdir()
        make_fixture(self.data)
        self.lab = engine.TrainingLab(self.data, self.root / 'artifacts')

    def tearDown(self): self.temp.cleanup()

    def test_explicit_cuda_and_metadata(self):
        with self.assertRaisesRegex(ValueError, 'requires CUDA'):
            engine.TrainingLab(self.data, self.root / 'other', training_precision='fp16')
        with cpu_amp(self.lab):
            value = self.lab.metadata()['execution']
            self.assertEqual(value['trainingPrecision'], 'FP16 AMP')
            self.assertEqual(value['transferPrecision'], 'FP16 AMP')
            self.assertEqual(value['previewPrecision'], 'FP32')
            self.assertEqual(value['optimizerStatePrecision'], 'FP32')
            self.assertEqual(value['maxAttemptsPerUpdate'], 8)

    def test_lm_overflow_reuses_batch_rng_and_counts_only_actual_step(self):
        lab = self.lab
        with cpu_amp(lab):
            attempts = []; original = lab.model.forward
            def forward(x, *args, **kwargs):
                attempts.append((x.clone(), torch.rand(1).item()))
                return original(x, *args, **kwargs)
            calls = 0
            def overflow_once(gradient):
                nonlocal calls
                calls += 1
                return torch.full_like(gradient, float('inf')) if calls == 1 else gradient
            hook = next(lab.model.parameters()).register_hook(overflow_once)
            rng = torch.random.get_rng_state().clone()
            with patch.object(lab.model, 'forward', forward):
                x, y = lab._input('the good story')
                result = lab._update(x, y, .0003)
            hook.remove()
            self.assertEqual(result['overflowRetries'], 1)
            self.assertEqual(lab.step_count, 1)
            self.assertEqual(len(attempts), 2)
            self.assertTrue(torch.equal(attempts[0][0], attempts[1][0]))
            self.assertEqual(attempts[0][1], attempts[1][1])
            final_rng = torch.random.get_rng_state()
            torch.random.set_rng_state(rng); torch.rand(1)
            self.assertTrue(torch.equal(torch.random.get_rng_state(), final_rng))
            self.assertEqual(lab._grad_scaler.get_scale(), 32768.)
            self.assertEqual({int(s['step']) for s in lab.optimizer.state.values()}, {1})
            self.assertFalse(lab.optimizer._optimizer_step_post_hooks)

    def test_exhausted_overflow_leaves_weights_unchanged_and_cleans_hooks_gradients(self):
        lab = self.lab
        with cpu_amp(lab):
            before = lab._capture()
            hook = next(lab.model.parameters()).register_hook(lambda g: torch.full_like(g, float('inf')))
            with self.assertRaisesRegex(FloatingPointError, '8 attempts'):
                lab.train(1)
            hook.remove()
            self.assertEqual(lab.step_count, 0)
            self.assertTrue(exact(before['model'], lab._capture()['model']))
            self.assertFalse(lab.optimizer.state)
            self.assertTrue(all(p.grad is None for p in lab.model.parameters()))
            self.assertFalse(lab.optimizer._optimizer_step_post_hooks)
            self.assertEqual(lab._grad_scaler.get_scale(), 256.)

    def test_failure_after_real_step_is_counted_and_never_replayed(self):
        lab = self.lab
        with cpu_amp(lab), patch.object(lab._grad_scaler, 'update', side_effect=RuntimeError('after step failed')):
            with self.assertRaisesRegex(RuntimeError, 'after step failed'):
                lab.train(1)
            self.assertEqual(lab.step_count, 1)
            self.assertEqual({int(s['step']) for s in lab.optimizer.state.values()}, {1})
            self.assertFalse(lab.optimizer._optimizer_step_post_hooks)
            self.assertTrue(all(p.grad is None for p in lab.model.parameters()))

    def test_single_transfer_branch_overflow_preserves_pair_atomicity_and_growth(self):
        lab = self.lab
        with cpu_amp(lab):
            lab.start_transfer(seed=7, auxiliary_weight=.5)
            pair = lab._transfer
            healthy, bad = pair['branches']['pretrained'], pair['branches']['scratch']
            for branch, tracker in ((healthy, 57), (bad, 13)):
                state = branch['gradScaler'].state_dict(); state['_growth_tracker'] = tracker
                branch['gradScaler'].load_state_dict(state)
            parent = lab._capture()
            calls = 0
            def overflow_once(gradient):
                nonlocal calls
                calls += 1
                # No healthy optimizer can run before the bad branch validates.
                if calls == 1:
                    self.assertFalse(healthy['optimizer'].state)
                    return torch.full_like(gradient, float('inf'))
                return gradient
            hook = next(bad['model'].parameters()).register_hook(overflow_once)
            lab.train_transfer(1)
            hook.remove()
            self.assertEqual(pair['step'], 1)
            self.assertEqual(calls, 2)
            self.assertEqual(healthy['gradScaler'].state_dict()['_growth_tracker'], 58)
            self.assertEqual(bad['gradScaler'].state_dict()['_growth_tracker'], 1)
            self.assertEqual(healthy['gradScaler'].get_scale(), 65536.)
            self.assertEqual(bad['gradScaler'].get_scale(), 32768.)
            for branch in pair['branches'].values():
                self.assertEqual({int(s['step']) for s in branch['optimizer'].state.values()}, {1})
                self.assertFalse(branch['optimizer']._optimizer_step_post_hooks)
            self.assertTrue(exact(parent, lab._capture()))

    def test_transfer_overflow_exhaustion_updates_neither_branch(self):
        lab = self.lab
        with cpu_amp(lab):
            lab.start_transfer(seed=7)
            before = lab._capture_transfer()
            branch = lab._transfer['branches']['scratch']
            hook = next(branch['model'].parameters()).register_hook(lambda g: torch.full_like(g, float('inf')))
            with self.assertRaisesRegex(FloatingPointError, 'neither branch was updated'):
                lab.train_transfer(1)
            hook.remove()
            self.assertEqual(lab._transfer['step'], 0)
            for name, current in lab._transfer['branches'].items():
                self.assertTrue(exact(before['branches'][name]['model'], current['model'].state_dict()))
                self.assertFalse(current['optimizer'].state)
                self.assertFalse(current['optimizer']._optimizer_step_post_hooks)
                self.assertTrue(all(p.grad is None for p in current['model'].parameters()))

    def test_scaler_roundtrip_preview_fp32_and_legacy_defaults(self):
        lab = self.lab
        with cpu_amp(lab):
            lab.train(2); lab.start_transfer(seed=7); lab.train_transfer(1)
            before = lab._capture()
            with patch('training_engine.torch.autocast', side_effect=AssertionError('preview must remain FP32')):
                preview = lab.preview('train-0')
            self.assertTrue(exact(before, lab._capture()))
            lab.commit(preview['previewId'])
            self.assertTrue(exact(before['gradScaler'], lab._capture()['gradScaler']))
            lab.save_live()
            saved = torch.load(lab.artifact_dir / 'live' / 'session.pt', weights_only=False)
            restored = engine.TrainingLab(self.data, lab.artifact_dir)
            restored.restore_live() # FP32 reader preserves dormant scaler snapshots.
            self.assertTrue(exact(saved['lm'], restored._capture()))
            self.assertTrue(exact(saved['transfer'], restored._capture_transfer()))
            saved['lm'].pop('gradScaler')
            for branch in saved['transfer']['branches'].values(): branch.pop('gradScaler')
            torch.save(saved, lab.artifact_dir / 'live' / 'session.pt')
            lab.restore_live()
            self.assertEqual(lab._grad_scaler.state_dict()['_growth_tracker'], 0)
            self.assertEqual(lab._grad_scaler.get_scale(), 65536.)
            for branch in lab._transfer['branches'].values():
                self.assertEqual(branch['gradScaler'].state_dict()['_growth_tracker'], 0)
                self.assertEqual(branch['gradScaler'].get_scale(), 65536.)

    def test_transfer_optimizer_exception_is_explicit_and_blocks_pair_replay(self):
        lab = self.lab
        with cpu_amp(lab):
            lab.start_transfer(seed=7)
            pair = lab._transfer
            second = pair['branches']['scratch']['optimizer']
            with patch.object(second, 'step', side_effect=RuntimeError('optimizer failed')):
                with self.assertRaisesRegex(RuntimeError, 'Restart transfer'):
                    lab.train_transfer(1)
            self.assertEqual(pair['step'], 0)
            self.assertTrue(lab.transfer_state()['requiresRestart'])
            self.assertEqual({int(s['step']) for s in pair['branches']['pretrained']['optimizer'].state.values()}, {1})
            self.assertFalse(second.state)
            with self.assertRaisesRegex(ValueError, 'Restart transfer'):
                lab.train_transfer(1)
            for branch in pair['branches'].values():
                self.assertFalse(branch['optimizer']._optimizer_step_post_hooks)
            lab.save_live(); lab.restore_live()
            with self.assertRaisesRegex(ValueError, 'Restart transfer'):
                lab.train_transfer(1)


if __name__ == '__main__': unittest.main()
