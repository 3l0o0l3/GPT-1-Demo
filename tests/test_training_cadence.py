"""Real updates, sparse metric provenance, persistence and hot-path result ownership."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import training_engine as engine
from test_training_engine import make_fixture
from test_training_amp import exact


class TrainingCadenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        data = self.root / 'data'; data.mkdir()
        make_fixture(data)
        self.lab = engine.TrainingLab(data, self.root / 'artifacts')

    def tearDown(self):
        self.temp.cleanup()

    def test_unaligned_lm_block_measures_exact_boundary_and_keeps_preview(self):
        lab = self.lab
        preview = lab.preview('train-0')
        lab.commit(preview['previewId'])
        self.assertEqual([row['step'] for row in lab.history], [0, 1])
        for _ in range(9): lab.train(10)
        lab.train(8)
        self.assertEqual((lab.step_count, lab.state()['lastMeasuredStep']), (99, 1))
        observed = []
        original = lab._measure
        def measure(timing=engine.NULL_SAMPLE):
            observed.append(lab.step_count)
            return original(timing=timing)
        with patch.object(lab, '_measure', side_effect=measure):
            state = lab.train(10)
        self.assertEqual(observed, [100])
        self.assertEqual((state['step'], state['lastMeasuredStep'], state['evaluationIntervalSteps']), (109, 100, 100))
        self.assertEqual([row['step'] for row in state['history']], [0, 1, 100])
        self.assertEqual(state['metrics']['valLoss'], state['history'][-1]['valLoss'])

    def test_transfer_and_restore_keep_actual_sample_step_without_checkpoint_fields(self):
        lab = self.lab
        lab.start_transfer(seed=19)
        for _ in range(10): lab.train_transfer(10)
        result = lab.train_transfer(7)
        self.assertEqual((result['step'], result['lastMeasuredStep']), (107, 100))
        for branch in result['branches']:
            self.assertEqual([row['step'] for row in branch['history']], [0, 100])
        before, transfer_before = lab._capture(), lab._capture_transfer()
        self.assertNotIn('lastMeasuredStep', before)
        self.assertNotIn('evaluationIntervalSteps', transfer_before)
        lab.save_live()
        lab.train_transfer(1)
        lab.restore_live()
        self.assertTrue(exact(before, lab._capture()))
        self.assertTrue(exact(transfer_before, lab._capture_transfer()))
        self.assertEqual(lab.transfer_state()['lastMeasuredStep'], 100)

    def test_paired_metric_publication_is_atomic_on_evaluation_failure(self):
        lab = self.lab
        lab.start_transfer()
        lab.train_transfer(3)
        before = {name: (copy.deepcopy(branch['metrics']), copy.deepcopy(branch['history']))
                  for name, branch in lab._transfer['branches'].items()}
        original = engine.evaluate_cached_classifier
        calls = 0
        def evaluate(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 4: raise RuntimeError('Second branch validation failed')
            return original(*args, **kwargs)
        with patch('training_engine.evaluate_cached_classifier', side_effect=evaluate):
            with self.assertRaisesRegex(RuntimeError, 'Second branch'):
                lab._measure_transfer(lab._transfer)
        after = {name: (branch['metrics'], branch['history']) for name, branch in lab._transfer['branches'].items()}
        self.assertEqual(before, after)

    def test_optional_metric_collection_preserves_update_and_skips_one_scalar_read(self):
        lab = self.lab
        x, y = lab._batch(lab.stories_train[:2])
        initial = lab._capture()
        original_float = torch.Tensor.__float__
        reads = []
        def observed(tensor):
            reads.append(tensor.numel())
            return original_float(tensor)
        with patch.object(torch.Tensor, '__float__', observed):
            reported = lab._update(x, y, .0003)
        count = len(reads)
        expected = lab._capture()
        lab._restore(initial); reads.clear()
        with patch.object(torch.Tensor, '__float__', observed):
            self.assertIsNone(lab._update(x, y, .0003, collect_metrics=False))
        self.assertIn('loss', reported)
        self.assertEqual(len(reads), count - 1)
        self.assertTrue(exact(expected, lab._capture()))

    def test_transfer_bindings_follow_pair_restore_and_survive_parent_reset(self):
        lab = self.lab
        lab.start_transfer(seed=7, auxiliary_weight=.5)
        bindings = lab._transfer_execution
        pair = lab._capture_transfer()
        lab.reset(seed=8)
        self.assertIs(lab._transfer_execution, bindings)
        self.assertTrue(exact(pair, lab._capture_transfer()))
        lab.save_live(); lab.restore_live()
        self.assertIsNot(lab._transfer_execution, bindings)
        self.assertTrue(exact(pair, lab._capture_transfer()))
        for name, binding in lab._transfer_execution.items():
            self.assertIs(binding.model, lab._transfer['branches'][name]['model'])
            self.assertIsNot(binding.model, bindings[name].model)
        self.assertNotIn('_transfer_execution', lab._capture_transfer())

    def test_failed_transfer_preparation_cannot_replace_live_pair(self):
        lab = self.lab
        lab.start_transfer(seed=7)
        before, pair = lab._capture(), lab._capture_transfer()
        bindings = lab._transfer_execution
        original = engine.PreparedTransferObjective.prepare
        calls = 0
        def prepare(binding):
            nonlocal calls
            calls += 1
            if calls == 2: raise RuntimeError('Candidate compilation failed')
            return original(binding)
        for action in ('start', 'restore'):
            calls = 0
            lab.save_live()
            with patch.object(engine.PreparedTransferObjective, 'prepare', prepare):
                with self.assertRaisesRegex(RuntimeError, 'Candidate compilation'):
                    lab.start_transfer(seed=19) if action == 'start' else lab.restore_live()
            self.assertIs(lab._transfer_execution, bindings)
            self.assertTrue(exact(before, lab._capture()))
            self.assertTrue(exact(pair, lab._capture_transfer()))


if __name__ == '__main__':
    unittest.main()
