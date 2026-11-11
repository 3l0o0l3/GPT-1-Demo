"""Timing observes original training and cannot become model/checkpoint state."""
from pathlib import Path
import copy
import json
import tempfile
import unittest
from unittest.mock import patch

from training_engine import TrainingLab
from training_timing import TrainingTiming
from test_training_engine import make_fixture
from test_training_amp import exact


class TimingIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        data = root / 'data'; data.mkdir()
        make_fixture(data)
        self.lab = TrainingLab(data, root / 'artifacts')

    def tearDown(self):
        self.temp.cleanup()

    def test_lm_timing_does_not_change_updates_optimizer_or_rng(self):
        lab = self.lab
        initial = lab._capture()
        lab._timing = TrainingTiming(lab.device, enabled=False)
        for _ in range(2): lab.train(10)
        expected = lab._capture()
        lab._restore(initial)
        lab._timing = TrainingTiming(lab.device)
        for _ in range(2): lab.train(10)
        self.assertTrue(exact(expected, lab._capture()))
        timing = lab.state()['timing']
        self.assertEqual([p['step'] for p in timing['steps']['trend']], [1, 20])
        self.assertEqual(timing['counters']['observedSteps'], 20)
        self.assertEqual(set(timing['steps']['phases']),
                         {'data', 'setup', 'forward_loss', 'loss_check',
                          'backward', 'gradient_check_clip', 'optimizer'})
        self.assertEqual(timing['steps']['total']['gpuMs']['count'], 0)
        self.assertNotIn('timing', lab._capture())
        published = lab.state()['timing']
        self.assertIs(copy.deepcopy(published), published)
        with self.assertRaises(TypeError):
            published['steps']['trend'][0]['cpuMs'] = -1
        self.assertEqual(json.loads(json.dumps(published)), lab._timing.snapshot('lm'))
        saved = lab._timing.snapshot('lm')
        saved['steps']['trend'].clear()
        self.assertEqual(len(lab.state()['timing']['steps']['trend']), 2)

    def test_transfer_timing_preserves_both_branches_and_parent(self):
        lab = self.lab
        lab.start_transfer(seed=7, auxiliary_weight=.5)
        lab.save_live()
        parent = lab._capture()
        lab._timing = TrainingTiming(lab.device, enabled=False)
        for _ in range(2): lab.train_transfer(10)
        expected = lab._capture_transfer()
        lab.restore_live()
        lab._timing = TrainingTiming(lab.device)
        for _ in range(2): lab.train_transfer(10)
        self.assertTrue(exact(parent, lab._capture()))
        self.assertTrue(exact(expected, lab._capture_transfer()))
        timing = lab.transfer_state()['timing']
        for branch in ('pretrained', 'scratch'):
            for phase in ('setup', 'forward_loss', 'loss_check', 'backward',
                          'gradient_check_clip', 'optimizer'):
                self.assertIn(branch + '.' + phase, timing['steps']['phases'])
        self.assertNotIn('timing', lab._capture_transfer())

    def test_evaluations_are_separate_and_reset_follows_model_ownership(self):
        lab = self.lab
        for _ in range(10): lab.train(10)
        lm = lab.state()['timing']
        self.assertEqual([r['step'] for r in lm['steps']['trend']], [1, 20, 40, 60, 80, 100])
        self.assertEqual([r['step'] for r in lm['evaluations']['trend']], [100])
        self.assertEqual(set(lm['evaluations']['phases']), {'train_evaluation', 'validation'})
        lab.start_transfer(seed=7)
        for _ in range(10): lab.train_transfer(10)
        transfer = lab.transfer_state()['timing']
        self.assertEqual([r['step'] for r in transfer['evaluations']['trend']], [100])
        self.assertEqual(set(transfer['evaluations']['phases']),
                         {b + '.' + p for b in ('pretrained', 'scratch')
                          for p in ('train_evaluation', 'validation')})
        lab.reset(seed=8)
        self.assertEqual(lab.state()['timing']['counters']['observedSteps'], 0)
        self.assertEqual(lab.transfer_state()['timing'], transfer)
        lab.train(1); lab.save_live()
        model, pair = lab._capture(), lab._capture_transfer()
        lab.restore_live()
        self.assertTrue(exact(model, lab._capture()))
        self.assertTrue(exact(pair, lab._capture_transfer()))
        self.assertEqual(lab.state()['timing']['steps']['count'], 0)
        self.assertEqual(lab.transfer_state()['timing']['steps']['count'], 0)

    def test_failed_update_stays_failed_and_preview_does_not_create_timing(self):
        lab = self.lab
        error = RuntimeError('original training failure')
        with patch.object(lab, '_update', side_effect=error):
            with self.assertRaises(RuntimeError) as caught:
                lab.train(1)
        self.assertIs(caught.exception, error)
        self.assertEqual(lab.step_count, 0)
        timing = lab.state()['timing']
        self.assertEqual(timing['steps']['failedCount'], 1)
        self.assertEqual(timing['steps']['total']['cpuMs']['count'], 0)
        self.assertFalse(timing['steps']['trend'][0]['success'])
        preview = lab.preview('train-0')
        self.assertEqual(lab.state()['timing'], timing)
        lab.commit(preview['previewId'])
        self.assertEqual(lab.state()['timing'], timing)


if __name__ == '__main__':
    unittest.main()
