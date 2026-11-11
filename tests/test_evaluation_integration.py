"""Fixed-cache integration parity and persistence; isolated CPU fixture only."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import training_engine as engine
from test_training_engine import make_fixture
from test_training_amp import exact


class EvaluationIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        data = self.root / 'data'; data.mkdir()
        make_fixture(data)
        self.lab = engine.TrainingLab(data, self.root / 'artifacts')

    def tearDown(self):
        self.temp.cleanup()

    def test_cached_metrics_equal_original_loops_before_and_after_updates(self):
        lab = self.lab
        for steps in (0, 3):
            if steps:
                lab.train(steps)
                lab._measure()
            self.assertEqual(lab.metrics['trainLoss'], lab._evaluate_lm(lab._eval_train))
            self.assertEqual(lab.metrics['valLoss'], lab._evaluate_lm(lab._eval_val))
        lab.start_transfer(seed=19)
        lab.train_transfer(2)
        lab._measure_transfer(lab._transfer)
        for branch in lab._transfer['branches'].values():
            train = engine.evaluate_classifier(branch['model'], lab.sentiment_train, lab.device)
            val = engine.evaluate_classifier(branch['model'], lab.sentiment_val, lab.device)
            self.assertEqual({key: branch['metrics'][key] for key in ('trainLoss', 'valLoss', 'accuracy')},
                             {'trainLoss': train['loss'], 'valLoss': val['loss'], 'accuracy': val['accuracy']})

    def test_hot_measurement_never_repacks_and_keeps_training_state(self):
        lab = self.lab
        lab.start_transfer(seed=17, auxiliary_weight=.5)
        lab.train_transfer(1)  # Include populated Adam state in the preservation check.
        lab._measure_transfer(lab._transfer)
        before, pair_before = lab._capture(), lab._capture_transfer()
        with (patch.object(lab, '_batch', side_effect=AssertionError('Hot evaluation repacked LM')),
              patch('training_engine.sentiment_batch', side_effect=AssertionError('Hot evaluation repacked classifier'))):
            lab._measure()
            lab._measure_transfer(lab._transfer)
        self.assertTrue(exact(before, lab._capture()))
        after = lab._capture_transfer()
        for name in ('pretrained', 'scratch'):
            after['branches'][name]['history'] = pair_before['branches'][name]['history']
        self.assertTrue(exact(pair_before, after))

    def test_cache_stays_runtime_only_across_preview_reset_and_restore(self):
        lab = self.lab
        caches = (lab._lm_eval_batches, lab._sentiment_eval_batches)
        pointers = [batch.inputs.data_ptr() for group in caches for batches in group.values() for batch in batches]
        before = lab._capture()
        lab.preview('train-0')
        self.assertTrue(exact(before, lab._capture()))
        lab.train(2); lab.start_transfer(seed=13); lab.train_transfer(1)
        lab.save_live()
        saved = torch.load(lab.artifact_dir / 'live' / 'session.pt', weights_only=False)
        self.assertNotIn('_lm_eval_batches', saved)
        self.assertNotIn('_sentiment_eval_batches', saved)
        before, pair_before = lab._capture(), lab._capture_transfer()
        lab.reset(seed=7)
        lab.restore_live()
        self.assertTrue(exact(before, lab._capture()))
        self.assertTrue(exact(pair_before, lab._capture_transfer()))
        self.assertIs(caches[0], lab._lm_eval_batches)
        self.assertIs(caches[1], lab._sentiment_eval_batches)
        self.assertEqual(pointers, [batch.inputs.data_ptr() for group in caches for batches in group.values() for batch in batches])

    def test_identical_training_trajectory_against_original_evaluation_paths(self):
        lab = self.lab
        initial = lab._capture()
        for _ in range(10):
            lab.train(10)
        candidate = lab._capture()
        lab._restore(initial)
        def original_lm(model, batches, loss_fn):
            rows = lab._eval_train if batches is lab._lm_eval_batches['train'] else lab._eval_val
            return lab._evaluate_lm(rows)
        with patch('training_engine.evaluate_cached_lm', side_effect=original_lm):
            for _ in range(10):
                lab.train(10)
        self.assertTrue(exact(candidate, lab._capture()))
        lab.start_transfer(seed=31, auxiliary_weight=.5)
        pair = lab._capture_transfer()
        for _ in range(10):
            lab.train_transfer(10)
        candidate_pair = lab._capture_transfer()
        lab._transfer = lab._load_transfer(pair)
        def original_classifier(model, batches, **kwargs):
            rows = (lab._fixed_subset(lab.sentiment_train, 128)
                    if batches is lab._sentiment_eval_batches['train'] else lab.sentiment_val)
            result = engine.evaluate_classifier(model, rows, lab.device)
            if kwargs:
                weight = kwargs['auxiliary_weight']
                lm_total = 0.
                if weight:
                    with torch.no_grad():
                        for offset in range(0, len(rows), 16):
                            part = rows[offset:offset + 16]
                            inputs, _, _ = engine.sentiment_batch(part, lab.device)
                            logits, _, _ = model(inputs)
                            lm_total += float(engine.sequence_loss(logits[:, :-1], inputs[:, 1:])) * len(part)
                lm_loss = lm_total / len(rows) if weight else None
                contribution = weight * lm_loss if weight else 0.
                result['objective'] = {'classificationLoss': result['loss'], 'lmLoss': lm_loss,
                    'auxiliaryWeight': weight, 'auxiliaryContribution': contribution,
                    'totalLoss': result['loss'] + contribution, 'sampleCount': len(rows)}
            return result
        with patch('training_engine.evaluate_cached_classifier', side_effect=original_classifier):
            for _ in range(10):
                lab.train_transfer(10)
        self.assertTrue(exact(candidate_pair, lab._capture_transfer()))

    def test_objective_snapshot_schema_and_independent_history_copies(self):
        state = self.lab.start_transfer(seed=7, auxiliary_weight=.5)
        for branch in state['branches']:
            objective = branch['metrics']['objective']
            self.assertEqual(set(objective), {'schemaVersion', 'step', 'classificationLoss', 'lmLoss',
                'auxiliaryWeight', 'auxiliaryContribution', 'totalLoss', 'sampleCount', 'split', 'measurement'})
            self.assertEqual((objective['schemaVersion'], objective['step'], objective['sampleCount']), (1, 0, 8))
            self.assertEqual((objective['split'], objective['measurement']), ('train', 'fixed-subset-eval'))
            weight = .5 if branch['name'] == 'pretrained' else 0.
            self.assertEqual(objective['auxiliaryWeight'], weight)
            self.assertEqual(objective['classificationLoss'], branch['metrics']['trainLoss'])
            self.assertEqual(objective['auxiliaryContribution'], weight * objective['lmLoss'] if weight else 0.)
            self.assertEqual(objective['totalLoss'], objective['classificationLoss'] + objective['auxiliaryContribution'])
            self.assertEqual(branch['history'][-1]['objective'], objective)
            if not weight: self.assertIsNone(objective['lmLoss'])
        branch = self.lab._transfer['branches']['pretrained']
        self.assertIsNot(branch['metrics']['objective'], branch['history'][-1]['objective'])
        state['branches'][0]['metrics']['objective']['totalLoss'] = -100
        self.assertGreater(branch['metrics']['objective']['totalLoss'], 0.)

    def test_legacy_checkpoint_without_objectives_restores_without_remeasurement(self):
        lab = self.lab
        lab.start_transfer(seed=23, auxiliary_weight=.5)
        lab.train_transfer(1)
        lab.save_live()
        path = lab.artifact_dir / 'live' / 'session.pt'
        saved = torch.load(path, weights_only=False)
        for branch in saved['transfer']['branches'].values():
            branch['metrics'].pop('objective')
            for point in branch['history']: point.pop('objective')
        torch.save(saved, path)
        with patch.object(engine.TrainingLab, '_measure_transfer', side_effect=AssertionError('Restore must not remeasure')):
            lab.restore_live()
        self.assertTrue(exact(saved['transfer'], lab._capture_transfer()))
        state = lab.transfer_state()
        self.assertTrue(all('objective' not in branch['metrics'] for branch in state['branches']))
        lab._measure_transfer(lab._transfer)
        for branch in lab.transfer_state()['branches']:
            self.assertNotIn('objective', branch['history'][0])
            self.assertEqual(branch['history'][-1]['objective']['step'], 1)

    def test_nonfinite_auxiliary_measurement_does_not_publish_partial_pair(self):
        lab = self.lab
        lab.start_transfer(seed=23, auxiliary_weight=.5)
        before = lab._capture_transfer()
        def nonfinite(logits, targets):
            return logits.new_tensor(float('nan'))
        with patch('training_engine.sequence_loss', side_effect=nonfinite):
            with self.assertRaisesRegex(FloatingPointError, 'non-finite objective'):
                lab._measure_transfer(lab._transfer)
        self.assertTrue(exact(before, lab._capture_transfer()))


if __name__ == '__main__':
    unittest.main()
