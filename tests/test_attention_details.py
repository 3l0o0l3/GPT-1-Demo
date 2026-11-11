"""Real isolated CPU inspection vectors; no recorded or production checkpoints."""
import json
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from training_engine import TrainingLab, sequence_loss
from test_training_engine import make_fixture
from test_training_amp import exact


class AttentionDetailsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        data = root / 'data'; data.mkdir()
        make_fixture(data)
        self.lab = TrainingLab(data, root / 'artifacts')
        self.text = 'the good story is nice .'

    def tearDown(self):
        self.temp.cleanup()

    @torch.no_grad()
    def original_inspect(self, text, position):
        lab = self.lab
        lab.model.eval()
        ids, targets = lab._input(text)
        logits, _, attention = lab.model(ids, return_attention=True)
        probabilities = logits[0, position].softmax(-1)
        target_id = int(targets[0, position])
        indexes = probabilities.topk(min(8, lab.vocab_size)).indices.tolist()
        if target_id not in indexes: indexes.append(target_id)
        return {
            'text': text, 'tokens': [{'id': int(i), 'piece': lab._piece(i)} for i in ids[0]],
            'targets': [{'id': int(i), 'piece': lab._piece(i)} for i in targets[0]],
            'position': position, 'target': {'id': target_id, 'piece': lab._piece(target_id)},
            'probabilities': [{'id': i, 'piece': lab._piece(i), 'p': float(probabilities[i])} for i in indexes],
            'targetProbability': float(probabilities[target_id]),
            'tokenLoss': float(F.cross_entropy(logits[:, position], targets[:, position])),
            'batchLoss': float(sequence_loss(logits, targets)),
            'attention': attention[0, 0].cpu().tolist(),
            'mask': torch.ones(ids.shape[1], ids.shape[1], dtype=torch.bool).triu(1).tolist(),
            'maskConvention': 'true means blocked future position; false means visible',
            'tokenPieceConvention': 'Raw byte-level BPE pieces; Ġ represents a preceding space.',
        }

    def test_legacy_inspection_fields_are_exact_and_details_are_json_finite(self):
        expected = self.original_inspect(self.text, 3)
        result = self.lab.inspect(self.text, 3)
        details = result.pop('attentionDetails')
        self.assertEqual(result, expected)
        self.assertEqual(set(details), {'schemaVersion', 'layer', 'head', 'headDimension',
            'queryPosition', 'scale', 'query', 'keys', 'values', 'scores', 'weights', 'context'})
        self.assertEqual((details['schemaVersion'], details['layer'], details['head']), (1, 1, 1))
        self.assertEqual(details['headDimension'], 32)
        self.assertEqual(details['scale'], math.sqrt(32))
        json.dumps(details, allow_nan=False)

    def test_actual_qkv_projection_scores_causal_softmax_and_weighted_context(self):
        ids, _ = self.lab._input(self.text)
        block = self.lab.model.blocks[0]
        with torch.no_grad():
            embedded = self.lab.model.token(ids) + self.lab.model.position(torch.arange(ids.shape[1]))[None]
            projections = F.linear(embedded, block.qkv.weight, block.qkv.bias)
            expected = [part.reshape(1, ids.shape[1], 4, 32)[0, :, 0]
                        for part in projections.chunk(3, dim=-1)]
        for position in (0, 3, ids.shape[1] - 1):
            with self.subTest(position=position):
                result = self.lab.inspect(self.text, position)
                details = result['attentionDetails']
                q, k, v = (torch.tensor(details[key], dtype=torch.float64) for key in ('query', 'keys', 'values'))
                self.assertEqual(details['queryPosition'], position)
                self.assertEqual(tuple(k.shape), (ids.shape[1], 32))
                self.assertEqual(tuple(v.shape), (ids.shape[1], 32))
                self.assertTrue(torch.equal(q.float(), expected[0][position]))
                self.assertTrue(torch.equal(k.float(), expected[1]))
                self.assertTrue(torch.equal(v.float(), expected[2]))
                independent_scores = k @ q / math.sqrt(32)
                for index, score in enumerate(details['scores']):
                    if index > position:
                        self.assertIsNone(score)
                        independent_scores[index] = float('-inf')
                    else:
                        self.assertAlmostEqual(score, float(independent_scores[index]), places=6)
                weights = torch.tensor(details['weights'], dtype=torch.float64)
                self.assertEqual(details['weights'], result['attention'][position])
                self.assertTrue(all(value == 0 for value in details['weights'][position + 1:]))
                torch.testing.assert_close(weights, independent_scores.softmax(-1), atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(torch.tensor(details['context'], dtype=torch.float64), weights @ v,
                                           atol=1e-6, rtol=1e-5)

    def test_bounded_context_and_clamped_positions(self):
        text = ' '.join(['the'] * 160)
        for requested, expected in ((999, 127), (-10, 0)):
            result = self.lab.inspect(text, requested)
            details = result['attentionDetails']
            self.assertEqual(len(result['tokens']), 128)
            self.assertEqual(details['queryPosition'], expected)
            self.assertTrue(all(len(details[key]) == 128 for key in ('keys', 'values', 'scores', 'weights')))
            self.assertEqual(len(details['query']), 32)
            self.assertEqual(len(details['context']), 32)

    def test_future_tokens_do_not_change_selected_attention_or_context(self):
        before = self.lab.inspect(self.text, 3)['attentionDetails']
        after = self.lab.inspect('the good story very bad .', 3)['attentionDetails']
        for key in ('query', 'scores', 'weights', 'context'):
            self.assertEqual(before[key], after[key])
        for key in ('keys', 'values'):
            self.assertEqual(before[key][:4], after[key][:4])

    def test_inspection_preserves_model_optimizer_gradients_rng_and_external_hooks(self):
        lab = self.lab
        lab.train(1)  # Populate Adam state; retain actual parameter gradients.
        block = lab.model.blocks[0]
        calls = []
        external_qkv = block.qkv.register_forward_hook(lambda m, args, out: calls.append(('qkv', torch.is_grad_enabled())))
        external_context = block.proj.register_forward_pre_hook(lambda m, args: calls.append(('context', torch.is_grad_enabled())))
        try:
            before = lab._capture()
            grads = [None if p.grad is None else p.grad.clone() for p in lab.model.parameters()]
            py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone()
            qkv_hooks, context_hooks = dict(block.qkv._forward_hooks), dict(block.proj._forward_pre_hooks)
            lab.inspect(self.text, 3)
            self.assertTrue(exact(before, lab._capture()))
            self.assertTrue(exact(grads, [p.grad for p in lab.model.parameters()]))
            self.assertEqual(random.getstate(), py_state)
            current_np = np.random.get_state()
            self.assertEqual(current_np[0], np_state[0])
            self.assertTrue(np.array_equal(current_np[1], np_state[1]))
            self.assertEqual(current_np[2:], np_state[2:])
            self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_state))
            self.assertEqual(block.qkv._forward_hooks, qkv_hooks)
            self.assertEqual(block.proj._forward_pre_hooks, context_hooks)
            self.assertEqual(calls, [('qkv', False), ('context', False)])
        finally:
            external_qkv.remove(); external_context.remove()

    def test_hooks_removed_when_forward_or_registration_fails(self):
        block = self.lab.model.blocks[0]
        cases = ((block.qkv, 'forward'), (block.proj, 'forward'), (block.proj, 'register_forward_pre_hook'))
        for module, method in cases:
            with self.subTest(method=method, module=module), patch.object(module, method, side_effect=RuntimeError('Forced failure')):
                with self.assertRaisesRegex(RuntimeError, 'Forced failure'):
                    self.lab.inspect(self.text, 3)
            self.assertEqual(len(block.qkv._forward_hooks), 0)
            self.assertEqual(len(block.proj._forward_pre_hooks), 0)
        self.assertIn('attentionDetails', self.lab.inspect(self.text, 3))


if __name__ == '__main__':
    unittest.main()
