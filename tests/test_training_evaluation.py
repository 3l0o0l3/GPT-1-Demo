"""Independent CPU parity tests for fixed evaluation inputs and scalar transfers."""
from dataclasses import FrozenInstanceError
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from training_evaluation import (
    prepare_lm, prepare_classifier, evaluate_lm, evaluate_classifier,
)


class ClassifierHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[.5, .2, .1], [-.5, -.3, -.1]]))

    def forward(self, hidden):
        return F.linear(hidden, self.weight)


class Model(nn.Module):
    """Deterministic model with real trainable parameters and ragged inputs."""
    def __init__(self):
        super().__init__()
        self.table = nn.Parameter(torch.tensor([
            [0., .1, .3, -.2, .5, -.4], [.2, -.5, 1.2, .3, -.1, .7],
            [.7, .8, -.1, -.7, .2, .5], [-.3, .4, .7, .1, .9, -.1],
            [.1, -.2, -.3, 1.3, .4, .6], [.4, .7, .2, .8, -.2, .5],
        ]))
        self.classifier = ClassifierHead()
        self.dropout = nn.Dropout(.7)
        self.calls = []
        self.lm_head_calls = []

    def forward(self, ids, lm_head=True):
        self.calls.append((tuple(ids.shape), self.training, torch.is_grad_enabled()))
        self.lm_head_calls.append(lm_head)
        logits = self.dropout(self.table[ids]) if lm_head else None
        hidden = torch.stack((ids.float() - 3, ids.float() * .05, torch.ones_like(ids).float()), -1)
        return logits, self.dropout(hidden), None


def pack_lm(rows):
    width = max(len(row) for row in rows)
    inputs = torch.zeros((len(rows), width), dtype=torch.long)
    targets = torch.zeros_like(inputs)
    for index, row in enumerate(rows):
        inputs[index, :len(row)] = torch.tensor(row)
        targets[index, :len(row)] = torch.tensor([value % 5 + 1 for value in row])
    return inputs, targets


def pack_classifier(rows):
    inputs, _ = pack_lm([row['ids'] for row in rows])
    return (inputs, torch.tensor([len(row['ids']) - 1 for row in rows]),
            torch.tensor([row['label'] for row in rows]))


def sequence_loss(logits, targets):
    losses = F.cross_entropy(logits.transpose(1, 2), targets, ignore_index=0, reduction='none')
    valid = targets.ne(0)
    return ((losses * valid).sum(1) / valid.sum(1).clamp_min(1)).mean()


@torch.no_grad()
def original_lm(model, rows, size=8):
    model.eval()
    total = 0.0
    for offset in range(0, len(rows), size):
        part = rows[offset:offset + size]
        inputs, targets = pack_lm(part)
        logits, _, _ = model(inputs)
        total += float(sequence_loss(logits, targets)) * len(part)
    return total / len(rows)


@torch.no_grad()
def original_classifier(model, rows, size=16):
    model.eval()
    total, correct, confusion = 0.0, 0, [[0, 0], [0, 0]]
    for offset in range(0, len(rows), size):
        part = rows[offset:offset + size]
        inputs, ends, labels = pack_classifier(part)
        _, hidden, _ = model(inputs, lm_head=False)
        logits = model.classifier(hidden[torch.arange(len(part)), ends])
        total += float(F.cross_entropy(logits, labels)) * len(part)
        predicted = logits.argmax(-1)
        correct += int((predicted == labels).sum())
        for label, prediction in zip(labels.tolist(), predicted.tolist()):
            confusion[label][prediction] += 1
    return {'loss': total / len(rows), 'accuracy': correct / len(rows), 'confusion': confusion}


class TrainingEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.model = Model()
        self.rows = [[(index + token) % 5 + 1 for token in range(1 + index % 5)] for index in range(35)]
        self.classifier_rows = [{'ids': row, 'label': index % 2} for index, row in enumerate(self.rows)]

    def test_lm_exact_original_results_and_default_batch_shapes(self):
        rows = self.rows[:19]
        batches = prepare_lm(rows, pack_lm)
        self.assertEqual(tuple(batch.count for batch in batches), (8, 8, 3))
        self.assertEqual(evaluate_lm(self.model, batches, sequence_loss), original_lm(self.model, rows))
        self.assertEqual([shape for shape, _, _ in self.model.calls[:3]],
                         [tuple(pack_lm(rows[start:start + 8])[0].shape) for start in (0, 8, 16)])

    def test_classifier_exact_original_uneven_batches_and_confusion(self):
        batches = prepare_classifier(self.classifier_rows, pack_classifier)
        self.assertEqual(tuple(batch.count for batch in batches), (16, 16, 3))
        actual = evaluate_classifier(self.model, batches)
        self.assertEqual(actual, original_classifier(self.model, self.classifier_rows))
        self.assertEqual(sum(map(sum, actual['confusion'])), 35)
        self.assertEqual(actual['accuracy'], (actual['confusion'][0][0] + actual['confusion'][1][1]) / 35)
        self.assertTrue(all(type(value) is int for row in actual['confusion'] for value in row))
        self.assertTrue(all(sum(row) for row in actual['confusion']))

    def test_lm_preserves_per_sequence_pad_weighting_and_short_final_batch(self):
        rows = [[1], [2, 3, 4, 5], [2, 5]]
        batches = prepare_lm(rows, pack_lm, batch_size=2)
        actual = evaluate_lm(self.model, batches, sequence_loss)
        self.assertEqual(actual, original_lm(self.model, rows, size=2))
        inputs, targets = pack_lm(rows)
        with torch.no_grad():
            logits, _, _ = self.model(inputs)
            token_mean = float(F.cross_entropy(logits.transpose(1, 2), targets, ignore_index=0))
        self.assertNotAlmostEqual(actual, token_mean, places=4)

    def test_auxiliary_components_share_forward_and_preserve_sequence_weighting(self):
        rows = [{'ids': [1, 2], 'label': 0}, {'ids': [1, 2, 3, 4, 5], 'label': 1},
                {'ids': [2, 5, 3], 'label': 0}]
        batches = prepare_classifier(rows, pack_classifier, batch_size=2)
        actual = evaluate_classifier(self.model, batches, auxiliary_weight=.5, lm_loss_fn=sequence_loss)
        self.assertEqual(len(self.model.calls), 2)  # Classification and LM use the same forward.
        self.assertEqual(self.model.lm_head_calls, [True, True])
        objective = actual['objective']
        # Independent per-sequence NLL: no PAD or short-final-batch weighting.
        sequence_means, all_token_losses = [], []
        for row in rows:
            ids = torch.tensor(row['ids'])
            token_losses = -self.model.table.detach()[ids[:-1]].log_softmax(-1).gather(1, ids[1:, None]).flatten()
            sequence_means.append(float(token_losses.mean()))
            all_token_losses.extend(token_losses.tolist())
        self.assertAlmostEqual(objective['lmLoss'], sum(sequence_means) / len(rows), places=6)
        self.assertNotAlmostEqual(objective['lmLoss'], sum(all_token_losses) / len(all_token_losses), places=4)
        original = original_classifier(self.model, rows, size=2)
        self.assertEqual({key: actual[key] for key in original}, original)
        self.assertEqual(objective['classificationLoss'], original['loss'])
        self.assertEqual(objective['sampleCount'], 3)
        self.assertEqual(objective['auxiliaryContribution'], .5 * objective['lmLoss'])
        self.assertEqual(objective['totalLoss'], objective['classificationLoss'] + .5 * objective['lmLoss'])

    def test_zero_auxiliary_does_not_compute_lm_or_invent_zero_lm_loss(self):
        def forbidden(*_): raise AssertionError('Inactive auxiliary LM must not run')
        batches = prepare_classifier(self.classifier_rows, pack_classifier)
        result = evaluate_classifier(self.model, batches, auxiliary_weight=0, lm_loss_fn=forbidden)
        self.assertEqual(self.model.lm_head_calls, [False] * len(batches))
        self.assertIsNone(result['objective']['lmLoss'])
        self.assertEqual(result['objective']['auxiliaryContribution'], 0)
        self.assertEqual(result['objective']['totalLoss'], result['loss'])
        for invalid in (True, -1, .3, float('nan')):
            with self.assertRaisesRegex(ValueError, 'auxiliary weight'):
                evaluate_classifier(self.model, batches, auxiliary_weight=invalid)
        with self.assertRaisesRegex(ValueError, 'sequence loss callable'):
            evaluate_classifier(self.model, batches, auxiliary_weight=.5)

    def test_single_example_and_equal_logit_tie_follow_original_argmax(self):
        self.model.classifier.weight.data.zero_()
        rows = [{'ids': [2], 'label': 1}]
        actual = evaluate_classifier(self.model, prepare_classifier(rows, pack_classifier))
        self.assertEqual(actual, original_classifier(self.model, rows))
        self.assertEqual(actual['confusion'], [[0, 0], [1, 0]])
        self.assertEqual(evaluate_lm(self.model, prepare_lm([[1]], pack_lm), sequence_loss),
                         original_lm(self.model, [[1]]))

    def test_each_batch_is_packed_once_and_cached_storage_survives_packer_mutation(self):
        calls, storage = [], []
        def pack(part):
            calls.append(len(part)); result = pack_classifier(part); storage.extend(result); return result
        batches = prepare_classifier(self.classifier_rows, pack)
        self.assertEqual(calls, [16, 16, 3])
        expected = evaluate_classifier(self.model, batches)
        for tensor in storage: tensor.zero_()
        self.assertEqual(evaluate_classifier(self.model, batches), expected)
        self.assertEqual(calls, [16, 16, 3])
        self.assertEqual(tuple(label for batch in batches for label in batch.cpu_labels),
                         tuple(row['label'] for row in self.classifier_rows))
        with self.assertRaises(FrozenInstanceError): batches[0].count = 100

    def test_lm_cache_has_private_storage_and_does_not_repack(self):
        calls, storage = [], []
        def pack(part):
            calls.append(len(part)); result = pack_lm(part); storage.extend(result); return result
        batches = prepare_lm(self.rows[:19], pack)
        expected = evaluate_lm(self.model, batches, sequence_loss)
        for tensor in storage: tensor.zero_()
        self.assertEqual(evaluate_lm(self.model, batches, sequence_loss), expected)
        self.assertEqual(calls, [8, 8, 3])
        self.assertTrue(all(not batch.inputs.requires_grad and not batch.targets.requires_grad for batch in batches))

    def test_no_grad_rng_or_parameter_changes_and_original_eval_mode_side_effect(self):
        for parameter in self.model.parameters(): parameter.grad = torch.ones_like(parameter) * .25
        parameters = [value.detach().clone() for value in self.model.parameters()]
        gradients = [value.grad.clone() for value in self.model.parameters()]
        py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone()
        lm = prepare_lm(self.rows[:19], pack_lm)
        classifier = prepare_classifier(self.classifier_rows, pack_classifier)
        self.model.train()
        first_lm = evaluate_lm(self.model, lm, sequence_loss)
        first_classifier = evaluate_classifier(self.model, classifier)
        first_objective = evaluate_classifier(self.model, classifier, auxiliary_weight=.5, lm_loss_fn=sequence_loss)
        self.assertEqual(first_lm, evaluate_lm(self.model, lm, sequence_loss))
        self.assertEqual(first_classifier, evaluate_classifier(self.model, classifier))
        self.assertEqual(first_objective, evaluate_classifier(self.model, classifier, auxiliary_weight=.5, lm_loss_fn=sequence_loss))
        self.assertTrue(all(not training and not gradients_enabled for _, training, gradients_enabled in self.model.calls))
        self.assertTrue(all(not module.training for module in self.model.modules()))
        for parameter, before, gradient in zip(self.model.parameters(), parameters, gradients):
            self.assertTrue(torch.equal(parameter, before))
            self.assertTrue(torch.equal(parameter.grad, gradient))
        self.assertEqual(random.getstate(), py_state)
        after_np = np.random.get_state()
        self.assertEqual(after_np[0], np_state[0]); self.assertTrue(np.array_equal(after_np[1], np_state[1]))
        self.assertEqual(after_np[2:], np_state[2:])
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_state))

    def test_measurement_host_transfers_are_constant_not_per_batch(self):
        lm = prepare_lm(self.rows, pack_lm)
        classifier = prepare_classifier(self.classifier_rows, pack_classifier)
        original_cpu, calls = torch.Tensor.cpu, []
        def observed_cpu(tensor, *args, **kwargs):
            calls.append(tuple(tensor.shape))
            return original_cpu(tensor, *args, **kwargs)
        with patch.object(torch.Tensor, 'cpu', observed_cpu):
            evaluate_lm(self.model, lm, sequence_loss)
            self.assertEqual(calls, [(5,)])
            calls.clear()
            evaluate_classifier(self.model, classifier)
            self.assertEqual(calls, [(3,), (35,)])
            calls.clear()
            evaluate_classifier(self.model, classifier, auxiliary_weight=.5, lm_loss_fn=sequence_loss)
            self.assertEqual(calls, [(3, 2), (35,)])

    def test_rejects_empty_or_invalid_batch_size_before_packing(self):
        def forbidden(_): raise AssertionError('Must not pack invalid requests')
        for prepare in (prepare_lm, prepare_classifier):
            with self.assertRaises(ValueError): prepare([], forbidden)
            for size in (0, -1, True, 1.5):
                with self.assertRaises(ValueError): prepare([1], forbidden, batch_size=size)
        with self.assertRaises(ValueError): evaluate_lm(self.model, (), sequence_loss)
        with self.assertRaises(ValueError): evaluate_classifier(self.model, ())

    def test_invalid_packed_tensors_and_nonbinary_labels(self):
        with self.assertRaisesRegex(ValueError, 'int64'):
            prepare_lm([[1]], lambda _: (torch.ones(1, 1), torch.ones(1, 1)))
        with self.assertRaisesRegex(ValueError, 'matching'):
            prepare_lm([[1]], lambda _: (torch.ones(1, 2, dtype=torch.long), torch.ones(1, 1, dtype=torch.long)))
        with self.assertRaisesRegex(ValueError, 'example count'):
            prepare_classifier([1], lambda _: (torch.ones(2, 1, dtype=torch.long), torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long)))
        with self.assertRaisesRegex(ValueError, '0 or 1'):
            prepare_classifier([{'ids': [1], 'label': 2}], pack_classifier)

    def test_dependency_failure_is_not_retried(self):
        calls = []
        def failed_loss(*_): calls.append(True); raise RuntimeError('Dependency failure')
        batches = prepare_lm(self.rows, pack_lm)
        with self.assertRaisesRegex(RuntimeError, 'Dependency failure'):
            evaluate_lm(self.model, batches, failed_loss)
        self.assertEqual(len(calls), 1)


if __name__ == '__main__':
    unittest.main()
