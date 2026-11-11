"""Batch contracts: padding, boundaries, sampling order and storage isolation."""
import copy
import random
import unittest

import torch

from training_engine import TrainingLab, BOS, EOS, PAD, CLS, sentiment_batch


class BatchPackingTests(unittest.TestCase):
    def make_lab(self, seed=43):
        # Packing does not require a model, tokenizer, CUDA, or data files.
        lab = TrainingLab.__new__(TrainingLab)
        lab.device = torch.device('cpu')
        lab.random = random.Random(seed)
        return lab

    def assert_int_batch(self, tensor, expected):
        self.assertEqual(tensor.tolist(), expected)
        self.assertEqual(tensor.dtype, torch.long)
        self.assertEqual(tensor.device.type, 'cpu')
        self.assertTrue(tensor.is_contiguous())

    def test_lm_eos_padding_truncation_and_index_order(self):
        lab = self.make_lab()
        rows = [{'ids': []}, {'ids': [5]}, {'ids': [6, 7, 8, 9, 10]}]
        before = copy.deepcopy(rows)
        state = lab.random.getstate()
        x, y = lab._batch(rows, indices=[2, 0, 1, 2], context=4)
        self.assert_int_batch(x, [[BOS, 6, 7, 8], [BOS, PAD, PAD, PAD],
                                  [BOS, 5, PAD, PAD], [BOS, 6, 7, 8]])
        self.assert_int_batch(y, [[6, 7, 8, 9], [EOS, PAD, PAD, PAD],
                                  [5, EOS, PAD, PAD], [6, 7, 8, 9]])
        self.assertEqual(rows, before)
        self.assertEqual(lab.random.getstate(), state)
        x[0, 0] = 999
        fresh_x, fresh_y = lab._batch(rows, indices=[2, 0, 1, 2], context=4)
        self.assertEqual(fresh_x[0, 0].item(), BOS)
        self.assertEqual(fresh_y[0, 0].item(), 6)
        self.assertEqual(y[0, 0].item(), 6)

    def test_random_windows_preserve_rng_consumption_and_exact_tokens(self):
        lab = self.make_lab()
        reference = random.Random(43)
        rows = [{'ids': []}, {'ids': list(range(5, 24))}, {'ids': [25, 26]}]
        # Include repeated indices; one RNG draw is required per selected row,
        # even for rows shorter than the context.
        selected = [1, 0, 2, 1]
        for context in (1, 4, 64, 128):
            expected = []
            for index in selected:
                sequence = [BOS, *rows[index]['ids'], EOS]
                start = reference.randrange(max(1, len(sequence) - context))
                expected.append(sequence[start:start + context + 1])
            width = max(len(sequence) - 1 for sequence in expected)
            x, y = lab._batch(rows, selected, random_windows=True, context=context)
            self.assert_int_batch(x, [seq[:-1] + [PAD] * (width - len(seq) + 1) for seq in expected])
            self.assert_int_batch(y, [seq[1:] + [PAD] * (width - len(seq) + 1) for seq in expected])
            self.assertEqual(lab.random.getstate(), reference.getstate())

    def test_sentiment_cls_position_truncation_boolean_labels_and_isolation(self):
        rows = [{'ids': [], 'label': False}, {'ids': [5], 'label': True},
                {'ids': [6, 7, 8, 9, 10], 'label': 0}]
        before = copy.deepcopy(rows)
        x, ends, labels = sentiment_batch(rows, 'cpu', context=4)
        self.assert_int_batch(x, [[BOS, CLS, PAD, PAD], [BOS, 5, CLS, PAD], [BOS, 6, 7, CLS]])
        self.assert_int_batch(ends, [1, 2, 3])
        self.assert_int_batch(labels, [0, 1, 0])
        self.assertEqual(x[torch.arange(3), ends].tolist(), [CLS] * 3)
        self.assertEqual(rows, before)
        x[0, 0] = 999
        self.assertEqual(sentiment_batch(rows, 'cpu', context=4)[0][0, 0].item(), BOS)


if __name__ == '__main__':
    unittest.main()
