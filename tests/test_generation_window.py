"""Fixed/sliding inference windows on temporary model/data fixtures."""
import copy
import unittest
from unittest.mock import MagicMock, patch

import torch

import test_live_use as live_fixture
from test_training_service import FakeLab
from training_engine import BOS, EOS
from training_service import TrainingApiError, TrainingService


class GenerationWindowTests(unittest.TestCase):
    setUp = live_fixture.LiveUseTests.setUp
    tearDown = live_fixture.LiveUseTests.tearDown
    assert_state_equal = live_fixture.LiveUseTests.assert_state_equal

    def controlled_generation(self, text, *, window_size, budget, sliding=True, eos_after=None):
        """Keep actual transformer forwards; control only sampled output logits."""
        lab = self.lab
        inputs, positions, events = [], [], []

        def input_hook(model, arguments):
            inputs.append(arguments[0][0].detach().cpu().tolist())

        def position_hook(module, arguments):
            positions.append(arguments[0].detach().cpu().tolist())

        def output_hook(model, arguments, output):
            logits, hidden, attention = output
            selected_id = EOS if eos_after == len(inputs) else (5, 7, 8, 9)[(len(inputs) - 1) % 4]
            controlled = torch.full_like(logits, -20.0)
            controlled[:, :, selected_id] = 20.0
            return controlled, hidden, attention

        handles = [lab.model.register_forward_pre_hook(input_hook),
                   lab.model.position.register_forward_pre_hook(position_hook),
                   lab.model.register_forward_hook(output_hook)]
        tokenizer = MagicMock(wraps=lab.tokenizer)
        try:
            with patch.object(lab, 'tokenizer', tokenizer):
                result = lab.generate(text, max_new_tokens=budget, temperature=0,
                                      sliding_window=sliding, window_size=window_size, on_event=events.append)
            self.assertEqual(tokenizer.encode.call_count, 1, 'Generated text must never be re-tokenized.')
        finally:
            for handle in handles:
                handle.remove()
        return result, inputs, positions, events

    def test_exact_suffix_positions_and_drop_counts_across_128_for_all_window_sizes(self):
        lab = self.lab
        lab.train(1)
        lab.start_transfer(seed=7)
        lab.train_transfer(1)
        lab.preview('train-0')
        before, transfer, preview = lab._capture(), lab._capture_transfer(), copy.deepcopy(lab._preview)
        prompt = [BOS] + lab.tokenizer.encode('the good').ids
        for size in (1, 64, 128):
            with self.subTest(window_size=size):
                result, inputs, positions, events = self.controlled_generation('the good', window_size=size, budget=140)
                self.assertEqual(len(result['tokens']), 140)
                self.assertEqual(result['stopReason'], 'length')
                self.assertEqual(result['context'], {'mode': 'sliding', 'windowSize': size,
                                                     'modelContext': 128, 'promptTokens': len(prompt)})
                self.assertEqual(events[0]['context'], result['context'])
                output_ids = [token['id'] for token in result['tokens']]
                for index, (actual, actual_positions, event) in enumerate(zip(inputs, positions, events[1:-1])):
                    full = prompt + output_ids[:index]
                    self.assertEqual(actual, full[-size:])
                    self.assertEqual(actual_positions, list(range(len(actual))))
                    self.assertEqual(event['window'], {'inputTokens': len(actual), 'droppedTokens': len(full) - len(actual)})
                    self.assertEqual(event['token'], result['tokens'][index])
                self.assertEqual(result['window'], events[-2]['window'])
                self.assertGreater(result['window']['droppedTokens'], 0)
                self.assertEqual(events[-1], {'type': 'done', 'result': result})
                self.assert_state_equal(before, lab._capture())
                self.assert_state_equal(transfer, lab._capture_transfer())
                self.assert_state_equal(preview, lab._preview)

    def test_long_prompt_is_cropped_immediately_only_in_sliding_mode(self):
        lab = self.lab
        text = ' '.join(['the', 'good', 'story', 'is', 'nice'] * 34)
        prompt = [BOS] + lab.tokenizer.encode(text).ids
        self.assertGreater(len(prompt), 128)
        for size in (1, 64, 128):
            with self.subTest(window_size=size):
                result, inputs, positions, events = self.controlled_generation(text, window_size=size, budget=3)
                self.assertEqual(inputs[0], prompt[-size:])
                self.assertNotIn(BOS, inputs[0])
                self.assertEqual(positions[0], list(range(size)))
                self.assertEqual(events[1]['window'], {'inputTokens': size, 'droppedTokens': len(prompt) - size})
                self.assertEqual(result['context']['promptTokens'], len(prompt))
                self.assertEqual(result['fullText'], lab.tokenizer.decode(prompt[1:] + [row['id'] for row in result['tokens']]))
        events = []
        with self.assertRaises(ValueError):
            lab.generate(text, sliding_window=False, window_size=1, on_event=events.append)
        self.assertEqual(events, [])

    def test_fixed_mode_keeps_128_total_positions_and_ignores_valid_window_size(self):
        for size in (1, 64, 128):
            with self.subTest(window_size=size):
                result, inputs, positions, events = self.controlled_generation('the', window_size=size, budget=1000, sliding=False)
                self.assertEqual(len(result['tokens']), 126)
                self.assertEqual(result['stopReason'], 'context')
                self.assertEqual(result['context'], {'mode': 'fixed', 'windowSize': 128, 'modelContext': 128, 'promptTokens': 2})
                self.assertEqual(result['window'], {'inputTokens': 127, 'droppedTokens': 0})
                self.assertEqual([len(ids) for ids in inputs], list(range(2, 128)))
                self.assertTrue(all(row['window']['droppedTokens'] == 0 for row in events[1:-1]))

    def test_eos_stops_before_sliding_budget_and_final_window_describes_eos_input(self):
        result, inputs, _, events = self.controlled_generation('the good', window_size=1, budget=1000, eos_after=3)
        self.assertEqual([row['id'] for row in result['tokens']], [5, 7, EOS])
        self.assertEqual(result['stopReason'], 'eos')
        self.assertEqual(len(inputs), 3)
        self.assertEqual(result['window'], {'inputTokens': 1, 'droppedTokens': 4})
        self.assertEqual(events[-2]['continuation'], events[-3]['continuation'])
        self.assertEqual(result['continuation'], self.lab.tokenizer.decode([5, 7]))

    def test_eos_takes_priority_over_output_cap_and_fixed_context_boundary(self):
        prompt_length = 1 + len(self.lab.tokenizer.encode('the').ids)
        fixed_remaining = 128 - prompt_length
        cases = (
            ('sliding_before_cap', True, 8, 3),
            ('fixed_before_cap', False, 8, 3),
            ('sliding_at_cap', True, 3, 3),
            ('fixed_at_cap', False, 3, 3),
            ('fixed_at_context', False, 1000, fixed_remaining),
            ('fixed_at_cap_and_context', False, fixed_remaining, fixed_remaining),
        )
        for name, sliding, budget, eos_after in cases:
            with self.subTest(case=name):
                result, inputs, _, events = self.controlled_generation(
                    'the', window_size=128, budget=budget, sliding=sliding, eos_after=eos_after)
                self.assertEqual(result['stopReason'], 'eos')
                self.assertEqual(len(inputs), eos_after, 'No model forward is allowed after EOS.')
                self.assertEqual(len(result['tokens']), eos_after)
                self.assertEqual([event['type'] for event in events],
                                 ['start'] + ['token'] * eos_after + ['done'])
                token_events = events[1:-1]
                self.assertEqual([event['index'] for event in token_events], list(range(eos_after)))
                self.assertEqual([event['token'] for event in token_events], result['tokens'])
                self.assertNotIn(EOS, [row['id'] for row in result['tokens'][:-1]])
                self.assertEqual(result['tokens'][-1]['id'], EOS)
                self.assertEqual(events[-1], {'type': 'done', 'result': result})
                self.assertEqual(result['continuation'], self.lab.tokenizer.decode(
                    [row['id'] for row in result['tokens'][:-1]]))
                if not sliding and eos_after == fixed_remaining:
                    self.assertEqual(result['window']['inputTokens'], 127)

    def test_sliding_json_stream_match_and_preserve_lm_pair_preview_rng(self):
        lab = self.lab
        lab.train(1)
        lab.start_transfer(seed=17, auxiliary_weight=.5)
        lab.train_transfer(1)
        lab.preview('train-0')
        before, transfer, pending = lab._capture(), lab._capture_transfer(), copy.deepcopy(lab._preview)
        for model in ('live', 'random'):
            for size in (1, 64, 128):
                with self.subTest(model=model, window_size=size):
                    args = dict(model=model, max_new_tokens=150, sliding_window=True, window_size=size, seed=5)
                    expected = lab.generate('the good', **args)
                    events = []
                    actual = lab.generate('the good', **args, on_event=events.append)
                    self.assertEqual(expected, actual)
                    self.assertEqual(events[-1], {'type': 'done', 'result': expected})
                    self.assertEqual(events[-2]['window'], expected['window'])
                    self.assert_state_equal(before, lab._capture())
                    self.assert_state_equal(transfer, lab._capture_transfer())
                    self.assert_state_equal(pending, lab._preview)

    def test_1000_token_budget_is_supported_and_callback_abort_preserves_state(self):
        lab = self.lab
        before = lab._capture()

        def repeat(ids, *args, **kwargs):
            logits = torch.zeros((*ids.shape, lab.vocab_size), device=ids.device)
            logits[:, :, 5] = 10
            return logits, None, None

        events = []
        with patch.object(lab.model, 'forward', side_effect=repeat):
            result = lab.generate('the', max_new_tokens=1000, temperature=0, sliding_window=True, window_size=1,
                                  on_event=events.append)
        self.assertEqual(len(result['tokens']), 1000)
        self.assertEqual(result['stopReason'], 'length')
        self.assertEqual(result['window'], {'inputTokens': 1, 'droppedTokens': 1000})
        self.assertEqual(len([event for event in events if event['type'] == 'token']), 1000)

        def disconnect(event):
            if event['type'] == 'token':
                torch.rand(4)
                raise BrokenPipeError('Disconnected')

        with self.assertRaises(BrokenPipeError):
            lab.generate('the', max_new_tokens=1000, sliding_window=True, window_size=1, on_event=disconnect)
        self.assert_state_equal(before, lab._capture())

    def test_window_and_budget_validation_is_strict_before_first_event(self):
        lab = self.lab
        invalid = ([{'sliding_window': value} for value in (1, 0, 'true', 'false', None)] +
                   [{'window_size': value} for value in (0, 129, True, False, 1.0, '64', None)] +
                   [{'max_new_tokens': value} for value in (0, 1001, True, '1000')])
        before = lab._capture()
        for arguments in invalid:
            events = []
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                lab.generate('the', on_event=events.append, **arguments)
            self.assertEqual(events, [])
        for sliding in (False, True):
            with self.assertRaises(ValueError):
                lab.generate('x' * 8193, sliding_window=sliding)
        self.assert_state_equal(before, lab._capture())


class WindowServiceTests(unittest.TestCase):
    def test_both_routes_accept_new_budgets_and_map_window_fields(self):
        service = TrainingService('.', factory=FakeLab)
        for action in ('generate', 'generate-stream'):
            for sliding in (False, True):
                for budget, size in ((65, 1), (129, 64), (1000, 128)):
                    with self.subTest(action=action, sliding=sliding, budget=budget, size=size):
                        result = service.dispatch('POST', '/api/training/' + action,
                            {'text': 'the', 'slidingWindow': sliding, 'windowSize': size, 'maxNewTokens': budget})
                        self.assertEqual(result['max_new_tokens'], budget)
                        self.assertEqual(result['sliding_window'], sliding)
                        self.assertEqual(result['window_size'], size)

    def test_invalid_window_inputs_reject_before_engine_initialization(self):
        service = TrainingService('.', factory=lambda **_: self.fail('Engine must not be initialized.'))
        invalid = ([{'slidingWindow': value} for value in (0, 1, 'true', None)] +
                   [{'windowSize': value} for value in (0, 129, True, False, 1.5, '64', None)] +
                   [{'maxNewTokens': value} for value in (1001, 0, True)])
        for action in ('generate', 'generate-stream'):
            for arguments in invalid:
                with self.subTest(action=action, arguments=arguments), self.assertRaises(TrainingApiError) as caught:
                    service.dispatch('POST', '/api/training/' + action, {'text': 'the', **arguments})
                self.assertEqual(caught.exception.status, 400)
                self.assertIsNone(service.lab)
                self.assertFalse(service.lock.locked())


if __name__ == '__main__':
    unittest.main()
