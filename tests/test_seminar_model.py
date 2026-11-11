"""Frozen inference/export contract against real isolated training fixtures."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch

from export_seminar_model import export_bundle, attach_classifier_bundle
from seminar_model import FrozenLanguageModel, file_sha256
from training_engine import TrainingLab, EOS, BOS
from test_training_engine import make_fixture


class SeminarModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.data = cls.root / 'data'
        cls.data.mkdir()
        make_fixture(cls.data)
        cls.lab = TrainingLab(cls.data, cls.root / 'training')
        cls.lab.train(2)
        cls.lab.start_transfer(seed=17)
        cls.lab.train_transfer(1)
        cls.lab.save_live()
        cls.session = cls.root / 'training/live/session.pt'
        cls.session_hash = file_sha256(cls.session)
        cls.bundle = cls.root / 'bundle'
        cls.metadata = export_bundle(cls.session, cls.data / 'tokenizer.json', cls.bundle)
        cls.model = FrozenLanguageModel(cls.bundle)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_bundle_contains_exact_weights_only_and_no_training_state(self):
        self.assertEqual({p.name for p in self.bundle.iterdir()}, {'model.pt', 'tokenizer.json', 'manifest.json'})
        saved = torch.load(self.bundle / 'model.pt', map_location='cpu', weights_only=True)
        self.assertNotIn('optimizer', saved)
        self.assertNotIn('transfer', saved)
        self.assertNotIn('random', saved)
        self.assertEqual(set(saved['model']), set(self.lab.model.state_dict()))
        for key, tensor in self.lab.model.state_dict().items():
            self.assertTrue(torch.equal(saved['model'][key], tensor.cpu()), key)
        self.assertEqual(file_sha256(self.session), self.session_hash)
        self.assertEqual(self.metadata['step'], 2)
        self.assertEqual(self.metadata['parameters'], sum(p.numel() for p in self.lab.model.parameters()))
        self.assertEqual((self.metadata['trainContext'], self.metadata['modelContext']), (64, 128))
        self.assertEqual(self.metadata['snapshotId'], file_sha256(self.bundle / 'model.pt'))
        returned = self.model.metadata()
        returned['step'] = 999
        returned['architecture']['width'] = 1
        self.assertEqual(self.model.metadata(), self.metadata)

    def test_frozen_forward_and_sampling_match_original_training_model(self):
        prompts = ['the good story', ' '.join(['the', 'good', 'story'] * 55)]
        before = {k: v.clone() for k, v in self.lab.model.state_dict().items()}
        for text in prompts:
            for temperature in (0, 0.8, 1e-300):
                with self.subTest(long=len(text) > 100, temperature=temperature):
                    expected = self.lab.generate(text, max_new_tokens=12, temperature=temperature,
                                                 seed=7, sliding_window=True, window_size=64)
                    actual = self.model.generate(text, max_new_tokens=12, temperature=temperature, seed=7)
                    for key in ('tokens', 'continuation', 'fullText', 'stopReason', 'context', 'window'):
                        self.assertEqual(actual[key], expected[key], key)
        self.assertEqual(self.lab.step_count, 2)
        for key, tensor in self.lab.model.state_dict().items():
            self.assertTrue(torch.equal(before[key], tensor))

    def test_stream_json_and_global_rng_unchanged(self):
        expected = self.model.generate('the good story', max_new_tokens=15, seed=27)
        before = torch.random.get_rng_state().clone()
        weights = {key: tensor.clone() for key, tensor in self.model._model.state_dict().items()}
        events = []
        result = self.model.generate('the good story', max_new_tokens=15, seed=27, on_event=events.append)
        self.assertEqual(result, expected)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertEqual(events[0]['type'], 'start')
        self.assertEqual(events[-1], {'type': 'done', 'result': result})
        for index, event in enumerate(events[1:-1]):
            self.assertEqual(event['index'], index)
            self.assertEqual(event['token'], result['tokens'][index])
            self.assertEqual(event['continuation'], self.lab.tokenizer.decode(
                [token['id'] for token in result['tokens'][:index + 1] if token['id'] != EOS]))
        events[-1]['result']['source']['step'] = 999
        self.assertEqual(result['source']['step'], 2)
        for key, tensor in self.model._model.state_dict().items():
            self.assertTrue(torch.equal(weights[key], tensor))
        self.assertTrue(all(not parameter.requires_grad for parameter in self.model._model.parameters()))

    def test_eos_ends_immediately_even_at_cap_and_never_fills_remaining_budget(self):
        for eos_at, cap in ((1, 160), (3, 160), (3, 3)):
            inputs, events = [], []

            def controlled(ids):
                inputs.append(ids[0].tolist())
                logits = torch.full((*ids.shape, self.lab.vocab_size), -100.0)
                logits[:, :, EOS if len(inputs) == eos_at else 5] = 100
                return logits

            with self.subTest(eos_at=eos_at, cap=cap), patch.object(self.model._model, 'forward', side_effect=controlled):
                result = self.model.generate('the good', max_new_tokens=cap, temperature=0, on_event=events.append)
            self.assertEqual(len(inputs), eos_at)
            self.assertEqual(len(result['tokens']), eos_at)
            self.assertEqual(result['tokens'][-1]['id'], EOS)
            self.assertEqual(result['stopReason'], 'eos')
            self.assertNotIn('[EOS]', result['continuation'])
            self.assertEqual(len(events), eos_at + 2)
            self.assertEqual(events[-1]['type'], 'done')

    def test_160_token_cap_and_exact_sliding_suffix(self):
        inputs = []
        text = ' '.join(['the', 'good', 'story'] * 70)
        prompt = [BOS] + self.lab.tokenizer.encode(text).ids

        def controlled(ids):
            inputs.append(ids[0].tolist())
            logits = torch.full((*ids.shape, self.lab.vocab_size), -100.0)
            logits[:, :, 5] = 100
            return logits

        events = []
        with patch.object(self.model._model, 'forward', side_effect=controlled):
            result = self.model.generate(text, temperature=0, on_event=events.append)
        self.assertEqual(len(result['tokens']), 160)
        self.assertEqual(result['stopReason'], 'length')
        for index, ids in enumerate(inputs):
            self.assertEqual(ids, (prompt + [5] * index)[-64:])
            self.assertEqual(events[index + 1]['window'],
                             {'inputTokens': 64, 'droppedTokens': len(prompt) + index - 64})
        self.assertEqual(result['window'], events[-2]['window'])

    def test_invalid_inputs_emit_nothing_and_disconnect_does_not_damage_next_request(self):
        invalid = [{'text': ''}, {'text': ' '}, {'text': 'a' * 20001},
                   {'max_new_tokens': True}, {'max_new_tokens': 0}, {'max_new_tokens': 161},
                   {'temperature': float('nan')}, {'temperature': True}, {'temperature': -1},
                   {'seed': True}, {'seed': -1}, {'seed': 2147483648}]
        for arguments in invalid:
            events = []
            with self.subTest(arguments=str(arguments)[:90]), self.assertRaises(ValueError):
                self.model.generate(**{'text': 'the', **arguments}, on_event=events.append)
            self.assertEqual(events, [])
        expected = self.model.generate('the good', max_new_tokens=4, seed=41)

        def disconnected(event):
            if event['type'] == 'token':
                raise BrokenPipeError('visitor disconnected')

        with self.assertRaises(BrokenPipeError):
            self.model.generate('the good', max_new_tokens=4, seed=41, on_event=disconnected)
        self.assertEqual(expected, self.model.generate('the good', max_new_tokens=4, seed=41))

    def test_runtime_rejects_hash_mismatch_extra_checkpoint_state_and_wrong_tensor_shapes(self):
        for corruption in ('file', 'metadata', 'weights', 'optimizer', 'tokenizer'):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as temporary:
                bundle = Path(temporary) / 'bundle'
                shutil.copytree(self.bundle, bundle)
                manifest = json.loads((bundle / 'manifest.json').read_text())
                if corruption == 'file':
                    with (bundle / 'model.pt').open('ab') as handle:
                        handle.write(b'invalid')
                elif corruption == 'metadata':
                    manifest['snapshotId'] = 'a' * 64
                else:
                    checkpoint = torch.load(bundle / 'model.pt', weights_only=True)
                    if corruption == 'weights':
                        checkpoint['model']['token.weight'] = torch.zeros((1, 1))
                    elif corruption == 'optimizer':
                        checkpoint['optimizer'] = {'state': {}}
                    else:
                        checkpoint['tokenizerSha256'] = 'b' * 64
                    torch.save(checkpoint, bundle / 'model.pt')
                    manifest['snapshotId'] = manifest['files']['model.pt'] = file_sha256(bundle / 'model.pt')
                (bundle / 'manifest.json').write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    FrozenLanguageModel(bundle)

    def test_export_rejects_existing_destination_or_wrong_tokenizer_without_overwriting(self):
        manifest_before = (self.bundle / 'manifest.json').read_bytes()
        with self.assertRaises(FileExistsError):
            export_bundle(self.session, self.data / 'tokenizer.json', self.bundle)
        self.assertEqual(manifest_before, (self.bundle / 'manifest.json').read_bytes())
        invalid_tokenizer = self.root / 'wrong-tokenizer.json'
        invalid_tokenizer.write_text('{}')
        target = self.root / 'must-not-exist'
        with self.assertRaises(ValueError):
            export_bundle(self.session, invalid_tokenizer, target)
        self.assertFalse(target.exists())
        self.assertEqual(file_sha256(self.session), self.session_hash)

    def test_load_preserves_rng_and_rejects_invalid_thread_count(self):
        before = torch.random.get_rng_state().clone()
        model = FrozenLanguageModel(self.bundle)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertEqual(model.metadata(), self.metadata)
        for value in (True, 0, 5, '1'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                FrozenLanguageModel(self.bundle, threads=value)

    def test_attached_classifier_matches_both_live_branches_and_preserves_story(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'attached'
            metadata = attach_classifier_bundle(self.session, self.bundle, output)
            frozen = FrozenLanguageModel(output)
            self.assertTrue(metadata['classifier']['available'])
            self.assertEqual(metadata['snapshotId'], self.metadata['snapshotId'])
            self.assertEqual((output / 'model.pt').read_bytes(), (self.bundle / 'model.pt').read_bytes())
            exported = torch.load(output / 'classifier.pt', weights_only=True)
            self.assertEqual(set(exported['branches']), {'pretrained', 'scratch'})
            for name, tensors in exported['branches'].items():
                self.assertEqual(set(tensors), set(self.lab._transfer['branches'][name]['model'].state_dict()))
                self.assertNotIn('optimizer', tensors)
            rng = torch.random.get_rng_state().clone()
            for text in ('the good story', 'bad story', 'the good ' * 70):
                actual = frozen.classify(text)
                for item in actual['branches']:
                    expected = self.lab.predict_transfer(text, branch=item['branch'])
                    for key in ('branch', 'prediction', 'step', 'source', 'runId'):
                        self.assertEqual(item[key], expected[key])
                    for first, second in zip(item['probabilities'], expected['probabilities']):
                        self.assertEqual(first['label'], second['label'])
                        self.assertAlmostEqual(first['p'], second['p'], places=6)
            self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
            self.assertEqual(file_sha256(self.session), self.session_hash)
            for name in ('pretrained', 'scratch'):
                self.assertTrue(all(not p.requires_grad for p in frozen._classifiers[name].parameters()))

    def test_classifier_requires_trained_pair_and_rejects_corrupt_or_extra_state(self):
        with self.assertRaisesRegex(ValueError, 'No trained classifier'):
            self.model.classify('good')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            saved = torch.load(self.session, weights_only=False)
            saved['transfer']['step'] = 0
            torch.save(saved, root / 'untrained.pt')
            with self.assertRaises(ValueError):
                attach_classifier_bundle(root / 'untrained.pt', self.bundle, root / 'untrained')
            self.assertFalse((root / 'untrained').exists())
            saved['transfer']['step'] = 1
            saved['transfer']['optimizerFailure'] = True
            torch.save(saved, root / 'failed-pair.pt')
            with self.assertRaisesRegex(ValueError, 'failed transfer pair'):
                attach_classifier_bundle(root / 'failed-pair.pt', self.bundle, root / 'failed-pair')
            self.assertFalse((root / 'failed-pair').exists())
            for corruption in ('hash', 'optimizer', 'nonfinite'):
                output = root / corruption
                attach_classifier_bundle(self.session, self.bundle, output)
                payload = torch.load(output / 'classifier.pt', weights_only=True)
                if corruption == 'optimizer':
                    payload['optimizer'] = {}
                elif corruption == 'nonfinite':
                    next(iter(payload['branches']['pretrained'].values())).flatten()[0] = float('nan')
                torch.save(payload, output / 'classifier.pt')
                if corruption != 'hash':
                    manifest = json.loads((output / 'manifest.json').read_text())
                    manifest['files']['classifier.pt'] = file_sha256(output / 'classifier.pt')
                    (output / 'manifest.json').write_text(json.dumps(manifest))
                else:
                    with (output / 'classifier.pt').open('ab') as stream:
                        stream.write(b'bad')
                with self.assertRaises(ValueError):
                    FrozenLanguageModel(output)


if __name__ == '__main__':
    unittest.main()
