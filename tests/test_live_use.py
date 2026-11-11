"""Live-use acceptance on isolated fixture data; no remote service or real data writes."""
from pathlib import Path
import copy
import hashlib
import random
import tempfile
import unittest
from unittest.mock import patch

import torch

from training_engine import TrainingLab, BOS, EOS, PAD, CLS, UNK, evaluate_classifier, evaluate_cached_classifier
from test_training_engine import make_fixture


def model_digest(model):
    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class LiveUseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / 'data'
        self.data.mkdir()
        make_fixture(self.data)
        self.lab = TrainingLab(self.data, self.root / 'artifacts')

    def tearDown(self):
        self.temp.cleanup()

    def assert_state_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertIsInstance(right, torch.Tensor)
            self.assertTrue(torch.equal(left.cpu(), right.cpu()))
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.assert_state_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(type(left), type(right))
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_state_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_generation_uses_current_weights_and_preserves_all_training_state(self):
        lab = self.lab
        lab.train(2)
        preview = lab.preview('train-0')
        before, pending = lab._capture(), copy.deepcopy(lab._preview)
        python_rng = random.getstate()
        live = lab.generate('the good', max_new_tokens=6, seed=7)
        self.assertEqual(live, lab.generate('the good', max_new_tokens=6, seed=7))
        baseline = lab.generate('the good', max_new_tokens=6, seed=7, model='random')
        self.assertEqual(baseline['source'], {'model': 'random', 'step': 0, 'seed': 42, 'source': 'random'})
        self.assertEqual(live['source']['step'], 2)
        for result in (live, baseline):
            self.assertTrue(1 <= len(result['tokens']) <= 6)
            self.assertTrue(all(0 < token['p'] <= 1 for token in result['tokens']))
            self.assertTrue(all(token['id'] not in (PAD, BOS, CLS, UNK) for token in result['tokens']))
        self.assert_state_equal(before, lab._capture())
        self.assert_state_equal(pending, lab._preview)
        self.assertEqual(python_rng, random.getstate())
        self.assertEqual(lab.commit(preview['previewId'])['step'], 3)

        # The selected logits must belong to the actual live parameters.
        text = 'the good'
        prompt = [BOS] + lab.tokenizer.encode(text).ids
        with torch.no_grad():
            logits = lab.model(torch.tensor([prompt]))[0][0, -1].clone()
        logits[[PAD, BOS, CLS, UNK]] = float('-inf')
        expected = (logits / 0.6).softmax(-1)
        result = lab.generate(text, max_new_tokens=1, temperature=0.6)
        token = result['tokens'][0]
        self.assertAlmostEqual(token['p'], float(expected[token['id']]), places=7)
        greedy = lab.generate(text, max_new_tokens=1, temperature=0)
        self.assertEqual(greedy['tokens'][0]['id'], int(logits.argmax()))
        self.assertEqual(greedy['tokens'][0]['p'], 1.0)
        for temperature in (1e-30, 1e-300, float.fromhex('0x0.0000000000001p-1022')):
            with self.subTest(temperature=temperature):
                tiny_temperature = lab.generate(text, max_new_tokens=1, temperature=temperature)
                self.assertEqual(tiny_temperature['tokens'][0]['id'], int(logits.argmax()))
                self.assertEqual(tiny_temperature['tokens'][0]['p'], 1.0)

    def test_extremely_small_temperature_preserves_tied_maxima_probabilities(self):
        lab = self.lab

        def tied_logits(ids, *args, **kwargs):
            logits = torch.zeros((*ids.shape, lab.vocab_size), device=ids.device)
            logits[:, :, [5, 7]] = 1.0
            return logits, None, None

        with patch.object(lab.model, 'forward', side_effect=tied_logits):
            for temperature in (1e-300, float.fromhex('0x0.0000000000001p-1022')):
                with self.subTest(temperature=temperature):
                    result = lab.generate('the', max_new_tokens=1, temperature=temperature)
                    self.assertIn(result['tokens'][0]['id'], (5, 7))
                    self.assertEqual(result['tokens'][0]['p'], 0.5)

    def test_random_baseline_is_initialization_seed_not_sampling_seed(self):
        lab = self.lab
        initial = lab.generate('the good', max_new_tokens=5, temperature=0, model='live')
        lab.train(3)
        random_a = lab.generate('the good', max_new_tokens=5, temperature=0, seed=3, model='random')
        random_b = lab.generate('the good', max_new_tokens=5, temperature=0, seed=999, model='random')
        self.assertEqual(initial['tokens'], random_a['tokens'])
        self.assertEqual(random_a['tokens'], random_b['tokens'])

    def test_generation_stopping_prompt_and_decoding(self):
        lab = self.lab
        observed = []

        def fixed_logits(ids, *args, **kwargs):
            observed.append(ids.clone())
            logits = torch.full((*ids.shape, lab.vocab_size), -100.0, device=ids.device)
            logits[:, :, EOS] = 100.0
            return logits, None, None

        with patch.object(lab.model, 'forward', side_effect=fixed_logits):
            output = lab.generate('the good', max_new_tokens=64)
        self.assertEqual(observed[0][0].tolist(), [BOS] + lab.tokenizer.encode('the good').ids)
        self.assertEqual(output['stopReason'], 'eos')
        self.assertEqual(len(output['tokens']), 1)
        self.assertEqual(output['continuation'], '')
        self.assertEqual(output['fullText'], lab.tokenizer.decode(lab.tokenizer.encode('the good').ids))

        def non_eos(ids, *args, **kwargs):
            logits = torch.zeros((*ids.shape, lab.vocab_size), device=ids.device)
            logits[:, :, 5] = 100.0
            return logits, None, None

        with patch.object(lab.model, 'forward', side_effect=non_eos):
            output = lab.generate(' '.join(['the'] * 126), max_new_tokens=64, temperature=0)
        self.assertEqual(output['stopReason'], 'context')
        self.assertEqual(len(output['tokens']), 1)
        self.assertEqual(output['fullText'], lab.tokenizer.decode([5] * 127))
        for arguments in ({'text': ' '.join(['the'] * 127)}, {'text': ''}, {'max_new_tokens': 1001},
                          {'max_new_tokens': True}, {'seed': -1}, {'seed': True},
                          {'temperature': float('nan')}, {'temperature': 1.6}, {'model': 'recorded'}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                lab.generate(**{'text': 'the', **arguments})

    def test_transfer_copy_fairness_updates_evaluation_and_parent_isolation(self):
        lab = self.lab
        self.assertEqual(lab.transfer_state(), {'available': False})
        lab.train(2)
        lab.preview('train-0')
        before, preview = lab._capture(), copy.deepcopy(lab._preview)
        original_hash = model_digest(lab.model)
        evaluated = []

        def measured(model, batches, **kwargs):
            evaluated.append(batches)
            return evaluate_cached_classifier(model, batches, **kwargs)

        with patch('training_engine.evaluate_cached_classifier', side_effect=measured):
            state = lab.start_transfer(seed=23, auxiliary_weight=0.5)
        self.assertEqual(state['source'], {'step': 2, 'seed': 42, 'source': 'random', 'modelSha256': original_hash})
        self.assertEqual([id(batches) for batches in evaluated],
                         [id(lab._sentiment_eval_batches[key]) for key in
                          ('train', 'validation', 'train', 'validation')])
        branches = lab._transfer['branches']
        trained, scratch = branches['pretrained']['model'], branches['scratch']['model']
        self.assert_state_equal(trained.classifier.state_dict(), scratch.classifier.state_dict())
        for name, parameter in trained.named_parameters():
            self.assertTrue(parameter.requires_grad)
            self.assertNotEqual(parameter.data_ptr(), dict(lab.model.named_parameters())[name].data_ptr())
            if not name.startswith('classifier.'):
                self.assert_state_equal(parameter, dict(lab.model.named_parameters())[name])
        self.assertNotEqual(model_digest(trained), model_digest(scratch))
        initial = {name: model_digest(branch['model']) for name, branch in branches.items()}
        batches = {'pretrained': [], 'scratch': []}
        handles = []
        for name, branch in branches.items():
            def capture(model, arguments, name=name):
                if model.training:
                    batches[name].append(arguments[0].detach().cpu().clone())
            handles.append(branch['model'].register_forward_pre_hook(capture))
        try:
            result = lab.train_transfer(steps=3)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(result['step'], 3)
        self.assertEqual(len(batches['pretrained']), 3)
        self.assert_state_equal(batches['pretrained'], batches['scratch'])
        for name, branch in branches.items():
            self.assertNotEqual(initial[name], model_digest(branch['model']))
            self.assertTrue(branch['optimizer'].state)
            self.assertTrue(all(parameter.grad is not None for parameter in branch['model'].parameters()))
            self.assertEqual([row['step'] for row in branch['history']], [0])
            self.assertEqual(result['lastMeasuredStep'], 0)
            prediction = lab.predict_transfer('this good story is nice .', branch=name)
            self.assertEqual(prediction['source'], state['source'])
            self.assertAlmostEqual(sum(row['p'] for row in prediction['probabilities']), 1.0, places=6)
            self.assertEqual(prediction['step'], 3)
        self.assert_state_equal(before, lab._capture())
        self.assert_state_equal(preview, lab._preview)
        # UI response mutation cannot change the pair's immutable provenance.
        result['source']['step'] = 9999
        self.assertEqual(lab.transfer_state()['source']['step'], 2)
        lab.train(1)
        self.assertEqual(lab.transfer_state()['source']['modelSha256'], original_hash)

    def test_transfer_reproducibility_and_missing_state_validation(self):
        lab = self.lab
        for operation in (lambda: lab.train_transfer(), lambda: lab.predict_transfer('good')):
            with self.assertRaises(ValueError):
                operation()
        for arguments in ({'seed': True}, {'seed': -1}, {'auxiliary_weight': 1}, {'auxiliary_weight': float('nan')}):
            with self.assertRaises(ValueError):
                lab.start_transfer(**arguments)
        lab.start_transfer(seed=7)
        with self.assertRaises(ValueError):
            lab.predict_transfer('good')
        for arguments in ({'steps': 11}, {'steps': True}, {'learning_rate': float('inf')}):
            with self.assertRaises(ValueError):
                lab.train_transfer(**arguments)
        first = lab.train_transfer(2)
        weights = {name: model_digest(branch['model']) for name, branch in lab._transfer['branches'].items()}
        second = lab.start_transfer(seed=7)
        self.assertNotEqual(first['runId'], second['runId'])
        second = lab.train_transfer(2)
        self.assertEqual(first['branches'], second['branches'])
        self.assertEqual(weights, {name: model_digest(branch['model']) for name, branch in lab._transfer['branches'].items()})

    def test_checkpoint_roundtrip_and_continuation_match_for_lm_and_transfer(self):
        lab = self.lab
        lab.train(2)
        lab.start_transfer(seed=19, auxiliary_weight=0.5)
        lab.train_transfer(2)
        before, transfer = lab._capture(), lab._capture_transfer()
        saved = lab.save_live()
        self.assertEqual((saved['saved'], saved['step'], saved['transferStep']), (True, 2, 2))
        self.assertTrue((lab.artifact_dir / 'live' / 'session.pt').is_file())
        self.assert_state_equal(before, lab._capture())
        self.assert_state_equal(transfer, lab._capture_transfer())
        lab.train(2)
        lab.train_transfer(2)
        expected_lm, expected_transfer = lab._capture(), lab._capture_transfer()
        lab.reset(seed=99)
        lab.start_transfer(seed=99)
        preview = lab.preview('train-0')['previewId']
        restored = lab.restore_live()
        self.assertEqual(restored['step'], 2)
        self.assertNotIn('previewId', restored)
        with self.assertRaises(ValueError):
            lab.commit(preview)
        self.assert_state_equal(before, lab._capture())
        self.assert_state_equal(transfer, lab._capture_transfer())
        lab.train(2)
        lab.train_transfer(2)
        self.assert_state_equal(expected_lm, lab._capture())
        self.assert_state_equal(expected_transfer, lab._capture_transfer())

    def test_restore_rejects_incompatible_or_corrupt_without_mutating_live_state(self):
        lab = self.lab
        with self.assertRaises(FileNotFoundError):
            lab.restore_live()
        lab.train(1)
        lab.start_transfer()
        lab.train_transfer(1)
        lab.save_live()
        path = lab.artifact_dir / 'live' / 'session.pt'
        saved = torch.load(path, weights_only=False)
        before, transfer = lab._capture(), lab._capture_transfer()
        for key, invalid in (('liveFormatVersion', 999), ('tokenizerSha256', 'mismatch'),
                             ('dataManifestSha256', 'mismatch'), ('config', {}), ('vocabSize', 10000)):
            altered = copy.deepcopy(saved)
            altered[key] = invalid
            torch.save(altered, path)
            with self.subTest(key=key), self.assertRaises(ValueError):
                lab.restore_live()
            self.assert_state_equal(before, lab._capture())
            self.assert_state_equal(transfer, lab._capture_transfer())
        altered = copy.deepcopy(saved)
        altered['transfer']['branches']['scratch']['model']['token.weight'] = torch.zeros(2, 2)
        torch.save(altered, path)
        with self.assertRaises(ValueError):
            lab.restore_live()
        self.assert_state_equal(before, lab._capture())
        self.assert_state_equal(transfer, lab._capture_transfer())

    def test_checkpoint_without_transfer_and_snapshot_storage_is_independent(self):
        lab = self.lab
        lab.train(1)
        before = lab._capture()
        self.assertIsNone(lab.save_live()['transferStep'])
        lab.start_transfer()
        first = next(iter(before['optimizer']['state'].values()))['exp_avg'].clone()
        lab.train(2)
        self.assertTrue(torch.equal(first, next(iter(before['optimizer']['state'].values()))['exp_avg']))
        lab.restore_live()
        self.assertFalse(lab.transfer_state()['available'])
        self.assert_state_equal(before, lab._capture())

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is not available in this local test environment')
    def test_cuda_generation_and_transfer_preserve_all_rng_and_lm_state(self):
        lab = TrainingLab(self.data, self.root / 'cuda-artifacts', device='cuda')
        lab.train(1)
        before = lab._capture()
        lab.generate('the good', model='random', max_new_tokens=3)
        for temperature in (1e-300, float.fromhex('0x0.0000000000001p-1022')):
            result = lab.generate('the good', max_new_tokens=1, temperature=temperature)
            self.assertEqual(result['tokens'][0]['p'], 1.0)
        lab.start_transfer(seed=11, auxiliary_weight=0.5)
        lab.train_transfer(1)
        lab.predict_transfer('the good book')
        self.assert_state_equal(before, lab._capture())
        transfer = lab._capture_transfer()
        lab.save_live()
        lab.train_transfer(1)
        lab.restore_live()
        self.assert_state_equal(before, lab._capture())
        self.assert_state_equal(transfer, lab._capture_transfer())
        self.assertTrue(all(parameter.is_cuda for parameter in lab.model.parameters()))


if __name__ == '__main__':
    unittest.main()
