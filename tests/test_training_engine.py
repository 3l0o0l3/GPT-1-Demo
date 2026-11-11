"""Engine correctness tests on an isolated tiny fixture, not reported experiments."""
from pathlib import Path
import copy
import hashlib
import json
import tempfile
import unittest

import torch
from tokenizers import Tokenizer, models, pre_tokenizers

from training_engine import TrainingLab, CONFIG, sequence_loss, record_experiment


def make_fixture(directory):
    words = ['[PAD]', '[BOS]', '[EOS]', '[CLS]', '[UNK]', 'the', 'a', 'good', 'bad',
             'story', 'is', 'nice', 'dull', '.', 'very', 'this', 'book', 'happy', 'sad']
    tok = Tokenizer(models.WordLevel({s: i for i, s in enumerate(words)}, unk_token='[UNK]'))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(directory / 'tokenizer.json'))
    train = [{'id': f'train-{i}', 'text': text, 'ids': tok.encode(text).ids}
             for i, text in enumerate(['the good story is nice .', 'the bad story is dull .',
                                      'a good book is nice .', 'a bad book is dull .'] * 4)]
    val = [{'id': f'val-{i}', 'text': text, 'ids': tok.encode(text).ids}
           for i, text in enumerate(['this good story is nice .', 'this bad story is dull .'])]
    for name, rows in [('stories-train.jsonl', train), ('stories-val.jsonl', val)]:
        (directory / name).write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    sentiment_train = [dict(r, label=i % 2 == 0, source='test-fixture') for i, r in enumerate(train[:8])]
    sentiment_val = [dict(r, label=i % 2 == 0, source='test-fixture') for i, r in enumerate(val)]
    # Unusable test row proves no tokenizer/model/evaluation path consumes it.
    sentiment = {'train': sentiment_train, 'validation': sentiment_val,
                 'test': [{'id': 'NEVER-USED', 'text': None, 'ids': None, 'label': 'INVALID'}]}
    (directory / 'sentiment.json').write_text(json.dumps(sentiment), encoding='utf-8')
    (directory / 'manifest.json').write_text(json.dumps({'version': 1, 'sources': [], 'counts': {},
                                                       'tokenizer': {'vocabSize': len(words)}}), encoding='utf-8')


def digest(state):
    h = hashlib.sha256()
    for key, tensor in state['model'].items():
        h.update(key.encode())
        h.update(tensor.numpy().tobytes())
    h.update(repr(state['random']).encode())
    h.update(state['torchRandom'].numpy().tobytes())
    h.update(str(state['step']).encode())
    for key, value in state['optimizer']['state'].items():
        h.update(str(key).encode())
        for k, v in value.items():
            h.update(k.encode())
            h.update(v.numpy().tobytes() if isinstance(v, torch.Tensor) else repr(v).encode())
    return h.hexdigest()


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.data = cls.root / 'data'
        cls.data.mkdir()
        make_fixture(cls.data)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def make_lab(self):
        return TrainingLab(self.data, self.root / 'artifacts', seed=42)

    def test_architecture_probabilities_and_causality(self):
        lab = self.make_lab()
        metadata = lab.metadata()['architecture']
        self.assertEqual((metadata['layers'], metadata['heads'], metadata['width'], metadata['context']), (4, 4, 128, 128))
        trace = lab.inspect('the good story is nice .', position=3)
        self.assertEqual(trace['target']['id'], trace['targets'][3]['id'])
        for i, row in enumerate(trace['attention']):
            self.assertAlmostEqual(sum(row), 1.0, places=5)
            for j, weight in enumerate(row):
                self.assertEqual(trace['mask'][i][j], j > i)
                if j > i:
                    self.assertEqual(weight, 0)
        x = torch.tensor([[1, 5, 7, 9, 10, 11, 13]])
        y = torch.tensor([[1, 5, 7, 12, 12, 12, 13]])
        with torch.no_grad():
            a = lab.model(x, return_attention=True)[0]
            b = lab.model(y, return_attention=True)[0]
        torch.testing.assert_close(a[:, :3], b[:, :3], atol=1e-6, rtol=1e-6)

    def test_finite_difference_gradient(self):
        lab = self.make_lab()
        model = copy.deepcopy(lab.model).double()
        x, y = lab._input('the good story is nice .')
        loss = sequence_loss(model(x, return_attention=True)[0], y)
        loss.backward()
        parameter = model.blocks[0].qkv.weight
        index = int(parameter.grad.flatten().abs().argmax())
        analytic = float(parameter.grad.flatten()[index])
        old = float(parameter.detach().flatten()[index])
        epsilon = 1e-5
        with torch.no_grad():
            parameter.flatten()[index] = old + epsilon
            plus = float(sequence_loss(model(x, return_attention=True)[0], y))
            parameter.flatten()[index] = old - epsilon
            minus = float(sequence_loss(model(x, return_attention=True)[0], y))
            parameter.flatten()[index] = old
        numerical = (plus - minus) / (2 * epsilon)
        self.assertAlmostEqual(analytic, numerical, delta=1e-6)

    def test_preview_rolls_back_adam_rng_commit_exactly_once(self):
        lab = self.make_lab()
        lab.train(1)  # Exercise a non-empty Adam state.
        original = lab._capture()
        first = lab.preview('train-0', learning_rate=0.0003)
        self.assertFalse(first['applied'])
        self.assertEqual(digest(lab._capture()), digest(original))
        self.assertAlmostEqual(first['loss'], first['before']['batchLoss'], places=5)
        self.assertTrue(first['gradientNorm'] > 0)
        self.assertLessEqual(first['clippedGradientNorm'], 1.0)
        w = first['weight']
        expected = -w['learningRate'] * w['mHat'] / (w['vHat'] ** 0.5 + w['epsilon'])
        self.assertAlmostEqual(w['delta'], expected, delta=1e-7)
        committed = lab.commit(first['previewId'])
        self.assertTrue(committed['lastTrace']['applied'])
        self.assertEqual(committed['step'], original['step'] + 1)
        after = lab.inspect(first['example']['text'], first['after']['position'])
        self.assertAlmostEqual(after['tokenLoss'], first['after']['tokenLoss'], places=7)
        with self.assertRaises(ValueError):
            lab.commit(first['previewId'])
        new_preview = lab.preview('train-0')
        lab.train(1)
        with self.assertRaises(ValueError):
            lab.commit(new_preview['previewId'])

    def test_reset_learning_and_boundaries(self):
        lab = self.make_lab()
        initial = lab._capture()
        before = lab.state()['metrics']['trainLoss']
        lab.train(5)
        learned = lab._evaluate_lm(lab._eval_train)
        self.assertLess(learned, before)
        lab.reset(seed=42)
        self.assertEqual(digest(lab._capture()), digest(initial))
        for kwargs in ({'steps': 0}, {'steps': 11}, {'learning_rate': float('nan')}):
            with self.assertRaises(ValueError):
                lab.train(**kwargs)
        with self.assertRaises(ValueError):
            lab.examples(split='test')
        with self.assertRaises(ValueError):
            lab.preview('val-0')
        with self.assertRaises(ValueError):
            lab.inspect('')
        isolated = TrainingLab(self.data, self.root / 'unavailable')
        self.assertFalse(isolated.recordings()['pretraining']['available'])
        with self.assertRaises(FileNotFoundError):
            isolated.predict('this book is good')

    def test_real_recordings_branch_comparability_and_reload(self):
        lab = self.make_lab()
        records = record_experiment(lab, pretrain_steps=2, finetune_steps=2,
                                    seeds=(7,), eval_every=2, finetune_eval_every=2)
        runs = records['transfer']['runs']
        self.assertEqual([r['name'] for r in runs], ['scratch', 'pretrained', 'auxiliary'])
        self.assertEqual(len({r['classifierInitializationSha256'] for r in runs}), 1)
        self.assertEqual(len({r['minibatchOrderSha256'] for r in runs}), 1)
        self.assertTrue(all(r['completed'] and r['history'][-1]['step'] == 2 for r in runs))
        for run in runs:
            self.assertEqual(sum(sum(row) for row in run['confusion']), len(lab.sentiment_val))
        self.assertTrue(lab.recordings()['pretraining']['available'])
        prediction = lab.predict('this good story is nice .', 'auxiliary')
        self.assertTrue(prediction['trained'])
        self.assertAlmostEqual(sum(p['p'] for p in prediction['probabilities']), 1.0, places=6)
        state = lab.reset('pretrained')
        self.assertEqual(state['source'], 'pretrained')
        self.assertEqual(state['step'], 2)


if __name__ == '__main__':
    unittest.main()
