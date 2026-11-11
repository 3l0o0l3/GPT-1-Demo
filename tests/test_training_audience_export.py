"""Current-weight export acceptance on isolated CPU fixtures, never production."""
import base64
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from seminar_model import FrozenLanguageModel
from training_audience_export import encode_audience_export
from training_engine import TrainingLab, CONFIG
from training_service import TrainingService, TrainingApiError
from test_training_engine import make_fixture


def exact(test, left, right):
    if isinstance(left, torch.Tensor):
        test.assertIsInstance(right, torch.Tensor)
        test.assertTrue(torch.equal(left, right))
    elif isinstance(left, dict):
        test.assertEqual(set(left), set(right))
        for key in left:
            exact(test, left[key], right[key])
    elif isinstance(left, (tuple, list)):
        test.assertEqual(type(left), type(right))
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            exact(test, a, b)
    else:
        test.assertEqual(left, right)


class AudienceExportTests(unittest.TestCase):
    def test_roundtrip_matches_current_weights_without_changing_any_training_state(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / 'data'
            data.mkdir()
            make_fixture(data)
            lab = TrainingLab(data, root / 'training')
            lab.train(2)
            lab.start_transfer(seed=7, auxiliary_weight=.5)
            lab.train_transfer(1)
            lab.preview('train-0')
            before = copy.deepcopy((lab._capture(), lab._capture_transfer(), lab._preview))
            training = lab.model.training
            files_before = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in root.rglob('*') if path.is_file()}
            exported = lab.audience_export()
            exact(self, before, (lab._capture(), lab._capture_transfer(), lab._preview))
            self.assertEqual(training, lab.model.training)
            self.assertEqual(files_before, {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                                           for path in root.rglob('*') if path.is_file()})
            self.assertEqual(set(exported), {'exportVersion', 'manifest', 'files'})
            self.assertEqual(set(exported['files']), {'model.pt', 'tokenizer.json'})
            bundle = root / 'bundle'
            bundle.mkdir()
            for name, encoded in exported['files'].items():
                raw = base64.b64decode(encoded, validate=True)
                self.assertEqual(hashlib.sha256(raw).hexdigest(), exported['manifest']['files'][name])
                (bundle / name).write_bytes(raw)
            (bundle / 'manifest.json').write_text(json.dumps(exported['manifest']), encoding='utf8')
            checkpoint = torch.load(io.BytesIO((bundle / 'model.pt').read_bytes()), weights_only=True)
            self.assertNotIn('optimizer', checkpoint)
            self.assertNotIn('transfer', checkpoint)
            exact(self, before[0]['model'], checkpoint['model'])
            frozen = FrozenLanguageModel(bundle)
            self.assertEqual(frozen.metadata()['step'], 2)
            self.assertEqual(frozen.metadata()['classifier'], {'available': False})
            live_output = lab.generate('the good', max_new_tokens=8, temperature=0,
                                       sliding_window=True, window_size=64)
            published_output = frozen.generate('the good', max_new_tokens=8, temperature=0)
            self.assertEqual(live_output['tokens'], published_output['tokens'])
            self.assertEqual(live_output['continuation'], published_output['continuation'])
            lab.train(1)
            self.assertEqual(frozen.metadata()['step'], 2)
            self.assertEqual(lab.audience_export()['exportVersion'], 1)
            self.assertEqual(lab.step_count, 3)

    def test_encoder_rejects_invalid_source_tensor_and_time(self):
        arguments = dict(source={'step': 0, 'seed': 42, 'source': 'random'}, config=CONFIG,
                         vocab_size=5, tokenizer_bytes=b'{}', data_manifest_bytes=b'{}')
        for weights in ({'weight': torch.tensor([float('nan')])},
                        {'weight': torch.tensor([1.], dtype=torch.float16)}, {}):
            with self.subTest(weights=list(weights)), self.assertRaises(ValueError):
                encode_audience_export(weights, **arguments)
        for timestamp in ('2026-11-11T10:00:00', False, ''):
            with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                encode_audience_export({'weight': torch.zeros(1)}, **arguments, saved_at=timestamp)
        with self.assertRaises(ValueError):
            encode_audience_export({'weight': torch.zeros(1)},
                **{**arguments, 'source': {'step': True, 'seed': 42, 'source': 'random'}})

    def test_service_export_is_read_only_fixed_get_and_busy_is_explicit(self):
        class ExportLab:
            calls = 0
            def __init__(self, **_):
                pass
            def audience_export(self):
                self.calls += 1
                return {'exportVersion': 1}
        service = TrainingService('.', factory=ExportLab)
        self.assertEqual(service.dispatch('GET', '/api/training/audience-export'), {'exportVersion': 1})
        self.assertEqual(service.lab.calls, 1)
        for method, path, body, status in (
                ('POST', '/api/training/audience-export', {}, 404),
                ('GET', '/api/training/audience-export?path=anything', None, 400)):
            with self.assertRaises(TrainingApiError) as error:
                service.dispatch(method, path, body)
            self.assertEqual(error.exception.status, status)
        service.lock.acquire()
        try:
            with self.assertRaises(TrainingApiError) as error:
                service.dispatch('GET', '/api/training/audience-export')
            self.assertEqual((error.exception.status, error.exception.code), (409, 'training_busy'))
        finally:
            service.lock.release()
        self.assertEqual(service.lab.calls, 1)


if __name__ == '__main__':
    unittest.main()
