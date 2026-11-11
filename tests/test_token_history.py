"""Real isolated model tests for live observations, persistence and concurrent reads."""
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch

from training_engine import TrainingLab, EOS
from training_service import TrainingService, TrainingApiError
from training_token_history import TokenHistory, validate_history
from test_training_engine import make_fixture, digest


class TokenHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / 'data').mkdir()
        make_fixture(self.root / 'data')
        self.lab = TrainingLab(self.root / 'data', self.root / 'artifacts')

    def tearDown(self):
        self.temp.cleanup()

    def test_probabilities_match_logits_and_every_position_inspection(self):
        lab = self.lab
        text = 'the good story is nice .'
        lab.train(2)
        lab.model.train()
        lab.model.blocks[0].eval()  # Preserve mixed training flags exactly.
        modes = [m.training for m in lab.model.modules()]
        state, python_rng = digest(lab._capture()), random.getstate()
        grads = [None if p.grad is None else p.grad.clone() for p in lab.model.parameters()]
        recording = lab.start_token_history(text)
        self.assertEqual(state, digest(lab._capture()))
        self.assertEqual(python_rng, random.getstate())
        self.assertEqual(modes, [m.training for m in lab.model.modules()])
        for before, p in zip(grads, lab.model.parameters()):
            self.assertTrue(before is None and p.grad is None or torch.equal(before, p.grad))
        frame = recording['frames'][0]
        self.assertEqual(frame['step'], 2)
        self.assertEqual(recording['targets'][-1]['id'], EOS)
        with torch.no_grad():
            ids, targets = lab._input(text)
            expected = lab.model(ids, return_attention=True)[0][0].softmax(-1)
        for index, (target, top) in enumerate(zip(recording['targets'], frame['topPredictions'])):
            self.assertAlmostEqual(frame['targetProbabilities'][index], float(expected[index, target['id']]), places=7)
            self.assertEqual(top['id'], int(expected[index].argmax()))
            self.assertAlmostEqual(top['p'], float(expected[index].max()), places=7)
            inspection = lab.inspect(text, index)
            self.assertAlmostEqual(inspection['targetProbability'], frame['targetProbabilities'][index], places=7)
            candidate_p, candidate_ids = expected[index].topk(8)
            self.assertEqual(frame['candidateIds'][index], candidate_ids.tolist())
            self.assertEqual(frame['candidateProbabilities'][index], candidate_p.tolist())
            for token_id, probability in zip(frame['candidateIds'][index], frame['candidateProbabilities'][index]):
                self.assertEqual(recording['candidatePieces'][str(token_id)], lab._piece(token_id))
                self.assertAlmostEqual(probability, next(item['p'] for item in inspection['probabilities'] if item['id'] == token_id), places=7)
        validate_history(recording, vocab_size=lab.vocab_size, current_step=2)

    def test_candidates_legacy_restore_stays_exact_and_later_frames_are_measured(self):
        lab = self.lab
        lab.start_token_history('the good story')
        lab.save_live()
        path = lab.artifact_dir / 'live' / 'session.pt'
        legacy = torch.load(path, weights_only=False)
        history = legacy['tokenHistory']
        del history['candidatePieces']
        for frame in history['frames']:
            del frame['candidateIds'], frame['candidateProbabilities']
        for active in (False, True):
            with self.subTest(active=active):
                history['active'] = active
                torch.save(legacy, path)
                lab.restore_live()
                self.assertEqual(history, lab.token_history())
                lab.save_live()
                self.assertEqual(history, torch.load(path, weights_only=False)['tokenHistory'])
                if not active:
                    self.assertEqual(history, lab.stop_token_history())
                    continue
                before = copy.deepcopy(history['frames'][0])
                lab.train(1)
                ended = lab.stop_token_history()
                self.assertEqual(ended['frames'][0], before)
                self.assertNotIn('candidateIds', ended['frames'][0])
                self.assertIn('candidateIds', ended['frames'][1])
                self.assertIn('candidatePieces', ended)
                validate_history(ended, vocab_size=lab.vocab_size, current_step=1)
                lab.save_live(); lab.restore_live()
                self.assertEqual(ended, lab.token_history())

    def test_malformed_candidates_fail_before_model_or_recording_changes(self):
        lab = self.lab
        lab.start_token_history('the good story')
        lab.save_live()
        path = lab.artifact_dir / 'live' / 'session.pt'
        saved = torch.load(path, weights_only=False)
        def reversed_rank(h):
            frame = h['frames'][0]
            frame['candidateIds'][0][1:3] = reversed(frame['candidateIds'][0][1:3])
            frame['candidateProbabilities'][0][1:3] = reversed(frame['candidateProbabilities'][0][1:3])
        def invalid_mass(h):
            frame = h['frames'][0]
            frame['candidateProbabilities'][0] = [.2] * 8
            frame['topPredictions'][0]['p'] = .2
            frame['targetProbabilities'][0] = .2 if h['targets'][0]['id'] in frame['candidateIds'][0] else .1
        changes = [
            lambda h: h['frames'][0].pop('candidateProbabilities'),
            lambda h: h['frames'][0]['candidateIds'].pop(),
            lambda h: h['frames'][0]['candidateIds'][0].pop(),
            lambda h: h['frames'][0]['candidateProbabilities'][0].__setitem__(0, float('nan')),
            lambda h: h['frames'][0]['candidateIds'][0].__setitem__(1, h['frames'][0]['candidateIds'][0][0]),
            lambda h: h['candidatePieces'].pop(str(h['frames'][0]['candidateIds'][0][0])),
            lambda h: h['candidatePieces'].__setitem__('01', 'bad-key'),
            lambda h: h['candidatePieces'].__setitem__(str(h['frames'][0]['candidateIds'][0][1]), 'wrong-piece'),
            reversed_rank, invalid_mass,
        ]
        for alter in changes:
            with self.subTest(alter=alter):
                bad = copy.deepcopy(saved); alter(bad['tokenHistory']); torch.save(bad, path)
                before, history = digest(lab._capture()), lab.token_history()
                with self.assertRaises(ValueError): lab.restore_live()
                self.assertEqual(before, digest(lab._capture()))
                self.assertEqual(history, lab.token_history())

    def test_full_candidate_history_for_fixed_vocab_fits_transport_limit(self):
        # Conservative serialization witness for the current 8192-token BPE:
        # its longest JSON-escaped token piece is26bytes, including quotes.
        from hashlib import sha256
        ids = list(range(8191, 8183, -1))
        pieces = {str(i): str(i).zfill(24) for i in range(8192)}
        text = 'a' * 8192
        frame = {'step': 0, 'targetProbabilities': [1.2345678901234567e-100] * 128,
                 'topPredictions': [{'id': ids[0], 'piece': pieces[str(ids[0])], 'p': .12345678901234568}] * 128,
                 'candidateIds': [ids] * 128, 'candidateProbabilities': [[.12345678901234568] * 8] * 128}
        value = {'version': 1, 'available': True, 'active': True,
                 'runId': '00000000-0000-0000-0000-000000000000', 'text': text,
                 'textSha256': sha256(text.encode()).hexdigest(), 'seed': 42, 'source': 'random',
                 'startStep': 0, 'intervalSteps': 100, 'capacity': 256, 'droppedFrames': 0,
                 'context': 128, 'totalTargets': 8193, 'truncated': True,
                 'tokens': [{'id': 0, 'piece': pieces['0']}] * 128,
                 'targets': [{'id': 0, 'piece': pieces['0']}] * 128,
                 'candidatePieces': pieces,
                 'frames': [dict(frame, step=i * 100) for i in range(256)]}
        validate_history(value, vocab_size=8192, current_step=25500)
        for ensure_ascii in (False, True):
            raw = json.dumps(value, ensure_ascii=ensure_ascii, allow_nan=False).encode('utf-8')
            self.assertLess(len(raw), 16 * 1024 * 1024)

    def test_real_training_captures_cadence_and_preview_only_captures_on_commit(self):
        lab = self.lab
        recording = lab.start_token_history('the good story is nice .')
        for _ in range(10):
            lab.train(10)
        history = lab.token_history()
        self.assertEqual([f['step'] for f in history['frames']], [0, 100])
        self.assertNotEqual(history['frames'][0]['targetProbabilities'], history['frames'][1]['targetProbabilities'])
        self.assertEqual(history['runId'], recording['runId'])
        before = copy.deepcopy(history)
        preview = lab.preview('train-0')
        self.assertEqual(before, lab.token_history())
        lab.commit(preview['previewId'])
        self.assertEqual([f['step'] for f in lab.token_history()['frames']], [0, 100, 101])
        lab.train(2)
        before_stop = digest(lab._capture())
        ended = lab.stop_token_history()
        self.assertEqual([f['step'] for f in ended['frames']], [0, 100, 101, 103])
        self.assertFalse(ended['active'])
        self.assertEqual(before_stop, digest(lab._capture()))
        self.assertEqual(lab.stop_token_history(), ended)

    def test_truncation_and_causal_prefix_are_real_not_filled_eos(self):
        lab = self.lab
        long = lab.start_token_history('the ' * 140)
        self.assertTrue(long['truncated'])
        self.assertEqual(long['totalTargets'], 141)
        self.assertEqual(len(long['tokens']), 128)
        self.assertNotIn(EOS, [t['id'] for t in long['targets']])
        a = lab.start_token_history('the good story is nice')['frames'][0]
        b = lab.start_token_history('the good story is dull')['frames'][0]
        self.assertEqual(a['topPredictions'][:4], b['topPredictions'][:4])
        self.assertEqual(a['targetProbabilities'][:4], b['targetProbabilities'][:4])

    def test_no_active_recording_adds_no_observation_and_transfer_does_not_append(self):
        lab = self.lab
        with patch.object(lab, '_token_history_observation', side_effect=AssertionError('Unrequested inference')):
            for _ in range(10): lab.train(10)
        lab.start_token_history('the good story')
        before = lab.token_history()
        lab.start_transfer(seed=42)
        lab.train_transfer(steps=2)
        self.assertEqual(before, lab.token_history())

    def test_save_restore_history_and_legacy_save_no_extra_field(self):
        lab = self.lab
        path = lab.artifact_dir / 'live' / 'session.pt'
        lab.save_live()
        legacy = torch.load(path, weights_only=False)
        self.assertNotIn('tokenHistory', legacy)
        initial = lab.start_token_history('the good story')
        lab.train(3)
        ended = lab.stop_token_history()
        self.assertEqual([f['step'] for f in ended['frames']], [0, 3])
        lab.save_live()
        exact = digest(lab._capture())
        lab.reset(seed=7)
        self.assertFalse(lab.token_history()['available'])
        lab.restore_live()
        self.assertEqual(digest(lab._capture()), exact)
        self.assertEqual(ended, lab.token_history())
        torch.save(legacy, path)
        lab.restore_live()
        self.assertFalse(lab.token_history()['available'])
        self.assertEqual(lab.step_count, 0)
        self.assertNotEqual(initial['runId'], lab.start_token_history('the good story')['runId'])

    def test_active_recording_resumes_after_restore_and_bad_history_is_atomic(self):
        lab = self.lab
        original = lab.start_token_history('the good story')
        lab.save_live()
        path = lab.artifact_dir / 'live' / 'session.pt'
        saved = torch.load(path, weights_only=False)
        lab.stop_token_history()
        lab.restore_live()
        self.assertEqual(lab.token_history(), original)
        cases = []
        for alter in (
            lambda h: h['frames'][0].update(step=1),
            lambda h: h['frames'][0]['targetProbabilities'].__setitem__(0, float('nan')),
            lambda h: h['targets'][0].update(id=lab.vocab_size),
            lambda h: h.update(text='another text'),
            lambda h: h.update(seed=7),
            lambda h: h['frames'][0]['topPredictions'][0].update(piece='mismatched token'),
        ):
            broken = copy.deepcopy(saved); alter(broken['tokenHistory']); cases.append(broken)
        for broken in cases:
            before, history = digest(lab._capture()), lab.token_history()
            torch.save(broken, path)
            with self.assertRaises(ValueError): lab.restore_live()
            self.assertEqual(before, digest(lab._capture()))
            self.assertEqual(history, lab.token_history())
        with self.assertRaises(FileNotFoundError): lab.reset('pretrained')
        self.assertEqual(original, lab.token_history())
        with self.assertRaises(ValueError): lab.start_token_history(' ')
        self.assertEqual(original, lab.token_history())

    def test_buffer_keeps_initial_and_latest_and_detaches_readers(self):
        lab = self.lab
        first = lab.start_token_history('the good story')
        recorder = TokenHistory()
        recorder.restore(first)
        retained_reader = recorder.snapshot()
        for step in range(1, 301):
            frame = copy.deepcopy(first['frames'][0]); frame['step'] = step
            recorder.append(frame)
        result = recorder.snapshot()
        self.assertEqual(len(result['frames']), 256)
        self.assertEqual(result['frames'][0]['step'], 0)
        self.assertEqual(result['frames'][1]['step'], 46)
        self.assertEqual(result['frames'][-1]['step'], 300)
        self.assertEqual(result['droppedFrames'], 45)
        self.assertEqual(retained_reader, first)
        result['frames'][-1]['targetProbabilities'][0] = -10
        self.assertGreaterEqual(recorder.snapshot()['frames'][-1]['targetProbabilities'][0], 0)
        same = copy.deepcopy(first['frames'][0]); same['step'] = 300
        recorder.append(same)
        self.assertEqual(len(recorder.snapshot()['frames']), 256)
        self.assertEqual(recorder.snapshot()['droppedFrames'], 45)
        validate_history(recorder.snapshot(), vocab_size=lab.vocab_size, current_step=300)

    def test_history_get_does_not_acquire_model_lock_or_touch_model(self):
        service = TrainingService(self.root, factory=lambda **kwargs: self.lab)
        service.dispatch('POST', '/api/training/token-history-start', {'text': 'the good story'})
        with service.lock:
            with patch.object(self.lab, '_token_history_observation', side_effect=AssertionError('GPU touched')):
                history = service.dispatch('GET', '/api/training/token-history')
                self.assertTrue(history['available'])
            for route in ('token-history-start', 'token-history-stop'):
                with self.assertRaises(TrainingApiError) as caught:
                    service.dispatch('POST', '/api/training/' + route, {'text': 'x'} if route.endswith('start') else {})
                self.assertEqual(caught.exception.status, 409)
        self.assertNotIn('tokenHistory', self.lab.state())
        self.assertNotIn('tokenHistory', service.jobs.get())

    def test_bad_http_contract_never_initializes_model(self):
        service = TrainingService(self.root, factory=lambda **kwargs: (_ for _ in ()).throw(AssertionError('initialized')))
        for method, route, body in (
            ('POST', 'token-history-start', {}), ('POST', 'token-history-start', {'text': ' '}),
            ('POST', 'token-history-start', {'text': 'a' * 8193}),
            ('POST', 'token-history-start', {'text': 'a', 'step': 0}),
            ('POST', 'token-history-stop', {'path': '/etc/passwd'}),
            ('POST', 'token-history', {}), ('GET', 'token-history-start', None),
            ('GET', 'token-history?step=1', None),
        ):
            with self.subTest(route=route, body=body):
                with self.assertRaises(TrainingApiError):
                    service.dispatch(method, '/api/training/' + route, body)
        self.assertIsNone(service.lab)


if __name__ == '__main__':
    unittest.main()
