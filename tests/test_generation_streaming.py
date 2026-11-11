"""Real engine and localhost HTTP streaming checks; never touch a running trainer."""
import copy
import http.client
import json
import threading
import unittest
from unittest.mock import patch

import torch

import test_live_use as live_fixture
from training_engine import EOS, TrainingLab
from training_service import TrainingApiError, TrainingService
from remote_training_server import GPUTrainingHandler, GPUTrainingServer


class EngineStreamingTests(unittest.TestCase):
    setUp = live_fixture.LiveUseTests.setUp
    tearDown = live_fixture.LiveUseTests.tearDown
    assert_state_equal = live_fixture.LiveUseTests.assert_state_equal

    def test_stream_and_json_results_match_and_tokens_are_cumulative_decoding(self):
        lab = self.lab
        lab.train(1)
        lab.preview('train-0')
        before, pending = lab._capture(), copy.deepcopy(lab._preview)
        for model in ('live', 'random'):
            for temperature in (0, .8, 1e-300):
                with self.subTest(model=model, temperature=temperature):
                    arguments = dict(max_new_tokens=7, seed=23, model=model, temperature=temperature)
                    expected = lab.generate('the good', **arguments)
                    events = []
                    result = lab.generate('the good', on_event=events.append, **arguments)
                    self.assertEqual(expected, result)
                    self.assertEqual(events[0], {'type': 'start', 'text': 'the good',
                        'source': result['source'], 'sampling': result['sampling'], 'maxNewTokens': 7,
                        'context': result['context']})
                    self.assertEqual(events[-1], {'type': 'done', 'result': result})
                    tokens = events[1:-1]
                    self.assertEqual(len(tokens), len(result['tokens']))
                    for index, event in enumerate(tokens):
                        self.assertEqual((event['type'], event['index']), ('token', index))
                        self.assertEqual(event['token'], result['tokens'][index])
                        ids = [row['id'] for row in result['tokens'][:index + 1] if row['id'] != EOS]
                        self.assertEqual(event['continuation'], lab.tokenizer.decode(ids))
                    self.assert_state_equal(before, lab._capture())
                    self.assert_state_equal(pending, lab._preview)

    def test_token_callback_happens_before_the_next_forward_computation(self):
        lab = self.lab
        calls, events = [], []

        def fixed_forward(ids, *args, **kwargs):
            calls.append(ids.shape[1])
            logits = torch.zeros((*ids.shape, lab.vocab_size))
            logits[:, :, 5] = 100
            return logits, None, None

        def receive(event):
            if event['type'] == 'start':
                self.assertEqual(len(calls), 0)
            elif event['type'] == 'token':
                self.assertEqual(len(calls), event['index'] + 1)
                self.assertNotIn('done', [row['type'] for row in events])
            events.append(event)

        with patch.object(lab.model, 'forward', side_effect=fixed_forward):
            result = lab.generate('the', max_new_tokens=4, temperature=0, on_event=receive)
        self.assertEqual(len(calls), 4)
        self.assertEqual([row['type'] for row in events], ['start'] + ['token'] * 4 + ['done'])
        # A receiver owns event data, not the returned result or live state.
        events[-1]['result']['tokens'][0]['id'] = 9999
        self.assertNotEqual(result['tokens'][0]['id'], 9999)

    def test_callback_abort_restores_weights_adam_preview_modes_and_rng(self):
        lab = self.lab
        lab.train(1)
        lab.preview('train-0')
        lab.model.train()
        lab.model.blocks[0].eval()
        before, pending = lab._capture(), copy.deepcopy(lab._preview)
        modes = [module.training for module in lab.model.modules()]
        for model in ('live', 'random'):
            for abort_at in ('start', 'token', 'done'):
                events = []

                def disconnect(event):
                    events.append(event['type'])
                    torch.rand(4)  # Receiver RNG use must also remain isolated.
                    if event['type'] == abort_at:
                        raise BrokenPipeError('client disconnected')

                with self.subTest(model=model, abort_at=abort_at), self.assertRaises(BrokenPipeError):
                    lab.generate('the', model=model, max_new_tokens=3, on_event=disconnect)
                self.assertEqual(events[-1], abort_at)
                self.assert_state_equal(before, lab._capture())
                self.assert_state_equal(pending, lab._preview)
                self.assertEqual(modes, [module.training for module in lab.model.modules()])

    def test_prompt_error_emits_nothing_and_service_holds_then_releases_lock(self):
        lab = self.lab
        service = TrainingService(self.root, factory=lambda **_: lab)
        before = lab._capture()
        events = []
        with self.assertRaises(TrainingApiError) as caught:
            service.dispatch('POST', '/api/training/generate-stream',
                             {'text': ' '.join(['the'] * 127)}, on_event=events.append)
        self.assertEqual(caught.exception.status, 400)
        self.assertEqual(events, [])
        self.assertFalse(service.lock.locked())

        def disconnect(event):
            self.assertTrue(service.lock.locked())
            with self.assertRaises(TrainingApiError) as busy:
                service.dispatch('GET', '/api/training/state')
            self.assertEqual(busy.exception.code, 'training_busy')
            if event['type'] == 'token':
                raise BrokenPipeError('client disconnected')

        with self.assertRaises(BrokenPipeError):
            service.dispatch('POST', '/api/training/generate-stream', {'text': 'the'}, on_event=disconnect)
        self.assertFalse(service.lock.locked())
        self.assertEqual(service.dispatch('GET', '/api/training/state'), lab.state())
        self.assert_state_equal(before, lab._capture())

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is not available in this local test environment')
    def test_cuda_callback_abort_restores_cpu_and_cuda_rng(self):
        lab = TrainingLab(self.data, self.root / 'cuda-stream-artifacts', device='cuda')
        lab.train(1)
        before = lab._capture()

        def disconnect(event):
            torch.rand(4)
            torch.rand(4, device='cuda')
            if event['type'] == 'token':
                raise BrokenPipeError('client disconnected')

        for model in ('live', 'random'):
            with self.subTest(model=model), self.assertRaises(BrokenPipeError):
                lab.generate('the good', model=model, max_new_tokens=3, on_event=disconnect)
            self.assert_state_equal(before, lab._capture())


class ServiceStreamingValidationTests(unittest.TestCase):
    def test_invalid_stream_requests_reject_before_engine_access(self):
        def unavailable(**kwargs):
            self.fail('Invalid requests must not initialize the engine.')

        service = TrainingService('.', factory=unavailable)
        invalid = [None, [], {}, {'text': ' '}, {'text': 'x' * 8193}, {'text': 'the', 'path': '/x'},
                   {'text': 'the', 'maxNewTokens': True}, {'text': 'the', 'maxNewTokens': 1001},
                   {'text': 'the', 'temperature': -1}, {'text': 'the', 'temperature': float('nan')},
                   {'text': 'the', 'model': 'saved'}, {'text': 'the', 'seed': -1}]
        for body in invalid:
            with self.subTest(body=body), self.assertRaises(TrainingApiError) as caught:
                service.dispatch('POST', '/api/training/generate-stream', body, on_event=lambda _: None)
            self.assertEqual(caught.exception.status, 400)
            self.assertFalse(service.lock.locked())
        for method, url, status in (('GET', '/api/training/generate-stream', 404),
                                     ('POST', '/api/training/generate-stream?seed=1', 400)):
            with self.assertRaises(TrainingApiError) as caught:
                service.dispatch(method, url, {'text': 'the'}, on_event=lambda _: None)
            self.assertEqual(caught.exception.status, status)


class ControlledStreamingLab:
    """A synchronous provider controlled by events, not sleep-based timing."""
    def __init__(self):
        self.first_token = threading.Event()
        self.continue_generation = threading.Event()
        self.finished = threading.Event()

    def state(self):
        return {'step': 7}

    def generate(self, text, max_new_tokens=48, temperature=.8, seed=42, model='live', on_event=None,
                 sliding_window=False, window_size=128):
        if text == 'reject-prompt':
            raise ValueError('Prompt exceeds context.')
        source = {'model': model, 'step': 7, 'seed': 42, 'source': 'random'}
        sampling = {'temperature': temperature, 'seed': seed}
        tokens = [{'id': 5, 'piece': 'café', 'p': .5}, {'id': 6, 'piece': '😊', 'p': .25}]
        result = {'text': text, 'continuation': 'café 😊', 'fullText': text + ' café 😊',
                  'tokens': tokens, 'stopReason': 'length', 'source': source, 'sampling': sampling}
        if on_event is None:
            return result
        try:
            on_event({'type': 'start', 'text': text, 'source': source,
                      'sampling': sampling, 'maxNewTokens': max_new_tokens})
            on_event({'type': 'token', 'index': 0, 'token': tokens[0], 'continuation': 'café'})
            self.first_token.set()
            if text == 'wait' and not self.continue_generation.wait(3):
                raise RuntimeError('Test did not release generation.')
            if text == 'fail-after-start':
                raise RuntimeError('A private implementation error must not be exposed.')
            on_event({'type': 'token', 'index': 1, 'token': tokens[1], 'continuation': 'café 😊'})
            on_event({'type': 'done', 'result': result})
            return result
        finally:
            self.finished.set()


class QuietHandler(GPUTrainingHandler):
    def log_message(self, *args):
        pass

    def _stream_event(self, event):
        if self.server.disconnect_on_token and event['type'] == 'token':
            raise BrokenPipeError('Simulated socket write failure')
        super()._stream_event(event)


class HomeHttpStreamingTests(unittest.TestCase):
    def setUp(self):
        self.lab = ControlledStreamingLab()
        self.server = GPUTrainingServer(('127.0.0.1', 0), QuietHandler)
        self.server.training = TrainingService('.', factory=lambda **_: self.lab)
        self.server.health = {'ok': True}
        self.server.disconnect_on_token = False
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
        self.thread.start()
        self.connections = []

    def tearDown(self):
        self.lab.continue_generation.set()
        for connection in self.connections:
            connection.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(1)

    def request(self, method, path, body=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=2)
        self.connections.append(connection)
        headers = {'Content-Type': 'application/json'} if body is not None else {}
        connection.request(method, path, json.dumps(body) if body is not None else None, headers=headers)
        return connection.getresponse()

    def test_first_token_is_flushed_while_generation_is_still_blocked(self):
        response = self.request('POST', '/api/training/generate-stream', {'text': 'wait'})
        self.assertEqual(response.status, 200)
        self.assertEqual(response.getheader('Content-Type'), 'application/x-ndjson; charset=utf-8')
        self.assertEqual(response.getheader('Connection'), 'close')
        self.assertEqual(response.getheader('Cache-Control'), 'no-store')
        self.assertIsNone(response.getheader('Content-Length'))
        self.assertEqual(json.loads(response.readline())['type'], 'start')
        token = json.loads(response.readline())
        self.assertEqual((token['type'], token['index'], token['continuation']), ('token', 0, 'café'))
        self.assertTrue(self.lab.first_token.wait(1))
        self.assertFalse(self.lab.finished.is_set())
        busy = self.request('GET', '/api/training/state')
        self.assertEqual(busy.status, 409)
        self.assertEqual(json.loads(busy.read())['error']['code'], 'training_busy')
        self.lab.continue_generation.set()
        rest = [json.loads(line) for line in response.read().splitlines()]
        self.assertEqual([event['type'] for event in rest], ['token', 'done'])
        self.assertEqual(rest[0]['continuation'], 'café 😊')
        self.assertTrue(self.lab.finished.wait(1))
        self.assertFalse(self.server.training.lock.locked())

    def test_prestart_error_keeps_json_status_and_existing_json_generate(self):
        for body in ({'text': 'reject-prompt'}, {'text': 'the', 'maxNewTokens': 1001}):
            with self.subTest(body=body):
                response = self.request('POST', '/api/training/generate-stream', body)
                self.assertEqual(response.status, 400)
                self.assertEqual(response.getheader('Content-Type'), 'application/json; charset=utf-8')
                self.assertIn('error', json.loads(response.read()))
                self.assertFalse(self.server.training.lock.locked())
        response = self.request('POST', '/api/training/generate', {'text': 'the'})
        self.assertEqual(response.status, 200)
        self.assertEqual(response.getheader('Content-Type'), 'application/json; charset=utf-8')
        self.assertEqual(json.loads(response.read())['continuation'], 'café 😊')

    def test_poststart_error_is_one_ndjson_error_not_a_second_http_response(self):
        response = self.request('POST', '/api/training/generate-stream', {'text': 'fail-after-start'})
        self.assertEqual(response.status, 200)
        wire = response.read()
        self.assertNotIn(b'HTTP/', wire)
        events = [json.loads(line) for line in wire.splitlines()]
        self.assertEqual([event['type'] for event in events], ['start', 'token', 'error'])
        self.assertEqual(events[-1]['error']['code'], 'training_failed')
        self.assertNotIn('private', events[-1]['error']['message'])
        self.assertFalse(self.server.training.lock.locked())

    def test_disconnected_callback_aborts_and_releases_service_lock(self):
        self.server.disconnect_on_token = True
        response = self.request('POST', '/api/training/generate-stream', {'text': 'the'})
        events = [json.loads(line) for line in response.read().splitlines()]
        self.assertEqual([event['type'] for event in events], ['start'])
        self.assertTrue(self.lab.finished.wait(1))
        self.assertFalse(self.server.training.lock.locked())
        response = self.request('GET', '/api/training/state')
        self.assertEqual(response.status, 200)
        self.assertEqual(json.loads(response.read()), {'step': 7})


if __name__ == '__main__':
    unittest.main()
