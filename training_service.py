"""Validated, serialized adapter between the training engine and localhost HTTP."""
from pathlib import Path
import math
import threading
import uuid
from urllib.parse import urlsplit, parse_qs
from training_jobs import TrainingJobs, TrainingJobError


class TrainingApiError(Exception):
    def __init__(self, status, code, message):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def integer(value, minimum, maximum, name):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise TrainingApiError(400, 'invalid_input', f'{name} must be an integer from {minimum} to {maximum}.')
    return value


def boolean(value, name):
    if type(value) is not bool:
        raise TrainingApiError(400, 'invalid_input', f'{name} must be a boolean.')
    return value


def text_value(value, name='text', maximum=8192):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise TrainingApiError(400, 'invalid_input', f'{name} must contain 1–{maximum} characters.')
    return value


def choice(value, allowed, name):
    if value not in allowed or not isinstance(value, str):
        raise TrainingApiError(400, 'invalid_input', f'{name} must be one of: {", ".join(allowed)}.')
    return value


def learning_rate(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 1e-5 <= value <= .01:
        raise TrainingApiError(400, 'invalid_input', 'Learning rate must be between 0.00001 and 0.01.')
    return float(value)


def temperature(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1.5:
        raise TrainingApiError(400, 'invalid_input', 'temperature must be between 0 and 1.5.')
    return float(value)


def auxiliary_weight(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value not in (0, .5):
        raise TrainingApiError(400, 'invalid_input', 'auxiliaryWeight must be 0 or 0.5.')
    return float(value)


def uuid_value(value, name):
    if not isinstance(value, str) or len(value) != 36:
        raise TrainingApiError(400, 'invalid_input', f'{name} must be a UUID.')
    try:
        parsed = str(uuid.UUID(value))
    except (ValueError, AttributeError):
        raise TrainingApiError(400, 'invalid_input', f'{name} must be a UUID.') from None
    if parsed != value.lower():
        raise TrainingApiError(400, 'invalid_input', f'{name} must be a UUID.')
    return parsed


class TrainingService:
    def __init__(self, root, factory=None):
        self.root = Path(root)
        self.lock = threading.Lock()
        self.lab = None
        self.factory = factory
        self.jobs = TrainingJobs(self.lock, self._lab)

    def _lab(self):
        if self.lab is None:
            if self.factory is None:
                from training_engine import TrainingLab
                factory = TrainingLab
            else:
                factory = self.factory
            self.lab = factory(data_dir=self.root / 'data', artifact_dir=self.root / '.training', seed=42, device='cpu')
        return self.lab

    def dispatch(self, method, url, body=None, on_event=None):
        parts = urlsplit(url)
        action = parts.path.removeprefix('/api/training/')
        query = parse_qs(parts.query, keep_blank_values=True)
        if method == 'GET':
            if action not in ('meta', 'state', 'recordings', 'examples', 'transfer-state', 'job', 'token-history', 'audience-export'):
                raise TrainingApiError(404, 'not_found', 'Unknown training route.')
            if action != 'examples' and query:
                raise TrainingApiError(400, 'invalid_input', 'This route accepts no query parameters.')
            if any(len(values) != 1 for values in query.values()) or set(query) - {'kind','split','limit'}:
                raise TrainingApiError(400, 'invalid_input', 'Unsupported or repeated query parameters.')
            if action == 'examples':
                kind = choice(query.get('kind', ['lm'])[0], ('lm','sentiment'), 'kind')
                split = choice(query.get('split', ['train'])[0], ('train','validation'), 'split')
                try:
                    limit = int(query.get('limit', ['8'])[0])
                except ValueError as exc:
                    raise TrainingApiError(400, 'invalid_input', 'limit must be an integer.') from exc
                limit = integer(limit, 1, 20, 'limit')
        elif method == 'POST':
            allowed = {'inspect':{'text','position'}, 'preview':{'exampleId','learningRate'},
                       'commit':{'previewId'}, 'train':{'steps','learningRate'},
                       'reset':{'source','seed'}, 'predict':{'text','branch'},
                       'generate':{'text','maxNewTokens','temperature','seed','model','slidingWindow','windowSize'},
                       'generate-stream':{'text','maxNewTokens','temperature','seed','model','slidingWindow','windowSize'},
                       'transfer-start':{'seed','auxiliaryWeight'},
                       'transfer-train':{'steps','learningRate'},
                       'transfer-predict':{'text','branch'},
                       'job-start':{'kind','steps','learningRate','requestId'},
                       'job-pause':{'jobId'},
                       'save-live':set(), 'restore-live':set(),
                       'token-history-start':{'text'}, 'token-history-stop':set()}
            if action not in allowed:
                raise TrainingApiError(404, 'not_found', 'Unknown training route.')
            if not isinstance(body, dict) or set(body) - allowed[action] or query:
                raise TrainingApiError(400, 'invalid_input', 'Unsupported request fields.')
            if action in ('inspect','predict','generate','generate-stream','transfer-predict','token-history-start'):
                text = text_value(body.get('text'))
            if action == 'inspect':
                position = integer(body.get('position', 0), 0, 127, 'position')
            if action in ('preview','train','transfer-train'):
                rate = learning_rate(body.get('learningRate', .0003))
            if action == 'preview':
                example = body.get('exampleId')
                if example is not None:
                    example = text_value(example, 'exampleId', 200)
            if action in ('train','transfer-train'):
                steps = integer(body.get('steps', 10 if action == 'transfer-train' else 5), 1, 10, 'steps')
            if action == 'commit':
                preview_id = text_value(body.get('previewId'), 'previewId', 200)
            if action == 'reset':
                source = choice(body.get('source', 'random'), ('random','pretrained'), 'source')
                seed = integer(body.get('seed', 42), 0, 2147483647, 'seed')
            if action == 'predict':
                branch = choice(body.get('branch', 'pretrained'), ('scratch','pretrained','auxiliary'), 'branch')
            if action in ('generate','generate-stream'):
                max_new_tokens = integer(body.get('maxNewTokens', 48), 1, 1000, 'maxNewTokens')
                sliding_window = boolean(body.get('slidingWindow', False), 'slidingWindow')
                window_size = integer(body.get('windowSize', 128), 1, 128, 'windowSize')
                sampling_temperature = temperature(body.get('temperature', .8))
                seed = integer(body.get('seed', 42), 0, 2147483647, 'seed')
                model = choice(body.get('model', 'live'), ('live','random'), 'model')
            if action == 'transfer-start':
                seed = integer(body.get('seed', 42), 0, 2147483647, 'seed')
                weight = auxiliary_weight(body.get('auxiliaryWeight', 0))
            if action == 'transfer-predict':
                branch = choice(body.get('branch', 'pretrained'), ('pretrained','scratch'), 'branch')
            if action == 'job-start':
                kind = choice(body.get('kind'), ('lm','transfer'), 'kind')
                steps = integer(body.get('steps'), 1, 5000, 'steps')
                rate = learning_rate(body.get('learningRate'))
                request_id = uuid_value(body.get('requestId'), 'requestId')
            if action == 'job-pause':
                job_id = uuid_value(body.get('jobId'), 'jobId')
        else:
            raise TrainingApiError(405, 'method_not_allowed', 'Use GET or POST.')

        # Job control and published progress never wait for model computation.
        try:
            if action == 'job':
                return self.jobs.get()
            if action == 'job-start':
                return self.jobs.start(kind, steps, rate, request_id)
            if action == 'job-pause':
                return self.jobs.pause(job_id)
        except TrainingJobError as exc:
            raise TrainingApiError(exc.status, exc.code, exc.message) from None
        if method == 'GET' and action in ('state','transfer-state'):
            cached = self.jobs.cached_state(action)
            if cached is not None:
                return cached
        if method == 'GET' and action == 'token-history' and self.lab is not None:
            # A separate immutable CPU publication: no GPU/model lock and no
            # large token history in the frequent background-job snapshots.
            return self.lab.token_history()

        # These public reads depend only on immutable architecture/data files,
        # not on trainable weights. State reads above use the worker snapshots.
        static_read = (method == 'GET' and action in ('meta','recordings','examples')
                       and self.lab is not None and self.jobs.active())
        acquired = not static_read and self.lock.acquire(blocking=False)
        if not static_read and not acquired:
            raise TrainingApiError(409, 'training_busy', 'A training operation is running. Wait for this batch to finish.')
        try:
            lab = self._lab()
            if method == 'GET':
                if action == 'meta': return lab.metadata()
                if action == 'state': return lab.state()
                if action == 'recordings': return lab.recordings()
                if action == 'examples': return lab.examples(kind=kind, split=split, limit=limit)
                if action == 'transfer-state': return lab.transfer_state()
                if action == 'token-history': return lab.token_history()
                if action == 'audience-export': return lab.audience_export()
            if action == 'inspect': return lab.inspect(text, position=position)
            if action == 'preview': return lab.preview(example_id=example, learning_rate=rate)
            if action == 'commit': return lab.commit(preview_id)
            if action == 'train': return lab.train(steps=steps, learning_rate=rate)
            if action == 'reset': return lab.reset(source=source, seed=seed)
            if action == 'predict': return lab.predict(text, branch=branch)
            if action == 'generate':
                return lab.generate(text, max_new_tokens=max_new_tokens, temperature=sampling_temperature,
                                    seed=seed, model=model, sliding_window=sliding_window, window_size=window_size)
            if action == 'generate-stream':
                return lab.generate(text, max_new_tokens=max_new_tokens, temperature=sampling_temperature,
                                    seed=seed, model=model, on_event=on_event,
                                    sliding_window=sliding_window, window_size=window_size)
            if action == 'transfer-start': return lab.start_transfer(seed=seed, auxiliary_weight=weight)
            if action == 'transfer-train': return lab.train_transfer(steps=steps, learning_rate=rate)
            if action == 'transfer-predict': return lab.predict_transfer(text, branch=branch)
            if action == 'save-live': return lab.save_live()
            if action == 'restore-live': return lab.restore_live()
            if action == 'token-history-start': return lab.start_token_history(text)
            if action == 'token-history-stop': return lab.stop_token_history()
        except (ImportError, ModuleNotFoundError) as exc:
            raise TrainingApiError(503, 'training_not_ready', 'Training dependencies are unavailable. Run the training-enabled launcher.') from exc
        except FileNotFoundError as exc:
            raise TrainingApiError(404, 'artifact_unavailable', 'The requested training data or checkpoint is not available yet.') from exc
        except (ValueError, KeyError) as exc:
            status = 409 if action in ('commit','transfer-train','transfer-predict','restore-live') else 400
            raise TrainingApiError(status, 'invalid_training_state', str(exc)) from exc
        finally:
            if acquired:
                self.lock.release()
