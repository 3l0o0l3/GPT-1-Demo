"""Bounded, immutable publication of actual per-token training observations.

The engine is the sole writer. Readers only copy an already-published CPU tree;
they never need the model lock or wait for a GPU operation.
"""
from copy import deepcopy
from hashlib import sha256
import math
import uuid


CAPACITY = 256
INTERVAL_STEPS = 100
CONTEXT = 128
CANDIDATES = 8


def _empty():
    return {'version': 1, 'available': False, 'active': False, 'frames': []}


def _integer(value, minimum, maximum):
    return type(value) is int and minimum <= value <= maximum


def _probability(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def _token(value, vocab_size):
    return (isinstance(value, dict) and _integer(value.get('id'), 0, vocab_size - 1)
            and isinstance(value.get('piece'), str) and len(value['piece']) <= 8192)


def validate_history(value, *, vocab_size, current_step):
    """Validate an optional saved CPU tree before changing live model state."""
    if value is None:
        return _empty()
    if not isinstance(value, dict) or type(value.get('version')) is not int or value.get('version') != 1:
        raise ValueError('Invalid token probability history version.')
    if value.get('available') is False:
        if value != _empty():
            raise ValueError('Invalid empty token probability history.')
        return _empty()
    if value.get('available') is not True or type(value.get('active')) is not bool:
        raise ValueError('Invalid token probability recording state.')
    try:
        if str(uuid.UUID(value['runId'])) != value['runId']:
            raise ValueError()
        text = value['text']
        if not isinstance(text, str) or not text.strip() or len(text) > 8192:
            raise ValueError()
        if value['textSha256'] != sha256(text.encode('utf-8')).hexdigest():
            raise ValueError()
        if (not _integer(value['seed'], 0, 2147483647)
                or value['source'] not in ('random', 'pretrained')
                or not _integer(value['startStep'], 0, current_step)
                or not _integer(value['intervalSteps'], INTERVAL_STEPS, INTERVAL_STEPS)
                or not _integer(value['capacity'], CAPACITY, CAPACITY)
                or not _integer(value['context'], CONTEXT, CONTEXT)
                or not _integer(value['droppedFrames'], 0, 10**12)
                or not _integer(value['totalTargets'], 1, 32769)
                or type(value['truncated']) is not bool):
            raise ValueError()
        tokens, targets, frames = value['tokens'], value['targets'], value['frames']
        if (not isinstance(tokens, list) or not 1 <= len(tokens) <= CONTEXT
                or not isinstance(targets, list) or len(tokens) != len(targets)
                or len(tokens) != min(CONTEXT, value['totalTargets'])
                or value['truncated'] != (value['totalTargets'] > CONTEXT)
                or not all(_token(t, vocab_size) for t in tokens + targets)
                or not isinstance(frames, list) or not 1 <= len(frames) <= CAPACITY):
            raise ValueError()
        pieces = value.get('candidatePieces', {})
        if (not isinstance(pieces, dict) or len(pieces) > vocab_size
                or any(not isinstance(key, str) or not key.isascii() or not key.isdecimal()
                       or str(int(key)) != key or not 0 <= int(key) < vocab_size
                       or not isinstance(piece, str) or len(piece) > 8192
                       for key, piece in pieces.items())):
            raise ValueError()
        previous = -1
        for frame in frames:
            step = frame['step']
            probabilities, predictions = frame['targetProbabilities'], frame['topPredictions']
            if (not _integer(step, value['startStep'], current_step) or step <= previous
                    or not isinstance(probabilities, list) or len(probabilities) != len(tokens)
                    or not isinstance(predictions, list) or len(predictions) != len(tokens)
                    or not all(_probability(p) for p in probabilities)):
                raise ValueError()
            for target, probability, top in zip(targets, probabilities, predictions):
                if (not _token(top, vocab_size) or not _probability(top.get('p'))
                        or top['p'] + 1e-7 < probability
                        or (top['id'] == target['id'] and abs(top['p'] - probability) > 1e-7)):
                    raise ValueError()
            has_ids, has_probabilities = 'candidateIds' in frame, 'candidateProbabilities' in frame
            if has_ids != has_probabilities:
                raise ValueError()
            if has_ids:
                candidate_ids, candidate_p = frame['candidateIds'], frame['candidateProbabilities']
                if (not isinstance(candidate_ids, list) or len(candidate_ids) != len(tokens)
                        or not isinstance(candidate_p, list) or len(candidate_p) != len(tokens)):
                    raise ValueError()
                count = min(CANDIDATES, vocab_size)
                for ids, probs, target, target_p, top in zip(candidate_ids, candidate_p, targets, probabilities, predictions):
                    if (not isinstance(ids, list) or len(ids) != count
                            or not isinstance(probs, list) or len(probs) != count
                            or not all(_integer(i, 0, vocab_size - 1) and str(i) in pieces for i in ids)
                            or len(set(ids)) != count or not all(_probability(p) for p in probs)
                            or any(left < right for left, right in zip(probs, probs[1:]))
                            or math.fsum(probs) > 1 + 1e-6
                            or top['id'] != ids[0] or abs(top['p'] - probs[0]) > 1e-7
                            or top['piece'] != pieces[str(ids[0])]):
                        raise ValueError()
                    if target['id'] in ids:
                        if abs(target_p - probs[ids.index(target['id'])]) > 1e-7:
                            raise ValueError()
                    elif target_p > probs[-1] + 1e-7 or math.fsum(probs) + target_p > 1 + 1e-6:
                        raise ValueError()
            previous = step
        if frames[0]['step'] != value['startStep']:
            raise ValueError()
    except (KeyError, TypeError, AttributeError, ValueError) as error:
        raise ValueError('Invalid saved token probability history.') from error
    return deepcopy(value)


class TokenHistory:
    """Copy-on-write observation state, independent of model and transport."""
    def __init__(self):
        self._published = _empty()

    @property
    def active(self):
        return self._published['active']

    @property
    def text(self):
        return self._published.get('text')

    def snapshot(self):
        return deepcopy(self._published)

    def clear(self):
        self._published = _empty()

    def restore(self, validated):
        self._published = deepcopy(validated)

    def begin(self, *, text, seed, source, total_targets, tokens, targets, frame, candidate_pieces=None):
        value = {
            'version': 1, 'available': True, 'active': True,
            'runId': str(uuid.uuid4()), 'text': text,
            'textSha256': sha256(text.encode('utf-8')).hexdigest(),
            'seed': seed, 'source': source, 'startStep': frame['step'],
            'intervalSteps': INTERVAL_STEPS, 'capacity': CAPACITY,
            'droppedFrames': 0, 'context': CONTEXT, 'totalTargets': total_targets,
            'truncated': total_targets > CONTEXT,
            'tokens': tokens, 'targets': targets, 'frames': [frame],
        }
        if candidate_pieces is not None:
            value['candidatePieces'] = candidate_pieces
        self._published = deepcopy(value)

    def append(self, frame, candidate_pieces=None):
        if not self.active:
            return
        old = self._published
        if frame['step'] < old['frames'][-1]['step']:
            raise ValueError('A token recording cannot move backwards in training.')
        frames = list(old['frames'])
        if frames[-1]['step'] == frame['step']:
            frames[-1] = deepcopy(frame)
        else:
            frames.append(deepcopy(frame))
        removed = max(0, len(frames) - CAPACITY)
        if removed:
            frames = frames[:1] + frames[-(CAPACITY - 1):]
        value = dict(old, frames=frames, droppedFrames=old['droppedFrames'] + removed)
        if candidate_pieces is not None:
            value['candidatePieces'] = {**old.get('candidatePieces', {}), **candidate_pieces}
        self._published = value

    def stop(self):
        if self.active:
            self._published = dict(self._published, active=False)
