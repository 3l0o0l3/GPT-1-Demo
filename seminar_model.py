"""Frozen CPU inference for the seminar; no training data, optimizer or HTTP.

The small decoder is intentionally standalone so deployment does not import the
training service. Its state keys and forward math match the trained GPT model,
including its unused single classifier head for exact checkpoint compatibility.
"""
from __future__ import annotations

import copy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from tokenizers import Tokenizer


CONFIG = dict(layers=4, heads=4, width=128, context=128, trainContext=64)
SPECIALS = {'[PAD]': 0, '[BOS]': 1, '[EOS]': 2, '[CLS]': 3, '[UNK]': 4}
PAD, BOS, EOS, CLS, UNK = range(5)
CHECKPOINT_FIELDS = {'formatVersion', 'config', 'vocabSize', 'model', 'source',
                     'tokenizerSha256', 'dataManifestSha256', 'savedAt'}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


class _DecoderBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.width, self.heads = width, heads
        self.qkv = nn.Linear(width, width * 3)
        self.proj = nn.Linear(width, width)
        self.ln1, self.ln2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width))

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [part.view(batch, length, self.heads, width // self.heads).transpose(1, 2)
                   for part in (q, k, v)]
        attention = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        attention = attention.transpose(1, 2).contiguous().view(batch, length, width)
        x = self.ln1(x + self.proj(attention))
        return self.ln2(x + self.ff(x))


class _GPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token = nn.Embedding(vocab_size, CONFIG['width'])
        self.position = nn.Embedding(CONFIG['context'], CONFIG['width'])
        self.blocks = nn.ModuleList([_DecoderBlock(CONFIG['width'], CONFIG['heads'])
                                     for _ in range(CONFIG['layers'])])
        self.classifier = nn.Linear(CONFIG['width'], 2)

    def hidden(self, ids):
        positions = torch.arange(ids.shape[1], device=ids.device)
        h = self.token(ids) + self.position(positions)[None]
        for block in self.blocks:
            h = block(h)
        return h

    def forward(self, ids):
        return F.linear(self.hidden(ids), self.token.weight)


def validate_checkpoint(checkpoint):
    """Validate the exported, tensor-only schema without accepting training state."""
    if not isinstance(checkpoint, dict) or set(checkpoint) != CHECKPOINT_FIELDS:
        raise ValueError('Unsupported frozen checkpoint fields.')
    if type(checkpoint['formatVersion']) is not int or checkpoint['formatVersion'] != 1:
        raise ValueError('Unsupported frozen checkpoint version.')
    if checkpoint['config'] != CONFIG:
        raise ValueError('The frozen checkpoint architecture is incompatible.')
    if type(checkpoint['vocabSize']) is not int or not 5 <= checkpoint['vocabSize'] <= 8192:
        raise ValueError('Invalid vocabulary size.')
    source = checkpoint['source']
    if not isinstance(source, dict) or set(source) != {'step', 'seed', 'source'}:
        raise ValueError('Invalid checkpoint provenance.')
    if (type(source['step']) is not int or source['step'] < 0 or
            type(source['seed']) is not int or not 0 <= source['seed'] <= 2147483647 or
            source['source'] not in ('random', 'pretrained')):
        raise ValueError('Invalid checkpoint provenance.')
    if not all(_digest(checkpoint[name]) for name in ('tokenizerSha256', 'dataManifestSha256')):
        raise ValueError('Invalid checkpoint provenance hashes.')
    if not isinstance(checkpoint['savedAt'], str):
        raise ValueError('Invalid snapshot save timestamp.')
    try:
        saved_at = datetime.fromisoformat(checkpoint['savedAt'])
    except ValueError as error:
        raise ValueError('Invalid snapshot save timestamp.') from error
    if saved_at.tzinfo is None:
        raise ValueError('Snapshot save timestamp must include its timezone.')
    state = checkpoint['model']
    if not isinstance(state, dict) or not state:
        raise ValueError('Missing model weights.')
    if any(not isinstance(key, str) or not isinstance(value, torch.Tensor) or
           value.dtype != torch.float32 or value.device.type != 'cpu' or
           not torch.isfinite(value).all() for key, value in state.items()):
        raise ValueError('Model weights must be finite CPU float32 tensors.')
    # Constructor RNG is incidental. Loading a bundle must not alter caller RNG.
    with torch.random.fork_rng(devices=[]):
        model = _GPT(checkpoint['vocabSize'])
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError('Model tensor names or shapes are incompatible.') from error
    return model.eval().requires_grad_(False)


def validate_classifier(checkpoint, vocab_size, tokenizer_hash):
    """Only the two frozen transfer backbones/heads and their provenance."""
    fields = {'formatVersion', 'config', 'vocabSize', 'tokenizerSha256', 'dataManifestSha256',
              'savedAt', 'source', 'step', 'runId', 'branches'}
    if not isinstance(checkpoint, dict) or set(checkpoint) != fields:
        raise ValueError('Unsupported classifier checkpoint fields.')
    if (type(checkpoint['formatVersion']) is not int or checkpoint['formatVersion'] != 1
            or checkpoint['config'] != CONFIG or checkpoint['vocabSize'] != vocab_size
            or checkpoint['tokenizerSha256'] != tokenizer_hash
            or type(checkpoint['step']) is not int or checkpoint['step'] < 1
            or not isinstance(checkpoint['runId'], str) or len(checkpoint['runId']) != 32
            or any(c not in '0123456789abcdef' for c in checkpoint['runId'])
            or not isinstance(checkpoint['branches'], dict)
            or set(checkpoint['branches']) != {'pretrained', 'scratch'}):
        raise ValueError('Invalid trained classifier provenance.')
    source = checkpoint['source']
    if (not isinstance(source, dict) or set(source) != {'step', 'seed', 'source', 'modelSha256'}
            or not _digest(source['modelSha256'])):
        raise ValueError('Invalid classifier source.')
    models = {}
    for name, weights in checkpoint['branches'].items():
        # Reuse the strict model tensor/architecture/provenance contract.
        equivalent = {key: copy.deepcopy(checkpoint[key]) for key in
                      ('formatVersion', 'config', 'vocabSize', 'tokenizerSha256', 'dataManifestSha256', 'savedAt')}
        equivalent.update(model=weights, source={key: source[key] for key in ('step', 'seed', 'source')})
        models[name] = validate_checkpoint(equivalent)
    return models


class FrozenLanguageModel:
    """One immutable snapshot; caller serializes inference and owns chat history."""

    def __init__(self, bundle_dir, threads=1):
        if type(threads) is not int or not 1 <= threads <= 4:
            raise ValueError('CPU threads must be an integer from 1 to 4.')
        torch.set_num_threads(threads)
        directory = Path(bundle_dir)
        manifest_path = directory / 'manifest.json'
        if manifest_path.stat().st_size > 16384:
            raise ValueError('Frozen manifest is too large.')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if (not isinstance(manifest, dict) or set(manifest) != {'bundleVersion', 'snapshotId', 'files'} or
                type(manifest['bundleVersion']) is not int or manifest['bundleVersion'] not in (1, 2) or
                not isinstance(manifest['files'], dict) or
                set(manifest['files']) != ({'model.pt', 'tokenizer.json'} if manifest['bundleVersion'] == 1
                                           else {'model.pt', 'tokenizer.json', 'classifier.pt'})):
            raise ValueError('Invalid frozen bundle manifest.')
        for name in manifest['files']:
            maximum = {'model.pt': 32 * 1024 * 1024, 'tokenizer.json': 2 * 1024 * 1024,
                       'classifier.pt': 64 * 1024 * 1024}[name]
            path = directory / name
            if path.stat().st_size > maximum or not _digest(manifest['files'][name]):
                raise ValueError('Invalid frozen bundle artifact.')
            if file_sha256(path) != manifest['files'][name]:
                raise ValueError(f'Frozen bundle hash mismatch: {name}.')
        if manifest['snapshotId'] != manifest['files']['model.pt']:
            raise ValueError('Snapshot ID does not match model digest.')
        checkpoint = torch.load(directory / 'model.pt', map_location='cpu', weights_only=True)
        self._model = validate_checkpoint(checkpoint)
        if checkpoint['tokenizerSha256'] != manifest['files']['tokenizer.json']:
            raise ValueError('Checkpoint tokenizer provenance does not match bundle.')
        self._tokenizer = Tokenizer.from_file(str(directory / 'tokenizer.json'))
        if (self._tokenizer.get_vocab_size() != checkpoint['vocabSize'] or
                any(self._tokenizer.token_to_id(name) != expected for name, expected in SPECIALS.items())):
            raise ValueError('Tokenizer vocabulary or special tokens do not match the model.')
        source = checkpoint['source']
        self._metadata = {
            'snapshotId': manifest['snapshotId'], 'step': source['step'], 'seed': source['seed'],
            'source': source['source'], 'parameters': sum(p.numel() for p in self._model.parameters()),
            'trainContext': CONFIG['trainContext'], 'modelContext': CONFIG['context'],
            'vocabSize': checkpoint['vocabSize'], 'savedAt': checkpoint['savedAt'],
            'tokenizerSha256': checkpoint['tokenizerSha256'],
            'dataManifestSha256': checkpoint['dataManifestSha256'],
            'architecture': dict(CONFIG), 'device': 'cpu',
            'description': 'Frozen tiny story language model; not instruction- or chat-fine-tuned.',
            'classifier': {'available': False},
        }
        self._classifiers = {}
        if manifest['bundleVersion'] == 2:
            frozen = torch.load(directory / 'classifier.pt', map_location='cpu', weights_only=True)
            self._classifiers = validate_classifier(frozen, checkpoint['vocabSize'], checkpoint['tokenizerSha256'])
            self._metadata['classifier'] = {
                'available': True, 'snapshotId': manifest['files']['classifier.pt'],
                'step': frozen['step'], 'source': copy.deepcopy(frozen['source']),
                'runId': frozen['runId'], 'branches': ['pretrained', 'scratch'],
            }

    def metadata(self):
        return copy.deepcopy(self._metadata)

    @torch.inference_mode()
    def classify(self, text):
        if not isinstance(text, str) or not text.strip() or len(text) > 800:
            raise ValueError('Review must contain 1–800 characters.')
        if not self._classifiers:
            raise ValueError('No trained classifier has been published yet.')
        info = self._metadata['classifier']
        ids = [BOS] + self._tokenizer.encode(text).ids[:CONFIG['context'] - 2] + [CLS]
        inputs = torch.tensor([ids], dtype=torch.long)
        results = []
        for branch in ('pretrained', 'scratch'):
            model = self._classifiers[branch]
            probabilities = model.classifier(model.hidden(inputs)[:, -1]).softmax(-1)[0]
            if not torch.isfinite(probabilities).all():
                raise FloatingPointError('Classifier produced non-finite probabilities.')
            results.append({'branch': branch, 'prediction': ('negative', 'positive')[int(probabilities.argmax())],
                            'probabilities': [{'label': label, 'p': float(probabilities[i])}
                                              for i, label in enumerate(('negative', 'positive'))],
                            'step': info['step'], 'source': copy.deepcopy(info['source']), 'runId': info['runId']})
        return {'text': text, 'snapshotId': info['snapshotId'], 'branches': results}

    @torch.inference_mode()
    def generate(self, text, max_new_tokens=160, temperature=0.8, seed=42, on_event=None):
        if not isinstance(text, str) or not text.strip() or len(text) > 20000:
            raise ValueError('Prompt must contain 1–20000 characters.')
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 160:
            raise ValueError('Maximum new tokens must be an integer from 1 to 160.')
        if (isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or
                not math.isfinite(temperature) or not 0 <= temperature <= 1.5):
            raise ValueError('Temperature must be between 0 and 1.5.')
        if type(seed) is not int or not 0 <= seed <= 2147483647:
            raise ValueError('Seed must be an integer from 0 to 2147483647.')
        if on_event is not None and not callable(on_event):
            raise ValueError('on_event must be callable.')
        prompt = [BOS] + self._tokenizer.encode(text).ids
        source = {'model': 'frozen', 'step': self._metadata['step'], 'seed': self._metadata['seed'],
                  'source': self._metadata['source'], 'snapshotId': self._metadata['snapshotId']}
        sampling = {'temperature': temperature, 'seed': seed}
        context = {'mode': 'sliding', 'windowSize': 64, 'modelContext': 128, 'promptTokens': len(prompt)}
        generated, tokens = [], []
        rng = torch.Generator(device='cpu').manual_seed(seed)

        def emit(event):
            if on_event is not None:
                on_event(copy.deepcopy(event))

        emit({'type': 'start', 'text': text, 'source': source, 'sampling': sampling,
              'maxNewTokens': max_new_tokens, 'context': context})
        stop_reason = 'length'
        for _ in range(max_new_tokens):
            all_ids = prompt + generated
            input_ids = all_ids[-64:]
            window = {'inputTokens': len(input_ids), 'droppedTokens': len(all_ids) - len(input_ids)}
            logits = self._model(torch.tensor([input_ids], dtype=torch.long))[0, -1].clone()
            if not torch.isfinite(logits).all():
                raise FloatingPointError('Generation logits are non-finite.')
            logits[[PAD, BOS, CLS, UNK]] = float('-inf')
            if temperature == 0:
                token_id, probability = int(logits.argmax()), 1.0
            else:
                if temperature < torch.finfo(torch.float64).tiny:
                    distribution = logits.eq(logits.max()).double()
                    distribution /= distribution.sum()
                else:
                    distribution = ((logits.double() - logits.max().double()) / temperature).softmax(-1)
                if not torch.isfinite(distribution).all():
                    raise FloatingPointError('Generation probabilities are non-finite.')
                token_id = int(torch.multinomial(distribution, 1, generator=rng))
                probability = float(distribution[token_id])
            generated.append(token_id)
            tokens.append({'id': token_id, 'piece': self._tokenizer.id_to_token(token_id) or '[UNK]',
                           'p': probability})
            continuation = self._tokenizer.decode([item for item in generated if item != EOS])
            emit({'type': 'token', 'index': len(tokens) - 1, 'token': tokens[-1],
                  'continuation': continuation, 'window': window})
            if token_id == EOS:
                stop_reason = 'eos'
                break
        continuation_ids = [item for item in generated if item != EOS]
        result = {'text': text, 'continuation': self._tokenizer.decode(continuation_ids),
                  'fullText': self._tokenizer.decode(prompt[1:] + continuation_ids),
                  'tokens': tokens, 'stopReason': stop_reason, 'source': source,
                  'sampling': sampling, 'context': context, 'window': window}
        emit({'type': 'done', 'result': result})
        return result
