"""Encode a bounded, weights-only audience bundle without changing training state.

The caller owns the training model lock. No optimizer, RNG, training text,
filesystem destination, or executable object crosses this boundary.
"""
from __future__ import annotations

import base64
import copy
from datetime import datetime, timezone
import hashlib
import io
import json

import torch


MAX_MODEL_BYTES = 12 * 1024 * 1024
MAX_TOKENIZER_BYTES = 2 * 1024 * 1024
MAX_REPLY_BYTES = 16 * 1024 * 1024


def encode_audience_export(state_dict, *, source, config, vocab_size,
                           tokenizer_bytes, data_manifest_bytes, saved_at=None):
    """Return exportVersion1 with two base64 files and their SHA256 manifest."""
    if (not isinstance(source, dict) or set(source) != {'step', 'seed', 'source'}
            or type(source['step']) is not int or source['step'] < 0
            or type(source['seed']) is not int or not 0 <= source['seed'] <= 2147483647
            or source['source'] not in ('random', 'pretrained')):
        raise ValueError('Invalid audience model source.')
    if type(vocab_size) is not int or not 5 <= vocab_size <= 8192:
        raise ValueError('Invalid audience vocabulary size.')
    if (not isinstance(tokenizer_bytes, bytes) or not 0 < len(tokenizer_bytes) <= MAX_TOKENIZER_BYTES
            or not isinstance(data_manifest_bytes, bytes) or not data_manifest_bytes):
        raise ValueError('Invalid audience tokenizer or data provenance.')
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError('Audience model weights are unavailable.')
    weights = {}
    for name, value in state_dict.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
            raise ValueError('Audience weights must be named FP32 tensors.')
        tensor = value.detach().to('cpu').clone().contiguous()
        if not torch.isfinite(tensor).all():
            raise ValueError('Audience weights must be finite.')
        weights[name] = tensor
    timestamp = datetime.now(timezone.utc).isoformat() if saved_at is None else saved_at
    if not isinstance(timestamp, str) or datetime.fromisoformat(timestamp).tzinfo is None:
        raise ValueError('Audience export requires a timestamp with a timezone.')
    digest = lambda raw: hashlib.sha256(raw).hexdigest()
    checkpoint = {
        'formatVersion': 1, 'config': copy.deepcopy(config), 'vocabSize': vocab_size,
        'model': weights, 'source': copy.deepcopy(source),
        'tokenizerSha256': digest(tokenizer_bytes),
        'dataManifestSha256': digest(data_manifest_bytes), 'savedAt': timestamp,
    }
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    model_bytes = buffer.getvalue()
    if len(model_bytes) > MAX_MODEL_BYTES:
        raise ValueError('Audience model export exceeds the size limit.')
    raw_files = {'model.pt': model_bytes, 'tokenizer.json': tokenizer_bytes}
    hashes = {name: digest(raw) for name, raw in raw_files.items()}
    result = {
        'exportVersion': 1,
        'manifest': {'bundleVersion': 1, 'snapshotId': hashes['model.pt'], 'files': hashes},
        'files': {name: base64.b64encode(raw).decode('ascii') for name, raw in raw_files.items()},
    }
    if len(json.dumps(result, separators=(',', ':')).encode('utf-8')) > MAX_REPLY_BYTES:
        raise ValueError('Audience export response exceeds the size limit.')
    return result
