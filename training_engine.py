"""Real, scaled GPT-1-style training for the local learning lab.

This module owns model/optimizer/random state. It never uses the sentiment test
split, contacts the network, or invents a training measurement.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import uuid

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tokenizers import Tokenizer
from training_evaluation import (prepare_lm, prepare_classifier,
                                 evaluate_lm as evaluate_cached_lm,
                                 evaluate_classifier as evaluate_cached_classifier)
from training_transfer_execution import PreparedTransferObjective
from training_timing import TrainingTiming, NULL_SAMPLE
from training_token_history import TokenHistory, validate_history, CANDIDATES
from training_audience_export import encode_audience_export

CONFIG = dict(layers=4, heads=4, width=128, context=128, trainContext=64)
PAD, BOS, EOS, CLS, UNK = range(5)
FORMAT_VERSION = 1
LIVE_FORMAT_VERSION = 1
EVALUATION_INTERVAL_STEPS = 100


class DecoderBlock(nn.Module):
    def __init__(self, width=128, heads=4):
        super().__init__()
        self.width, self.heads = width, heads
        self.qkv = nn.Linear(width, width * 3)
        self.proj = nn.Linear(width, width)
        self.ln1, self.ln2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width))

    def forward(self, x, return_attention=False):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(batch, length, self.heads, width // self.heads).transpose(1, 2) for t in (q, k, v)]
        if return_attention:
            blocked = torch.ones(length, length, device=x.device, dtype=torch.bool).triu(1)
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(width // self.heads)
            weights = scores.masked_fill(blocked, float('-inf')).softmax(dim=-1)
            a = weights @ v
        else:
            weights = None
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        a = a.transpose(1, 2).contiguous().view(batch, length, width)
        x = self.ln1(x + self.proj(a))  # GPT-1-style post-normalization.
        return self.ln2(x + self.ff(x)), weights


class GPT(nn.Module):
    def __init__(self, vocab_size, config=None):
        super().__init__()
        self.config = dict(CONFIG if config is None else config)
        c = self.config
        self.token = nn.Embedding(vocab_size, c['width'])
        self.position = nn.Embedding(c['context'], c['width'])
        self.blocks = nn.ModuleList([DecoderBlock(c['width'], c['heads']) for _ in range(c['layers'])])
        self.classifier = nn.Linear(c['width'], 2)
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ids, return_attention=False, lm_head=True):
        if ids.shape[1] > self.config['context']:
            raise ValueError('Input exceeds the model context.')
        positions = torch.arange(ids.shape[1], device=ids.device)
        h = self.token(ids) + self.position(positions)[None]
        attn = None
        for i, block in enumerate(self.blocks):
            h, weights = block(h, return_attention and i == 0)
            if weights is not None:
                attn = weights
        logits = F.linear(h, self.token.weight) if lm_head else None  # Tied embeddings.
        return logits, h, attn


def sequence_loss(logits, targets):
    """Mean over valid tokens within each sequence, then mean over sequences."""
    losses = F.cross_entropy(logits.transpose(1, 2), targets, ignore_index=PAD, reduction='none')
    valid = targets.ne(PAD)
    return ((losses * valid).sum(1) / valid.sum(1).clamp_min(1)).mean()


def _read_jsonl(path):
    rows = []
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                row['ids'] = np.asarray(row['ids'], dtype=np.int64)
                rows.append(row)
    if not rows:
        raise ValueError(f'No examples in {Path(path).name}.')
    return rows


def _json_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def _atomic_save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(data, temp)
    temp.replace(path)


def _cpu_copy(value):
    """Detached snapshots must not alias CUDA Adam buffers or model storage."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_copy(item) for item in value)
    return copy.deepcopy(value)


@contextmanager
def _isolated_rng(device, state=None):
    # manual_seed seeds every CUDA device, so preserve all initialized generators.
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() or device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        if state is not None:
            torch.random.set_rng_state(state['torchRandom'].cpu())
            if devices and state.get('cudaRandom') is not None:
                torch.cuda.set_rng_state_all([item.cpu() for item in state['cudaRandom']])
        try:
            yield
        finally:
            if state is not None:
                state['torchRandom'] = torch.random.get_rng_state().clone()
                state['cudaRandom'] = torch.cuda.get_rng_state_all() if devices else None


def _model_sha256(model):
    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


AMP_MAX_ATTEMPTS = 8


def _new_grad_scaler(precision, saved=None):
    """Validate the additive scaler snapshot; legacy checkpoints omit it."""
    if saved is not None:
        keys = {'scale', 'growth_factor', 'backoff_factor', 'growth_interval', '_growth_tracker'}
        if not isinstance(saved, dict) or set(saved) != keys:
            raise ValueError('Invalid gradient-scaler checkpoint.')
        numeric = ('scale', 'growth_factor', 'backoff_factor')
        if any(isinstance(saved[key], bool) or not isinstance(saved[key], (float, int)) or
               not math.isfinite(saved[key]) for key in numeric):
            raise ValueError('Invalid gradient-scaler checkpoint.')
        if (saved['scale'] <= 0 or saved['growth_factor'] <= 1 or not 0 < saved['backoff_factor'] < 1 or
                type(saved['growth_interval']) is not int or saved['growth_interval'] < 1 or
                type(saved['_growth_tracker']) is not int or not 0 <= saved['_growth_tracker'] < saved['growth_interval']):
            raise ValueError('Invalid gradient-scaler checkpoint.')
    scaler = torch.amp.GradScaler('cuda') if precision == 'fp16' else None
    if scaler is not None and saved is not None:
        scaler.load_state_dict(copy.deepcopy(saved))
    return scaler, copy.deepcopy(saved)


def _scaler_snapshot(scaler, dormant=None):
    return copy.deepcopy(scaler.state_dict() if scaler is not None else dormant)


def _retry_rng(device):
    return (torch.random.get_rng_state(), torch.cuda.get_rng_state_all() if device.type == 'cuda' else None)


def _restore_retry_rng(saved):
    torch.random.set_rng_state(saved[0])
    if saved[1] is not None:
        torch.cuda.set_rng_state_all(saved[1])


def _clip_amp_gradients(model):
    parameters = list(model.parameters())
    norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=False))
    if not math.isfinite(norm) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in parameters):
        # A non-finite aggregate with individually finite gradients is not an
        # overflow recorded by GradScaler, so it must not reach optimizer.step.
        raise FloatingPointError('The gradient norm is non-finite; no optimizer update was applied.')
    return norm


class TrainingLab:
    def __init__(self, data_dir: Path, artifact_dir: Path, seed=42, device='cpu', compile_training=False,
                 training_precision='fp32', timing_enabled=True):
        if type(compile_training) is not bool:
            raise ValueError('compile_training must be a boolean.')
        self.compile_training = compile_training
        if training_precision not in ('fp32', 'fp16'):
            raise ValueError('training_precision must be fp32 or fp16.')
        self.training_precision = training_precision
        torch.set_num_threads(max(1, min(8, int(os.environ.get('GPT_LAB_THREADS', '4')))))
        self.data_dir, self.artifact_dir = Path(data_dir), Path(artifact_dir)
        self.device = torch.device(device)
        self._timing = TrainingTiming(self.device, enabled=timing_enabled)
        self._token_history = TokenHistory()
        if self.training_precision == 'fp16' and self.device.type != 'cuda':
            raise ValueError('FP16 AMP training requires CUDA.')
        self.manifest = json.loads((self.data_dir / 'manifest.json').read_text(encoding='utf-8'))
        self.tokenizer = Tokenizer.from_file(str(self.data_dir / 'tokenizer.json'))
        for token, expected in zip(('[PAD]', '[BOS]', '[EOS]', '[CLS]', '[UNK]'), range(5)):
            if self.tokenizer.token_to_id(token) != expected:
                raise ValueError(f'Tokenizer special token mismatch: {token}.')
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.stories_train = _read_jsonl(self.data_dir / 'stories-train.jsonl')
        self.stories_val = _read_jsonl(self.data_dir / 'stories-val.jsonl')
        sentiment = json.loads((self.data_dir / 'sentiment.json').read_text(encoding='utf-8'))
        # The test section is intentionally neither retained nor evaluated.
        self.sentiment_train = sentiment['train']
        self.sentiment_val = sentiment['validation']
        del sentiment
        self._story_index = {e['id']: e for e in self.stories_train}
        self._eval_train = self._fixed_subset(self.stories_train, 32)
        self._eval_val = self._fixed_subset(self.stories_val, 32)
        # Fixed evaluation data only: runtime-owned tensors, never checkpoint
        # state. Batch shapes/order and FP32 evaluation math remain unchanged.
        self._lm_eval_batches = {
            'train': prepare_lm(self._eval_train, self._batch),
            'validation': prepare_lm(self._eval_val, self._batch),
        }
        self._sentiment_eval_batches = {
            'train': prepare_classifier(self._fixed_subset(self.sentiment_train, 128),
                                        lambda rows: sentiment_batch(rows, self.device)),
            'validation': prepare_classifier(self.sentiment_val,
                                             lambda rows: sentiment_batch(rows, self.device)),
        }
        self._preview = None
        self._predict_cache = {}
        self._transfer = None
        self._transfer_execution = {}
        self._initialize(seed)

    @staticmethod
    def _fixed_subset(rows, limit):
        indexes = np.linspace(0, len(rows) - 1, min(limit, len(rows)), dtype=int)
        return [rows[int(i)] for i in indexes]

    def _initialize(self, seed):
        self._invalidate_training_execution()
        self.seed = int(seed)
        torch.manual_seed(self.seed)
        self.random = random.Random(self.seed)
        self.model = GPT(self.vocab_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0003, weight_decay=0.0)
        self._grad_scaler, self._dormant_grad_scaler = _new_grad_scaler(self.training_precision)
        self.step_count, self.source = 0, 'random'
        self.history, self.last_trace = [], None
        self._preview = None
        self._measure()
        self._timing.reset('lm')

    def _invalidate_training_execution(self):
        # Runtime-only callable: the original model/Adam remain checkpoint owners.
        self._compiled_training_model = None
        self._compiled_training_call = None
        self._compile_warmup_seconds = None

    def _execution_metadata(self):
        ready = (self.compile_training and self._compiled_training_model is self.model
                 and self._compiled_training_call is not None)
        transfer_enabled = self.compile_training and self.device.type == 'cuda' and self.training_precision == 'fp16'
        transfer_pair = getattr(self, '_transfer', None)
        bindings = getattr(self, '_transfer_execution', {})
        transfer_ready = bool(transfer_enabled and transfer_pair and set(bindings) == {'pretrained', 'scratch'}
                              and all(binding.model is transfer_pair['branches'][name]['model']
                                      and binding.metadata()['ready'] for name, binding in bindings.items()))
        precision = 'FP16 AMP' if self.training_precision == 'fp16' else 'FP32'
        return {'trainingPrecision': precision, 'parameterPrecision': 'FP32',
                'optimizerStatePrecision': 'FP32', 'lossScaling': 'dynamic' if self.training_precision == 'fp16' else 'none',
                'optimizerStepAccounting': 'successful-updates-only',
                'maxAttemptsPerUpdate': AMP_MAX_ATTEMPTS if self.training_precision == 'fp16' else 1,
                'evaluationPrecision': 'FP32', 'previewPrecision': 'FP32', 'generationPrecision': 'FP32',
                'transferPrecision': precision,
                'optimizer': 'Adam', 'optimizerImplementation': 'default',
                'lmBackend': 'compiled' if ready else 'eager',
                'compileEnabled': self.compile_training, 'compileReady': bool(ready),
                'compiledShape': [8, CONFIG['trainContext']],
                'warmupSeconds': self._compile_warmup_seconds if ready else None,
                'evaluationBackend': 'eager', 'previewBackend': 'eager',
                'evaluationIntervalSteps': EVALUATION_INTERVAL_STEPS,
                'transferBackend': 'compiled-dynamic' if transfer_ready else 'eager',
                'transferCompileEnabled': bool(transfer_enabled), 'transferCompileReady': transfer_ready,
                'transferDynamicLength': transfer_ready,
                'transferWarmupSeconds': sum(binding.metadata()['warmupSeconds'] for binding in bindings.values())
                                        if transfer_ready else None}

    def prepare_training_execution(self):
        """Prepare only the CUDA 8x64 LM callable without advancing training.

        Other shapes, inspection, generation, evaluation, transfer and the
        one-update trace remain eager. This method requires exclusive model
        access, is idempotent for the current model identity, and fails explicitly
        when compilation is enabled but unavailable. Runtime caches are never saved.
        """
        if not self.compile_training:
            return self._execution_metadata()
        if self.device.type != 'cuda':
            raise RuntimeError('Compiled training requires CUDA; disable compilation for a CPU lab.')
        if self._execution_metadata()['compileReady']:
            return self._execution_metadata()
        self._invalidate_training_execution()
        original_model = self.model
        modes = [(module, module.training) for module in original_model.modules()]
        gradients = [(parameter, parameter.grad,
                      parameter.grad.detach().clone() if parameter.grad is not None else None)
                     for parameter in original_model.parameters()]
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        sampler_rng = self.random.getstate()
        started = time.perf_counter()
        try:
            with _isolated_rng(self.device):
                candidate = torch.compile(original_model, mode='reduce-overhead', dynamic=False)
                original_model.train()
                # Deterministic valid vocabulary indices; no training data or RNG
                # consumption. Three passes prepare forward/backward CUDA graphs.
                x = torch.arange(8 * CONFIG['trainContext'], device=self.device, dtype=torch.long)
                x = x.reshape(8, CONFIG['trainContext']).remainder(self.vocab_size)
                y = x.roll(-1, dims=1)
                for _ in range(3):
                    for parameter in original_model.parameters():
                        parameter.grad = None
                    torch.compiler.cudagraph_mark_step_begin()
                    with torch.autocast('cuda', dtype=torch.float16, enabled=self.training_precision == 'fp16'):
                        logits, _, _ = candidate(x)
                        loss = sequence_loss(logits, y)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Non-finite compilation warmup loss.')
                    # Unscaled warmup prepares the same AMP graph without changing
                    # the scaler's scale, growth tracker or per-optimizer state.
                    loss.backward()
                    # Finite checks are read-only; no clipping or optimizer update.
                    gradients_finite = all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                                           for parameter in original_model.parameters())
                    if not gradients_finite:
                        raise FloatingPointError('Non-finite compilation warmup gradient.')
                torch.cuda.synchronize(self.device)
            self._compiled_training_model = original_model
            self._compiled_training_call = candidate
            self._compile_warmup_seconds = time.perf_counter() - started
        except Exception as error:
            self._invalidate_training_execution()
            raise RuntimeError('Training compilation failed; the model was not advanced. Compilation must be repaired before compiled training.') from error
        finally:
            for parameter, original, saved in gradients:
                if original is not None:
                    original.copy_(saved)
                parameter.grad = original
            for module, training in modes:
                module.training = training
            self.random.setstate(sampler_rng)
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
        return self._execution_metadata()

    def metadata(self):
        return {
            'architecture': {**CONFIG, 'parameters': sum(p.numel() for p in self.model.parameters()),
                             'vocabSize': self.vocab_size, 'normalization': 'post-LN',
                             'activation': 'GELU', 'positions': 'learned', 'tiedLmHead': True,
                             'dropout': 0, 'batchSize': 8, 'optimizer': 'Adam',
                             'weightDecay': 0, 'gradientClipNorm': 1, 'cpuThreads': torch.get_num_threads()},
            'data': self.manifest, 'device': str(self.device), 'seed': self.seed,
            'execution': self._execution_metadata(),
            'description': 'Scaled GPT-1-style method experiment; not the original GPT-1 checkpoint or benchmark reproduction. Real PyTorch training on public text. CPU training windows contain up to 64 input tokens; the model and inspection support 128. Sentiment test data are never used.',
        }

    def examples(self, kind='lm', split='train', limit=8):
        if kind not in ('lm', 'sentiment') or split not in ('train', 'validation'):
            raise ValueError('Use kind lm/sentiment and split train/validation. Test data are not exposed.')
        if not isinstance(limit, int) or not 1 <= limit <= 50:
            raise ValueError('Example limit must be between 1 and 50.')
        rows = (self.stories_train if split == 'train' else self.stories_val) if kind == 'lm' else (self.sentiment_train if split == 'train' else self.sentiment_val)
        return {'examples': [{k: e[k] for k in ('id', 'text', 'label', 'source') if k in e} for e in rows[:limit]]}

    def _piece(self, token_id):
        return self.tokenizer.id_to_token(int(token_id)) or '[UNK]'

    def _input(self, text):
        if not isinstance(text, str) or not text.strip() or len(text) > 8192:
            raise ValueError('Text must contain 1–8192 characters.')
        ids = [BOS] + self.tokenizer.encode(text).ids + [EOS]
        ids = ids[:CONFIG['context'] + 1]
        return torch.tensor([ids[:-1]], device=self.device), torch.tensor([ids[1:]], device=self.device)

    @torch.no_grad()
    def inspect(self, text: str, position: int = 0):
        self.model.eval()
        ids, targets = self._input(text)
        position = min(max(0, int(position)), ids.shape[1] - 1)
        first_block = self.model.blocks[0]
        captured, hooks = {}, []

        def capture_qkv(module, inputs, output):
            captured['qkv'] = output.detach()

        def capture_context(module, inputs):
            captured['context'] = inputs[0].detach()

        # Inspection-only observers: never leave hooks attached for training or
        # compilation, including when a forward or hook registration fails.
        try:
            hooks.append(first_block.qkv.register_forward_hook(capture_qkv))
            hooks.append(first_block.proj.register_forward_pre_hook(capture_context))
            logits, _, attention = self.model(ids, return_attention=True)
        finally:
            for hook in hooks:
                hook.remove()
        probabilities = logits[0, position].softmax(-1)
        target_id = int(targets[0, position])
        count = min(8, self.vocab_size)
        indexes = probabilities.topk(count).indices.tolist()
        if target_id not in indexes:
            indexes.append(target_id)
        length = ids.shape[1]
        head_dimension = first_block.width // first_block.heads
        query, keys, values = [tensor.view(1, length, first_block.heads, head_dimension)[0, :, 0]
                               for tensor in captured['qkv'].chunk(3, dim=-1)]
        scale = math.sqrt(head_dimension)
        scores = (query[position] @ keys.transpose(0, 1)) / scale
        context = captured['context'][0, position, :head_dimension]
        # One additional host transfer for all new vector details, not one per
        # token or component. Existing attention values remain the authority.
        packed = torch.cat((query[position], keys.flatten(), values.flatten(), scores, context)).cpu().tolist()
        cursor = head_dimension
        key_rows = [packed[cursor + row * head_dimension:cursor + (row + 1) * head_dimension]
                    for row in range(length)]
        cursor += length * head_dimension
        value_rows = [packed[cursor + row * head_dimension:cursor + (row + 1) * head_dimension]
                      for row in range(length)]
        cursor += length * head_dimension
        score_values = [packed[cursor + key] if key <= position else None for key in range(length)]
        cursor += length
        attention_values = attention[0, 0].cpu().tolist()
        details = {'schemaVersion': 1, 'layer': 1, 'head': 1,
                   'headDimension': head_dimension, 'queryPosition': position, 'scale': scale,
                   'query': packed[:head_dimension], 'keys': key_rows, 'values': value_rows,
                   'scores': score_values, 'weights': list(attention_values[position]),
                   'context': packed[cursor:cursor + head_dimension]}
        return {
            'text': text, 'tokens': [{'id': int(i), 'piece': self._piece(i)} for i in ids[0]],
            'targets': [{'id': int(i), 'piece': self._piece(i)} for i in targets[0]],
            'position': position, 'target': {'id': target_id, 'piece': self._piece(target_id)},
            'probabilities': [{'id': i, 'piece': self._piece(i), 'p': float(probabilities[i])} for i in indexes],
            'targetProbability': float(probabilities[target_id]),
            'tokenLoss': float(F.cross_entropy(logits[:, position], targets[:, position])),
            'batchLoss': float(sequence_loss(logits, targets)),
            'attention': attention_values,
            'attentionDetails': details,
            'mask': torch.ones(length, length, dtype=torch.bool).triu(1).tolist(),
            'maskConvention': 'true means blocked future position; false means visible',
            'tokenPieceConvention': 'Raw byte-level BPE pieces; Ġ represents a preceding space.',
        }

    def token_history(self):
        """CPU-only published observation read, safe during background training."""
        return self._token_history.snapshot()

    @torch.no_grad()
    def _token_history_observation(self, text):
        ids, targets = self._input(text)
        modes = [(module, module.training) for module in self.model.modules()]
        try:
            self.model.eval()
            with _isolated_rng(self.device):
                # Match inspect's eager FP32 causal attention path, independent
                # of the FP16 training graph and without changing any gradients.
                logits, _, _ = self.model(ids, return_attention=True)
                probabilities = logits[0].float().softmax(-1)
                target_p = probabilities.gather(1, targets[0, :, None]).squeeze(1)
                count = min(CANDIDATES, self.vocab_size)
                top_p, top_ids = probabilities.topk(count, dim=-1)
                # One compact GPU-to-CPU transfer, not one read per token.
                measured = torch.cat((target_p[:, None], top_p, top_ids.float()), dim=1).cpu().tolist()
        finally:
            for module, mode in modes:
                module.training = mode
        candidate_ids = [[int(i) for i in row[1 + count:]] for row in measured]
        pieces = {str(token_id): self._piece(token_id) for token_id in {i for row in candidate_ids for i in row}}
        frame = {'step': self.step_count,
                 'targetProbabilities': [row[0] for row in measured],
                 'topPredictions': [{'id': choices[0], 'piece': pieces[str(choices[0])], 'p': row[1]}
                                    for row, choices in zip(measured, candidate_ids)],
                 'candidateIds': candidate_ids,
                 'candidateProbabilities': [row[1:1 + count] for row in measured]}
        return ids, targets, frame, pieces

    def start_token_history(self, text):
        ids, targets, frame, pieces = self._token_history_observation(text)
        self._token_history.begin(
            text=text, seed=self.seed, source=self.source,
            total_targets=len(self.tokenizer.encode(text).ids) + 1,
            tokens=[{'id': i, 'piece': self._piece(i)} for i in ids[0].tolist()],
            targets=[{'id': i, 'piece': self._piece(i)} for i in targets[0].tolist()], frame=frame,
            candidate_pieces=pieces)
        return self.token_history()

    def _record_token_history(self):
        if self._token_history.active:
            _, _, frame, pieces = self._token_history_observation(self._token_history.text)
            self._token_history.append(frame, candidate_pieces=pieces)

    def stop_token_history(self):
        self._record_token_history()
        self._token_history.stop()
        return self.token_history()

    def _validated_token_history(self, value, saved_lm):
        history = validate_history(value, vocab_size=self.vocab_size, current_step=saved_lm['step'])
        if history['available']:
            ids, targets = self._input(history['text'])
            expected_tokens = [{'id': i, 'piece': self._piece(i)} for i in ids[0].tolist()]
            expected_targets = [{'id': i, 'piece': self._piece(i)} for i in targets[0].tolist()]
            if (history['tokens'] != expected_tokens or history['targets'] != expected_targets
                    or history['totalTargets'] != len(self.tokenizer.encode(history['text']).ids) + 1
                    or history['seed'] != saved_lm['seed'] or history['source'] != saved_lm['source']):
                raise ValueError('Saved token history does not match its model and tokenizer.')
            pieces = {(item['id'], item['piece']) for frame in history['frames'] for item in frame['topPredictions']}
            if any(self._piece(token_id) != piece for token_id, piece in pieces):
                raise ValueError('Saved token history contains an incompatible token.')
            if any(self._piece(int(token_id)) != piece for token_id, piece in history.get('candidatePieces', {}).items()):
                raise ValueError('Saved candidate probabilities contain an incompatible token.')
        return history

    def _batch(self, rows, indices=None, random_windows=False, context=None):
        context = context or CONFIG['trainContext']
        if indices is not None:
            rows = [rows[i] for i in indices]
        pairs = []
        for row in rows:
            seq = [BOS] + list(map(int, row['ids'])) + [EOS]
            start = self.random.randrange(max(1, len(seq) - context)) if random_windows else 0
            seq = seq[start:start + context + 1]
            pairs.append((seq[:-1], seq[1:]))
        size = max(len(x) for x, _ in pairs)
        # Pack on CPU once, then transfer two contiguous batches. Creating and
        # assigning a CUDA tensor per row adds small copies and stream waits.
        x = np.full((len(pairs), size), PAD, dtype=np.int64)
        y = np.full_like(x, PAD)
        for i, (inputs, targets) in enumerate(pairs):
            x[i, :len(inputs)] = inputs
            y[i, :len(targets)] = targets
        return torch.from_numpy(x).to(self.device), torch.from_numpy(y).to(self.device)

    @torch.no_grad()
    def _evaluate_lm(self, rows):
        self.model.eval()
        total = 0.0
        for offset in range(0, len(rows), 8):
            part = rows[offset:offset + 8]
            x, y = self._batch(part)
            logits, _, _ = self.model(x)
            total += float(sequence_loss(logits, y)) * len(part)
        return total / len(rows)

    def _measure(self, timing=NULL_SAMPLE):
        with timing.phase('train_evaluation'):
            train_loss = evaluate_cached_lm(self.model, self._lm_eval_batches['train'], sequence_loss)
        with timing.phase('validation'):
            val_loss = evaluate_cached_lm(self.model, self._lm_eval_batches['validation'], sequence_loss)
        self.metrics = {
            'trainLoss': train_loss,
            'valLoss': val_loss,
            'evalDescription': f'Fixed evenly spaced subsets: {len(self._eval_train)} training documents and {len(self._eval_val)} held-out validation documents; first {CONFIG["trainContext"]} input tokens of each. Mean token NLL within each sequence, then sequence mean. No test data.',
        }
        item = {'step': self.step_count, 'trainLoss': self.metrics['trainLoss'], 'valLoss': self.metrics['valLoss']}
        if self.history and self.history[-1]['step'] == self.step_count:
            self.history[-1] = item
        else:
            self.history.append(item)

    def audience_export(self):
        """Read the current LM into an isolated audience bundle; caller holds the model lock."""
        return encode_audience_export(self.model.state_dict(),
            source={'step': self.step_count, 'seed': self.seed, 'source': self.source},
            config=CONFIG, vocab_size=self.vocab_size,
            tokenizer_bytes=(self.data_dir / 'tokenizer.json').read_bytes(),
            data_manifest_bytes=(self.data_dir / 'manifest.json').read_bytes())

    def state(self):
        self._timing.drain()
        result = {'step': self.step_count, 'mode': 'live', 'source': self.source,
                  'metrics': copy.deepcopy(self.metrics), 'history': copy.deepcopy(self.history),
                  'lastMeasuredStep': self.history[-1]['step'] if self.history else None,
                  'evaluationIntervalSteps': EVALUATION_INTERVAL_STEPS,
                  'timing': self._timing.published_snapshot('lm')}
        if self.last_trace:
            result['lastTrace'] = copy.deepcopy(self.last_trace)
        if self._preview:
            result['previewId'] = self._preview['id']
        return result

    def _capture(self):
        return {
            'formatVersion': FORMAT_VERSION, 'config': CONFIG, 'vocabSize': self.vocab_size,
            'model': {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()},
            'optimizer': _cpu_copy(self.optimizer.state_dict()),
            'gradScaler': _scaler_snapshot(self._grad_scaler, self._dormant_grad_scaler),
            'random': self.random.getstate(), 'torchRandom': torch.random.get_rng_state().clone(),
            'cudaRandom': torch.cuda.get_rng_state_all() if self.device.type == 'cuda' else None,
            'step': self.step_count, 'seed': self.seed, 'source': self.source,
            'history': copy.deepcopy(self.history), 'metrics': copy.deepcopy(self.metrics),
            'lastTrace': copy.deepcopy(self.last_trace),
            'dataManifestSha256': hashlib.sha256((self.data_dir / 'manifest.json').read_bytes()).hexdigest(),
        }

    def _restore(self, state):
        if state.get('formatVersion') != FORMAT_VERSION or state.get('config') != CONFIG or state.get('vocabSize') != self.vocab_size:
            raise ValueError('Incompatible model checkpoint.')
        expected_manifest = hashlib.sha256((self.data_dir / 'manifest.json').read_bytes()).hexdigest()
        if state.get('dataManifestSha256') != expected_manifest:
            raise ValueError('Checkpoint data/tokenizer provenance does not match the current manifest.')
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self._grad_scaler, self._dormant_grad_scaler = _new_grad_scaler(self.training_precision, state.get('gradScaler'))
        self.random.setstate(state['random'])
        torch.random.set_rng_state(state['torchRandom'].cpu())
        if state.get('cudaRandom') is not None and self.device.type == 'cuda':
            torch.cuda.set_rng_state_all([item.cpu() for item in state['cudaRandom']])
        self.step_count, self.seed, self.source = state['step'], state['seed'], state['source']
        self.history, self.metrics = copy.deepcopy(state['history']), copy.deepcopy(state['metrics'])
        self.last_trace = copy.deepcopy(state['lastTrace'])
        self.model.eval()

    @staticmethod
    def _rate(value):
        value = float(value)
        if not math.isfinite(value) or not 1e-5 <= value <= 0.01:
            raise ValueError('Learning rate must be between 0.00001 and 0.01.')
        return value

    def _update(self, x, y, learning_rate, trace=False, collect_metrics=True, timing=NULL_SAMPLE):
        with timing.phase('setup'):
            self.model.train()
            for group in self.optimizer.param_groups:
                group['lr'] = learning_rate
        if self.training_precision == 'fp16' and not trace:
            return self._update_fp16(x, y, collect_metrics=collect_metrics, timing=timing)
        with timing.phase('setup'):
            self.optimizer.zero_grad(set_to_none=True)
        with timing.phase('forward_loss'):
            if self.compile_training and not trace and tuple(x.shape) == (8, CONFIG['trainContext']):
                self.prepare_training_execution()
                torch.compiler.cudagraph_mark_step_begin()
                logits, _, _ = self._compiled_training_call(x)
            else:
                logits, _, _ = self.model(x)
            loss = sequence_loss(logits, y)
        with timing.phase('loss_check'):
            if not torch.isfinite(loss):
                raise FloatingPointError('Training loss is non-finite; reset and reduce the learning rate.')
        with timing.phase('backward'):
            loss.backward()
        selected = None
        if trace:
            for name, parameter in self.model.named_parameters():
                if parameter.grad is not None:
                    magnitude, index = parameter.grad.detach().flatten().abs().max(0)
                    if selected is None or float(magnitude) > selected[0]:
                        selected = (float(magnitude), name, parameter, int(index))
            _, name, parameter, index = selected
            before, gradient = float(parameter.detach().flatten()[index]), float(parameter.grad.flatten()[index])
        with timing.phase('gradient_check_clip'):
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True))
        with timing.phase('optimizer'):
            self.optimizer.step()
        self.step_count += 1
        if not collect_metrics and not trace:
            return None
        result = {'loss': float(loss.detach()), 'gradientNorm': gradient_norm,
                  'clippedGradientNorm': min(1.0, gradient_norm)}
        if trace:
            state = self.optimizer.state[parameter]
            adam_step = int(state['step'])
            m, v = float(state['exp_avg'].flatten()[index]), float(state['exp_avg_sq'].flatten()[index])
            after = float(parameter.detach().flatten()[index])
            result['weight'] = {'name': name, 'index': index, 'before': before, 'gradient': gradient,
                                'm': m, 'v': v, 'delta': after - before, 'after': after,
                                'mHat': m / (1 - 0.9 ** adam_step), 'vHat': v / (1 - 0.999 ** adam_step),
                                'learningRate': learning_rate, 'adamStep': adam_step, 'epsilon': 1e-8,
                                'gradientConvention': 'raw gradient before global-norm clipping',
                                'clippedGradient': gradient * min(1.0, 1.0 / (gradient_norm + 1e-6))}
        return result

    def _update_fp16(self, x, y, collect_metrics=True, timing=NULL_SAMPLE):
        with timing.phase('setup'):
            compiled = self.compile_training and tuple(x.shape) == (8, CONFIG['trainContext'])
            if compiled:
                self.prepare_training_execution()
            retry_rng = _retry_rng(self.device)
        applied = 0
        def applied_step(optimizer, args, kwargs):
            nonlocal applied
            applied += 1
            self.step_count += 1
        hook = self.optimizer.register_step_post_hook(applied_step)
        try:
            for attempt in range(AMP_MAX_ATTEMPTS):
                with timing.phase('setup'):
                    self.optimizer.zero_grad(set_to_none=True)
                with timing.phase('forward_loss'):
                    with torch.autocast('cuda', dtype=torch.float16):
                        if compiled:
                            torch.compiler.cudagraph_mark_step_begin()
                            logits, _, _ = self._compiled_training_call(x)
                        else:
                            logits, _, _ = self.model(x)
                        loss = sequence_loss(logits, y)
                with timing.phase('loss_check'):
                    if not torch.isfinite(loss):
                        raise FloatingPointError('FP16 forward loss is non-finite; reset or reduce the learning rate.')
                with timing.phase('backward'):
                    self._grad_scaler.scale(loss).backward()
                with timing.phase('gradient_check_clip'):
                    self._grad_scaler.unscale_(self.optimizer)
                    norm = _clip_amp_gradients(self.model)
                with timing.phase('optimizer'):
                    self._grad_scaler.step(self.optimizer)
                    self._grad_scaler.update()
                if applied:
                    if applied != 1 or not math.isfinite(norm):
                        raise RuntimeError('Invalid FP16 optimizer-step accounting.')
                    if not collect_metrics:
                        return None
                    return {'loss': float(loss.detach()), 'gradientNorm': norm,
                            'clippedGradientNorm': min(1.0, norm), 'overflowRetries': attempt}
                # Retry exactly this batch and any stochastic forward draws. The
                # current GPT has dropout=0; this also protects future dropout.
                _restore_retry_rng(retry_rng)
            raise FloatingPointError('FP16 gradients overflowed in 8 attempts; no optimizer update was applied. Reduce the learning rate or reset.')
        except Exception:
            self.optimizer.zero_grad(set_to_none=True)
            # Clear any unfinished unscale/step stage while preserving the most
            # recent scale. Never retry after the post-step hook observed a step.
            self._grad_scaler, self._dormant_grad_scaler = _new_grad_scaler(
                self.training_precision, _scaler_snapshot(self._grad_scaler))
            if not applied:
                _restore_retry_rng(retry_rng)
            raise
        finally:
            hook.remove()

    def preview(self, example_id=None, learning_rate=0.0003):
        rate = self._rate(learning_rate)
        if example_id is None:
            example = self.stories_train[self.step_count % len(self.stories_train)]
        else:
            if example_id not in self._story_index:
                raise ValueError('Choose an example from the training split.')
            example = self._story_index[example_id]
        original = self._capture()
        self._preview = None
        try:
            selected_position = 5
            before = self.inspect(example['text'], selected_position)
            x, y = self._input(example['text'])
            update = self._update(x, y, rate, trace=True)
            after = self.inspect(example['text'], selected_position)
            self._measure()
            preview_id = uuid.uuid4().hex
            trace = {'previewId': preview_id, 'step': self.step_count, 'example': {'id': example['id'], 'text': example['text']},
                     'before': before, **update, 'after': after, 'applied': False,
                     'lossDescription': 'One real training example: average next-token NLL across all displayed valid targets. The selected token loss is a separate component.'}
            resulting_state = self._capture()
        finally:
            self._restore(original)
        self._preview = {'id': preview_id, 'state': resulting_state, 'trace': trace}
        return copy.deepcopy(trace)

    def commit(self, preview_id):
        if not self._preview or preview_id != self._preview['id']:
            raise ValueError('This preview is stale or already applied. Prepare a new update.')
        cached = self._preview
        self._preview = None
        self._restore(cached['state'])
        self.last_trace = copy.deepcopy(cached['trace'])
        self.last_trace['applied'] = True
        self._record_token_history()
        return self.state()

    def train(self, steps=5, learning_rate=0.0003):
        if not isinstance(steps, int) or not 1 <= steps <= 10:
            raise ValueError('Train 1–10 optimizer steps per request.')
        rate = self._rate(learning_rate)
        self._preview, self.last_trace = None, None
        for _ in range(steps):
            sample = self._timing.begin_step('lm', self.step_count + 1)
            success = False
            try:
                with sample.phase('data'):
                    indices = [self.random.randrange(len(self.stories_train)) for _ in range(8)]
                    x, y = self._batch(self.stories_train, indices, random_windows=True)
                self._update(x, y, rate, collect_metrics=False, timing=sample)
                success = True
            finally:
                sample.end(success=success)
            if self.step_count % EVALUATION_INTERVAL_STEPS == 0:
                measurement = self._timing.begin_evaluation('lm', self.step_count)
                success = False
                try:
                    self._measure(timing=measurement)
                    if self._token_history.active:
                        with measurement.phase('token_observation'):
                            self._record_token_history()
                    success = True
                finally:
                    measurement.end(success=success)
        return self.state()

    def reset(self, source='random', seed=42):
        if source == 'random':
            self._initialize(seed)
        elif source == 'pretrained':
            path = self.artifact_dir / 'pretrained.pt'
            if not path.is_file():
                raise FileNotFoundError('No recorded pretrained checkpoint is available yet.')
            state = torch.load(path, map_location=self.device, weights_only=False)
            self._restore(state)
            self.source, self._preview, self.last_trace = 'pretrained', None, None
            self._timing.reset('lm')
        else:
            raise ValueError('Reset source must be random or pretrained.')
        self.prepare_training_execution()
        self._token_history.clear()
        return self.state()

    def recordings(self):
        path = self.artifact_dir / 'recordings.json'
        if not path.is_file():
            return {'pretraining': {'available': False, 'history': [], 'checkpoints': []},
                    'transfer': {'available': False, 'runs': []},
                    'notes': ['No recorded run has been produced yet. Live training uses actual model computation.']}
        return json.loads(path.read_text(encoding='utf-8'))

    @staticmethod
    def _seed(value):
        if type(value) is not int or not 0 <= value <= 2147483647:
            raise ValueError('Seed must be an integer between 0 and 2147483647.')
        return value

    @staticmethod
    def _text(value):
        if not isinstance(value, str) or not value.strip() or len(value) > 8192:
            raise ValueError('Text must contain 1–8192 characters.')
        return value

    @torch.no_grad()
    def generate(self, text, max_new_tokens=48, temperature=0.8, seed=42, model='live', on_event=None,
                 sliding_window=False, window_size=128):
        """Autoregressive inference without changing the live training session."""
        self._text(text)
        self._seed(seed)
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 1000:
            raise ValueError('Generate 1–1000 new tokens per request.')
        if type(sliding_window) is not bool:
            raise ValueError('sliding_window must be a boolean.')
        if type(window_size) is not int or not 1 <= window_size <= CONFIG['context']:
            raise ValueError('Window size must be an integer between 1 and 128.')
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or not 0 <= temperature <= 1.5:
            raise ValueError('Temperature must be between 0 and 1.5.')
        if model not in ('live', 'random'):
            raise ValueError('Generation model must be live or random.')
        if on_event is not None and not callable(on_event):
            raise ValueError('on_event must be callable when supplied.')
        prompt = [BOS] + self.tokenizer.encode(text).ids
        if not sliding_window and len(prompt) >= CONFIG['context']:
            raise ValueError('Prompt must contain fewer than 128 tokens including BOS.')
        generated, tokens = [], []
        source = {'model': model, 'step': self.step_count if model == 'live' else 0,
                  'seed': self.seed, 'source': self.source if model == 'live' else 'random'}
        sampling = {'temperature': temperature, 'seed': seed}
        effective_window = window_size if sliding_window else CONFIG['context']
        context = {'mode': 'sliding' if sliding_window else 'fixed', 'windowSize': effective_window,
                   'modelContext': CONFIG['context'], 'promptTokens': len(prompt)}
        budget = max_new_tokens if sliding_window else min(max_new_tokens, CONFIG['context'] - len(prompt))

        def emit(event):
            if on_event is not None:
                on_event(copy.deepcopy(event))

        with _isolated_rng(self.device):
            if model == 'random':
                torch.manual_seed(self.seed)
                selected = GPT(self.vocab_size).to(self.device)
            else:
                selected = self.model
            modes = [(module, module.training) for module in selected.modules()]
            selected.eval()
            sampling_rng = torch.Generator(device=self.device).manual_seed(seed)
            stop_reason = 'length'
            try:
                emit({'type': 'start', 'text': text, 'source': source, 'sampling': sampling,
                      'maxNewTokens': max_new_tokens, 'context': context})
                for _ in range(budget):
                    full_ids = prompt + generated
                    input_ids = full_ids[-effective_window:] if sliding_window else full_ids
                    window = {'inputTokens': len(input_ids), 'droppedTokens': len(full_ids) - len(input_ids)}
                    # Each forward is independent: retain exact token IDs, drop
                    # old context and let GPT restart its learned positions at 0.
                    x = torch.tensor([input_ids], device=self.device)
                    logits = selected(x)[0][0, -1].clone()
                    if not torch.isfinite(logits).all():
                        raise FloatingPointError('Generation logits are non-finite.')
                    logits[[PAD, BOS, CLS, UNK]] = float('-inf')
                    if temperature == 0:
                        token_id, probability = int(logits.argmax()), 1.0
                    else:
                        # Scale in float64: a valid temperature such as 1e-300
                        # becomes zero in float32, which would produce 0/0.
                        # Below float64's normal range, every nonzero float32
                        # logit gap already has exp(gap / T) == 0 numerically.
                        # Preserve tied maxima as a uniform distribution.
                        if temperature < torch.finfo(torch.float64).tiny:
                            distribution = logits.eq(logits.max()).double()
                            distribution /= distribution.sum()
                        else:
                            shifted = logits.double() - logits.max().double()
                            distribution = (shifted / temperature).softmax(-1)
                        if not torch.isfinite(distribution).all():
                            raise FloatingPointError('Generation probabilities are non-finite.')
                        token_id = int(torch.multinomial(distribution, 1, generator=sampling_rng))
                        probability = float(distribution[token_id])
                    generated.append(token_id)
                    tokens.append({'id': token_id, 'piece': self._piece(token_id), 'p': probability})
                    if on_event is not None:
                        emit({'type': 'token', 'index': len(tokens) - 1, 'token': tokens[-1],
                              'continuation': self.tokenizer.decode([item for item in generated if item != EOS]),
                              'window': window})
                    if token_id == EOS:
                        stop_reason = 'eos'
                        break
                    if not sliding_window and len(prompt) + len(generated) == CONFIG['context']:
                        stop_reason = 'context'
                        break
                # Decode entire BPE sequences, never concatenate raw byte-level pieces.
                continuation_ids = [item for item in generated if item != EOS]
                result = {'text': text, 'continuation': self.tokenizer.decode(continuation_ids),
                          'fullText': self.tokenizer.decode(prompt[1:] + continuation_ids),
                          'tokens': tokens, 'stopReason': stop_reason,
                          'source': source, 'sampling': sampling, 'context': context, 'window': window}
                emit({'type': 'done', 'result': result})
            finally:
                for module, training in modes:
                    module.training = training
        return result

    def _measure_transfer(self, pair, timing=NULL_SAMPLE):
        measured = {}
        for name, branch in pair['branches'].items():
            with timing.phase(name + '.train_evaluation'):
                train = evaluate_cached_classifier(branch['model'], self._sentiment_eval_batches['train'],
                    auxiliary_weight=pair['auxiliaryWeight'] if name == 'pretrained' else 0,
                    lm_loss_fn=sequence_loss)
            with timing.phase(name + '.validation'):
                validation = evaluate_cached_classifier(branch['model'], self._sentiment_eval_batches['validation'])
            metrics = {'trainLoss': train['loss'], 'valLoss': validation['loss'], 'accuracy': validation['accuracy']}
            if not all(math.isfinite(value) for value in metrics.values()):
                raise FloatingPointError('Transfer evaluation produced non-finite metrics.')
            objective = {'schemaVersion': 1, 'step': pair['step'], **train['objective'],
                         'split': 'train', 'measurement': 'fixed-subset-eval'}
            if not all(math.isfinite(objective[key]) for key in
                       ('classificationLoss', 'auxiliaryContribution', 'totalLoss')) or (
                       objective['lmLoss'] is not None and not math.isfinite(objective['lmLoss'])):
                raise FloatingPointError('Transfer evaluation produced a non-finite objective.')
            metrics['objective'] = objective
            measured[name] = metrics
        # Publish a complete paired observation, never a mixture of two steps.
        for name, metrics in measured.items():
            branch = pair['branches'][name]
            branch['metrics'] = metrics
            branch['history'].append({'step': pair['step'], **copy.deepcopy(metrics)})

    def _prepare_transfer_bindings(self, pair):
        """Prepare replacements before publishing a pair; never serialize callables."""
        if pair is None:
            return {}
        enabled = self.compile_training and self.device.type == 'cuda' and self.training_precision == 'fp16'
        bindings = {}
        for name, branch in pair['branches'].items():
            binding = PreparedTransferObjective(branch['model'], sequence_loss,
                auxiliary_weight=pair['auxiliaryWeight'] if name == 'pretrained' else 0,
                precision=self.training_precision, enabled=bool(enabled))
            binding.prepare()
            bindings[name] = binding
        return bindings

    def start_transfer(self, seed=42, auxiliary_weight=0):
        """Own two independent copies; never optimize the parent language model."""
        self._seed(seed)
        if isinstance(auxiliary_weight, bool) or auxiliary_weight not in (0, 0.5):
            raise ValueError('Auxiliary weight must be 0 or 0.5.')
        pair = {'runId': uuid.uuid4().hex,
                'source': {'step': self.step_count, 'seed': self.seed, 'source': self.source,
                           'modelSha256': _model_sha256(self.model)},
                'seed': seed, 'step': 0, 'auxiliaryWeight': float(auxiliary_weight),
                'random': random.Random(seed + 10000), 'branches': {},
                'evalDescription': f'Classification cross-entropy on a fixed evenly spaced subset of {min(128, len(self.sentiment_train))} training reviews and all {len(self.sentiment_val)} validation reviews; accuracy uses the full validation split. No test data or user-entered text. Both branches use identical training batches and freshly initialized classifier heads; only pretrained uses the selected auxiliary LM loss.'}
        with _isolated_rng(self.device):
            torch.manual_seed(seed)
            scratch = GPT(self.vocab_size).to(self.device)
            pretrained = copy.deepcopy(self.model)
            pretrained.classifier.load_state_dict(scratch.classifier.state_dict())
            for name, selected in (('pretrained', pretrained), ('scratch', scratch)):
                scaler, dormant = _new_grad_scaler(self.training_precision)
                pair['branches'][name] = {'model': selected,
                    'optimizer': torch.optim.Adam(selected.parameters(), lr=0.0003, weight_decay=0.0),
                    'gradScaler': scaler, 'dormantGradScaler': dormant,
                    'metrics': {}, 'history': []}
            self._measure_transfer(pair)
            bindings = self._prepare_transfer_bindings(pair)
            pair['torchRandom'] = torch.random.get_rng_state().clone()
            pair['cudaRandom'] = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        self._transfer = pair
        self._transfer_execution = bindings
        self._timing.reset('transfer')
        return self.transfer_state()

    def transfer_state(self):
        if self._transfer is None:
            return {'available': False}
        self._timing.drain()
        pair = self._transfer
        measured_steps = {branch['history'][-1]['step'] if branch['history'] else None
                          for branch in pair['branches'].values()}
        return {'available': True,
                'lastMeasuredStep': next(iter(measured_steps)) if len(measured_steps) == 1 else None,
                'evaluationIntervalSteps': EVALUATION_INTERVAL_STEPS,
                'timing': self._timing.published_snapshot('transfer'),
                'requiresRestart': bool(pair.get('optimizerFailure', False)),
                **{key: copy.deepcopy(pair[key]) for key in ('runId', 'source', 'seed', 'step', 'auxiliaryWeight', 'evalDescription')},
                'branches': [{'name': name, 'metrics': copy.deepcopy(branch['metrics']),
                              'history': copy.deepcopy(branch['history'])}
                             for name, branch in pair['branches'].items()]}

    def train_transfer(self, steps=10, learning_rate=0.0003):
        if type(steps) is not int or not 1 <= steps <= 10:
            raise ValueError('Train 1–10 paired optimizer steps per request.')
        rate = self._rate(learning_rate)
        if self._transfer is None:
            raise ValueError('Start a live transfer pair before fine-tuning.')
        pair = self._transfer
        if pair.get('optimizerFailure'):
            raise ValueError('A transfer optimizer previously failed during a paired update. Restart transfer before continuing.')
        with _isolated_rng(self.device, pair):
            for _ in range(steps):
                sample = self._timing.begin_step('transfer', pair['step'] + 1)
                success = False
                try:
                    with sample.phase('data'):
                        rows = [self.sentiment_train[pair['random'].randrange(len(self.sentiment_train))] for _ in range(8)]
                        x, ends, labels = sentiment_batch(rows, self.device)
                    if self.training_precision == 'fp16':
                        self._update_transfer_fp16(pair, x, ends, labels, rate, timing=sample)
                    else:
                        # Prepare and validate both gradients before changing either branch.
                        for name, branch in pair['branches'].items():
                            with sample.phase(name + '.setup'):
                                selected, optimizer = branch['model'], branch['optimizer']
                                selected.train()
                                for group in optimizer.param_groups:
                                    group['lr'] = rate
                                optimizer.zero_grad(set_to_none=True)
                            with sample.phase(name + '.forward_loss'):
                                weight = pair['auxiliaryWeight'] if name == 'pretrained' else 0
                                lm_logits, hidden, _ = selected(x, lm_head=weight > 0)
                                logits = selected.classifier(hidden[torch.arange(len(rows), device=self.device), ends])
                                objective = F.cross_entropy(logits, labels)
                                if weight:
                                    objective = objective + weight * sequence_loss(lm_logits[:, :-1], x[:, 1:])
                            with sample.phase(name + '.loss_check'):
                                if not torch.isfinite(objective):
                                    raise FloatingPointError(f'Non-finite {name} fine-tuning objective.')
                            with sample.phase(name + '.backward'):
                                objective.backward()
                            with sample.phase(name + '.gradient_check_clip'):
                                torch.nn.utils.clip_grad_norm_(selected.parameters(), 1.0, error_if_nonfinite=True)
                        for name, branch in pair['branches'].items():
                            with sample.phase(name + '.optimizer'):
                                branch['optimizer'].step()
                        pair['step'] += 1
                    success = True
                finally:
                    sample.end(success=success)
                if pair['step'] % EVALUATION_INTERVAL_STEPS == 0:
                    measurement = self._timing.begin_evaluation('transfer', pair['step'])
                    success = False
                    try:
                        self._measure_transfer(pair, timing=measurement)
                        success = True
                    finally:
                        measurement.end(success=success)
        return self.transfer_state()

    def _update_transfer_fp16(self, pair, x, ends, labels, rate, timing=NULL_SAMPLE):
        """Validate both AMP gradients before changing either comparison branch."""
        retry_rng = _retry_rng(self.device)
        applied = {name: 0 for name in pair['branches']}
        commit_started = False
        hooks = []
        for name, branch in pair['branches'].items():
            def counted(optimizer, args, kwargs, name=name):
                applied[name] += 1
            hooks.append(branch['optimizer'].register_step_post_hook(counted))
        try:
            for attempt in range(AMP_MAX_ATTEMPTS):
                finite = {}
                for name, branch in pair['branches'].items():
                    with timing.phase(name + '.setup'):
                        selected, optimizer, scaler = branch['model'], branch['optimizer'], branch['gradScaler']
                        selected.train()
                        for group in optimizer.param_groups:
                            group['lr'] = rate
                        optimizer.zero_grad(set_to_none=True)
                        binding = self._transfer_execution.get(name)
                        if binding is None or binding.model is not selected:
                            raise RuntimeError('Transfer execution is not bound to the current model; restore or restart transfer.')
                    with timing.phase(name + '.forward_loss'):
                        objective = binding(x, ends, labels)
                    with timing.phase(name + '.loss_check'):
                        if not torch.isfinite(objective):
                            raise FloatingPointError(f'Non-finite {name} FP16 fine-tuning objective; neither branch was updated.')
                    with timing.phase(name + '.backward'):
                        scaler.scale(objective).backward()
                    with timing.phase(name + '.gradient_check_clip'):
                        scaler.unscale_(optimizer)
                        finite[name] = math.isfinite(_clip_amp_gradients(selected))
                if not all(finite.values()):
                    # Public update(new_scale) clears the pending unscale stage
                    # without stepping a healthy branch. Bad branches back off;
                    # healthy branches retain their scale and growth progress.
                    for name, branch in pair['branches'].items():
                        scaler = branch['gradScaler']
                        state = scaler.state_dict()
                        scale = state['scale'] if finite[name] else state['scale'] * state['backoff_factor']
                        scaler.update(new_scale=scale)
                        if not finite[name]:
                            adjusted = scaler.state_dict()
                            adjusted['_growth_tracker'] = 0
                            scaler.load_state_dict(adjusted)
                        branch['optimizer'].zero_grad(set_to_none=True)
                    _restore_retry_rng(retry_rng)
                    continue
                # Both branches have finite unscaled/clipped gradients. A later
                # exception is explicit and must never trigger a pair replay.
                commit_started = True
                for name, branch in pair['branches'].items():
                    with timing.phase(name + '.optimizer'):
                        branch['gradScaler'].step(branch['optimizer'])
                if any(count != 1 for count in applied.values()):
                    raise RuntimeError('Invalid paired FP16 optimizer-step accounting.')
                pair['step'] += 1
                for name, branch in pair['branches'].items():
                    with timing.phase(name + '.optimizer'):
                        branch['gradScaler'].update()
                return
            raise FloatingPointError('FP16 transfer gradients overflowed in 8 attempts; neither branch was updated. Reduce the learning rate or restart transfer.')
        except Exception as error:
            for branch in pair['branches'].values():
                branch['optimizer'].zero_grad(set_to_none=True)
                branch['gradScaler'], branch['dormantGradScaler'] = _new_grad_scaler(
                    self.training_precision, _scaler_snapshot(branch['gradScaler']))
            if not any(applied.values()):
                _restore_retry_rng(retry_rng)
            if commit_started and any(count != 1 for count in applied.values()):
                # An arbitrary optimizer exception can leave partial internal
                # state. Persist the invalid-pair marker and fail closed.
                pair['optimizerFailure'] = True
                raise RuntimeError('A transfer optimizer failed during a paired update. Restart transfer before continuing.') from error
            raise
        finally:
            for hook in hooks:
                hook.remove()

    @torch.no_grad()
    def predict_transfer(self, text, branch='pretrained'):
        self._text(text)
        if branch not in ('pretrained', 'scratch'):
            raise ValueError('Live transfer branch must be pretrained or scratch.')
        if self._transfer is None or self._transfer['step'] < 1:
            raise ValueError('Fine-tune the live transfer pair before classification.')
        pair = self._transfer
        selected = pair['branches'][branch]['model']
        modes = [(module, module.training) for module in selected.modules()]
        with _isolated_rng(self.device):
            try:
                selected.eval()
                ids = [BOS] + self.tokenizer.encode(text).ids[:CONFIG['context'] - 2] + [CLS]
                _, hidden, _ = selected(torch.tensor([ids], device=self.device), lm_head=False)
                probabilities = selected.classifier(hidden[:, -1]).softmax(-1)[0]
            finally:
                for module, training in modes:
                    module.training = training
        return {'text': text, 'branch': branch,
                'prediction': ('negative', 'positive')[int(probabilities.argmax())],
                'probabilities': [{'label': label, 'p': float(probabilities[i])} for i, label in enumerate(('negative', 'positive'))],
                'runId': pair['runId'], 'source': copy.deepcopy(pair['source']), 'step': pair['step']}

    def _capture_transfer(self):
        if self._transfer is None:
            return None
        pair = self._transfer
        saved = {key: _cpu_copy(value) for key, value in pair.items() if key not in ('random', 'branches')}
        saved['random'] = pair['random'].getstate()
        saved['branches'] = {name: {'model': _cpu_copy(branch['model'].state_dict()),
                                   'optimizer': _cpu_copy(branch['optimizer'].state_dict()),
                                   'gradScaler': _scaler_snapshot(branch.get('gradScaler'), branch.get('dormantGradScaler')),
                                   'metrics': copy.deepcopy(branch['metrics']), 'history': copy.deepcopy(branch['history'])}
                             for name, branch in pair['branches'].items()}
        return saved

    def save_live(self):
        saved_at = datetime.now(timezone.utc).isoformat()
        saved = {'liveFormatVersion': LIVE_FORMAT_VERSION, 'config': dict(CONFIG), 'vocabSize': self.vocab_size,
                 'dataManifestSha256': hashlib.sha256((self.data_dir / 'manifest.json').read_bytes()).hexdigest(),
                 'tokenizerSha256': hashlib.sha256((self.data_dir / 'tokenizer.json').read_bytes()).hexdigest(),
                 'savedAt': saved_at, 'lm': self._capture(), 'transfer': self._capture_transfer()}
        token_history = self.token_history()
        if token_history['available']:
            saved['tokenHistory'] = token_history
        _atomic_save(self.artifact_dir / 'live' / 'session.pt', saved)
        return {'saved': True, 'step': self.step_count,
                'transferStep': self._transfer['step'] if self._transfer is not None else None, 'savedAt': saved_at}

    def _load_transfer(self, saved):
        if saved is None:
            return None
        if not isinstance(saved, dict) or set(saved['branches']) != {'pretrained', 'scratch'}:
            raise ValueError('Incompatible live transfer checkpoint.')
        self._seed(saved['seed'])
        if type(saved['step']) is not int or saved['step'] < 0 or saved['auxiliaryWeight'] not in (0, 0.5):
            raise ValueError('Incompatible live transfer progress.')
        pair = {key: _cpu_copy(value) for key, value in saved.items() if key not in ('random', 'branches')}
        pair['random'] = random.Random()
        pair['random'].setstate(saved['random'])
        pair['branches'] = {}
        # Also validate private Torch/CUDA RNG before accepting the session.
        with _isolated_rng(self.device, pair):
            for name in ('pretrained', 'scratch'):
                branch = saved['branches'][name]
                selected = GPT(self.vocab_size).to(self.device)
                selected.load_state_dict(branch['model'])
                selected.eval()
                optimizer = torch.optim.Adam(selected.parameters(), lr=0.0003, weight_decay=0.0)
                optimizer.load_state_dict(branch['optimizer'])
                scaler, dormant = _new_grad_scaler(self.training_precision, branch.get('gradScaler'))
                pair['branches'][name] = {'model': selected, 'optimizer': optimizer,
                                         'gradScaler': scaler, 'dormantGradScaler': dormant,
                                         'metrics': copy.deepcopy(branch['metrics']),
                                         'history': copy.deepcopy(branch['history'])}
        # Construction is incidental; continuation uses the saved private RNG.
        pair['torchRandom'] = saved['torchRandom'].cpu().clone()
        pair['cudaRandom'] = _cpu_copy(saved.get('cudaRandom'))
        return pair

    def restore_live(self):
        path = self.artifact_dir / 'live' / 'session.pt'
        if not path.is_file():
            raise FileNotFoundError('No saved live session is available.')
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if not isinstance(saved, dict) or saved.get('liveFormatVersion') != LIVE_FORMAT_VERSION or saved.get('config') != CONFIG or saved.get('vocabSize') != self.vocab_size:
            raise ValueError('Incompatible live session checkpoint.')
        for key, filename in (('dataManifestSha256', 'manifest.json'), ('tokenizerSha256', 'tokenizer.json')):
            if saved.get(key) != hashlib.sha256((self.data_dir / filename).read_bytes()).hexdigest():
                raise ValueError('Saved live session data/tokenizer provenance does not match.')
        token_history = self._validated_token_history(saved.get('tokenHistory'), saved['lm'])
        # Build complete replacement objects first. A rejected file cannot partly
        # overwrite the live weights, optimizer, preview, transfer pair or RNG.
        with _isolated_rng(self.device):
            candidate = object.__new__(TrainingLab)
            candidate.data_dir, candidate.device, candidate.vocab_size = self.data_dir, self.device, self.vocab_size
            candidate.compile_training = self.compile_training
            candidate.training_precision = self.training_precision
            candidate._invalidate_training_execution()
            candidate.model = GPT(self.vocab_size).to(self.device)
            candidate.optimizer = torch.optim.Adam(candidate.model.parameters(), lr=0.0003, weight_decay=0.0)
            candidate.random = random.Random()
            try:
                candidate._restore(saved['lm'])
                transfer = self._load_transfer(saved['transfer'])
            except (KeyError, TypeError, RuntimeError, IndexError) as error:
                raise ValueError('Invalid live session state.') from error
            candidate.prepare_training_execution()
            transfer_execution = self._prepare_transfer_bindings(transfer)
        torch.random.set_rng_state(saved['lm']['torchRandom'].cpu())
        if self.device.type == 'cuda' and saved['lm'].get('cudaRandom') is not None:
            torch.cuda.set_rng_state_all([item.cpu() for item in saved['lm']['cudaRandom']])
        for key in ('model', 'optimizer', 'random', 'step_count', 'seed', 'source', 'history', 'metrics', 'last_trace',
                    '_compiled_training_model', '_compiled_training_call', '_compile_warmup_seconds',
                    '_grad_scaler', '_dormant_grad_scaler'):
            setattr(self, key, getattr(candidate, key))
        self._transfer, self._preview = transfer, None
        self._transfer_execution = transfer_execution
        self._timing.reset()
        self._token_history.restore(token_history)
        return self.state()

    @torch.no_grad()
    def predict(self, text, branch='pretrained'):
        if branch not in ('scratch', 'pretrained', 'auxiliary'):
            raise ValueError('Unknown transfer branch.')
        if not isinstance(text, str) or not text.strip() or len(text) > 8192:
            raise ValueError('Text must contain 1–8192 characters.')
        runs = [r for r in self.recordings()['transfer']['runs'] if r['name'] == branch and r.get('checkpoint')]
        if not runs:
            raise FileNotFoundError(f'No trained {branch} classifier checkpoint is available.')
        run = sorted(runs, key=lambda r: r['seed'])[0]
        # Only our fixed filenames are accepted; the browser cannot choose paths.
        path = self.artifact_dir / f'transfer-{branch}-seed{int(run["seed"])}.pt'
        if not path.is_file():
            raise FileNotFoundError('The recorded classifier checkpoint is missing.')
        key = (branch, path.stat().st_mtime_ns)
        if key not in self._predict_cache:
            state = torch.load(path, map_location=self.device, weights_only=False)
            if state['config'] != CONFIG or state['vocabSize'] != self.vocab_size:
                raise ValueError('Classifier checkpoint is incompatible with this tokenizer.')
            expected_manifest = hashlib.sha256((self.data_dir / 'manifest.json').read_bytes()).hexdigest()
            if state.get('dataManifestSha256') != expected_manifest:
                raise ValueError('Classifier checkpoint provenance does not match the current data manifest.')
            model = GPT(self.vocab_size).to(self.device)
            model.load_state_dict(state['model'])
            model.eval()
            self._predict_cache = {k: v for k, v in self._predict_cache.items() if k[0] != branch}
            self._predict_cache[key] = model
        model = self._predict_cache[key]
        ids = [BOS] + self.tokenizer.encode(text).ids[:CONFIG['context'] - 2] + [CLS]
        x = torch.tensor([ids], device=self.device)
        _, hidden, _ = model(x, lm_head=False)
        p = model.classifier(hidden[:, -1]).softmax(-1)[0]
        return {'text': text, 'branch': branch,
                'probabilities': [{'label': label, 'p': float(p[i])} for i, label in enumerate(('negative', 'positive'))],
                'prediction': ('negative', 'positive')[int(p.argmax())],
                'checkpoint': {'seed': run['seed'], 'step': run['history'][-1]['step']}, 'trained': True}


def sentiment_batch(rows, device, context=128):
    sequences = [[BOS] + list(map(int, r['ids']))[:context - 2] + [CLS] for r in rows]
    x = np.full((len(rows), max(map(len, sequences))), PAD, dtype=np.int64)
    lengths = []
    for i, sequence in enumerate(sequences):
        x[i, :len(sequence)] = sequence
        lengths.append(len(sequence) - 1)
    return (torch.from_numpy(x).to(device),
            torch.from_numpy(np.asarray(lengths, dtype=np.int64)).to(device),
            torch.from_numpy(np.asarray([r['label'] for r in rows], dtype=np.int64)).to(device))


@torch.no_grad()
def evaluate_classifier(model, rows, device, batch_size=16):
    model.eval()
    total, correct, confusion = 0.0, 0, [[0, 0], [0, 0]]
    for start in range(0, len(rows), batch_size):
        part = rows[start:start + batch_size]
        x, ends, labels = sentiment_batch(part, device)
        _, hidden, _ = model(x, lm_head=False)
        logits = model.classifier(hidden[torch.arange(len(part), device=device), ends])
        total += float(F.cross_entropy(logits, labels)) * len(part)
        predicted = logits.argmax(-1)
        correct += int((predicted == labels).sum())
        for y, p in zip(labels.tolist(), predicted.tolist()):
            confusion[y][p] += 1
    return {'loss': total / len(rows), 'accuracy': correct / len(rows), 'confusion': confusion}


def record_experiment(lab, pretrain_steps=200, finetune_steps=100, seeds=(42,),
                      eval_every=25, finetune_eval_every=25,
                      learning_rate=0.0003, finetune_learning_rate=0.0003):
    """Record genuine observations, with equal downstream minibatches per seed.

    One common pretraining run supplies all downstream pretrained branches. A
    seed denotes classifier initialization, scratch-body initialization, and the
    shared downstream example order; it does not pretend to be a new LM run.
    """
    if pretrain_steps < 1 or finetune_steps < 0 or eval_every < 1 or finetune_eval_every < 1:
        raise ValueError('Pretraining requires positive steps; fine-tuning steps may be zero.')
    seeds = tuple(int(s) for s in seeds)
    if not seeds:
        raise ValueError('Provide at least one downstream seed.')
    lab._rate(learning_rate)
    lab._rate(finetune_learning_rate)
    lab.artifact_dir.mkdir(parents=True, exist_ok=True)
    recordings_path = lab.artifact_dir / 'recordings.json'
    started = time.time()
    records = {
        'formatVersion': FORMAT_VERSION,
        'architecture': lab.metadata()['architecture'],
        'dataManifestSha256': hashlib.sha256((lab.data_dir / 'manifest.json').read_bytes()).hexdigest(),
        'pretraining': {'available': True, 'seed': lab.seed, 'history': copy.deepcopy(lab.history), 'checkpoints': [],
                        'completed': False, 'requestedSteps': pretrain_steps, 'learningRate': learning_rate},
        'transfer': {'available': False, 'runs': [], 'requestedSteps': finetune_steps,
                     'seeds': list(seeds), 'learningRate': finetune_learning_rate,
                     'batchSize': 8, 'evaluationSplit': 'validation',
                     'trainingEvaluationExamples': min(128, len(lab.sentiment_train)),
                     'validationExamples': len(lab.sentiment_val),
                     'pretrainingSeed': lab.seed},
        'notes': [
            'All curves are measured from actual training; checkpoint playback is recorded, not live training.',
            'One common LM pretraining run supplies the pretrained and auxiliary branches. Reported downstream seeds vary head initialization, scratch initialization and downstream minibatch order.',
            'Per seed, all three branches have identical classifier initialization, minibatch indices, sequence truncation, Adam settings and optimizer-step budget. All model parameters are trainable.',
            'TinyStories LM evaluation uses fixed evenly spaced subsets of up to 32 documents per split and their first 64 input tokens. It is not the full-corpus perplexity.',
            'Sentiment validation metrics and confusion matrices use the entire validation split. Training classifier loss uses a fixed subset of up to 128 training reviews. Test data are not used.',
            'Transfer history lmLoss is the actual mean language-model loss of the last training minibatch before its update; it is not a validation LM loss.',
            'Auxiliary LM predicts the next token in BOS + review + CLS. Padding is ignored; token losses are averaged within each sequence and then across the batch. Class labels are not LM tokens.',
            'Small data subsets, short training and a scaled model can produce weak or mixed transfer; no gain is assumed.',
        ],
    }

    def save_pretraining_observation():
        step = lab.step_count
        checkpoint = lab.artifact_dir / 'checkpoints' / f'lm-step-{step:06d}.pt'
        _atomic_save(checkpoint, lab._capture())
        probe = lab.inspect(lab.stories_val[0]['text'], 5)
        records['pretraining']['history'] = copy.deepcopy(lab.history)
        records['pretraining']['checkpoints'].append({'step': step, 'label': f'Observed step {step}',
                                                     'probe': probe,
                                                     'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()})
        _json_write(recordings_path, records)

    save_pretraining_observation()
    for _ in range(pretrain_steps):
        indexes = [lab.random.randrange(len(lab.stories_train)) for _ in range(8)]
        x, y = lab._batch(lab.stories_train, indexes, random_windows=True)
        lab._update(x, y, learning_rate)
        if lab.step_count % eval_every == 0 or lab.step_count == pretrain_steps:
            lab._measure()
            save_pretraining_observation()
            print(json.dumps({'phase': 'pretraining', 'step': lab.step_count, **lab.metrics,
                              'elapsedSeconds': round(time.time() - started, 2)}), flush=True)
    lab.source = 'pretrained'
    pretrained_state = lab._capture()
    _atomic_save(lab.artifact_dir / 'pretrained.pt', pretrained_state)
    records['pretraining']['completed'] = True
    records['pretraining']['checkpointSha256'] = hashlib.sha256((lab.artifact_dir / 'pretrained.pt').read_bytes()).hexdigest()
    _json_write(recordings_path, records)

    if finetune_steps:
        train_eval = lab._fixed_subset(lab.sentiment_train, 128)
        for seed in seeds:
            # Isolate this initialization from the live lab and from each branch.
            global_rng = torch.random.get_rng_state()
            torch.manual_seed(seed)
            initial_model = GPT(lab.vocab_size).to(lab.device)
            initial_state = {k: v.detach().cpu().clone() for k, v in initial_model.state_dict().items()}
            head_state = {k: v.detach().cpu().clone() for k, v in initial_model.classifier.state_dict().items()}
            del initial_model
            torch.random.set_rng_state(global_rng)
            ordering = random.Random(seed + 10000)
            minibatches = [[ordering.randrange(len(lab.sentiment_train)) for _ in range(8)] for _ in range(finetune_steps)]
            order_hash = hashlib.sha256(json.dumps(minibatches, separators=(',', ':')).encode()).hexdigest()
            head_hash = hashlib.sha256(b''.join(t.numpy().tobytes() for t in head_state.values())).hexdigest()
            for name, auxiliary_weight in (('scratch', 0.0), ('pretrained', 0.0), ('auxiliary', 0.5)):
                model = GPT(lab.vocab_size).to(lab.device)
                model.load_state_dict(initial_state if name == 'scratch' else pretrained_state['model'])
                model.classifier.load_state_dict(head_state)
                optimizer = torch.optim.Adam(model.parameters(), lr=finetune_learning_rate, weight_decay=0.0)
                run = {'seed': seed, 'name': name, 'lambda': auxiliary_weight, 'history': [],
                       'minibatchOrderSha256': order_hash, 'classifierInitializationSha256': head_hash,
                       'pretrainingSeed': pretrained_state['seed'] if name != 'scratch' else None,
                       'completed': False, 'checkpoint': False}
                records['transfer']['runs'].append(run)
                records['transfer']['available'] = True

                def measure_classifier(step, lm_loss=None):
                    train_metric = evaluate_classifier(model, train_eval, lab.device)
                    val_metric = evaluate_classifier(model, lab.sentiment_val, lab.device)
                    item = {'step': step, 'trainLoss': train_metric['loss'], 'valLoss': val_metric['loss'],
                            'accuracy': val_metric['accuracy']}
                    if lm_loss is not None:
                        item['lmLoss'] = lm_loss
                    run['history'].append(item)
                    run['confusion'] = val_metric['confusion']
                    run['confusionConvention'] = 'rows=true negative/positive; columns=predicted negative/positive'
                    _json_write(recordings_path, records)
                    print(json.dumps({'phase': 'transfer', 'seed': seed, 'branch': name, **item,
                                      'elapsedSeconds': round(time.time() - started, 2)}), flush=True)

                measure_classifier(0)
                for step, indexes in enumerate(minibatches, 1):
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    rows = [lab.sentiment_train[i] for i in indexes]
                    x, ends, labels = sentiment_batch(rows, lab.device)
                    lm_logits, hidden, _ = model(x)
                    logits = model.classifier(hidden[torch.arange(len(rows), device=lab.device), ends])
                    task_loss = F.cross_entropy(logits, labels)
                    lm_loss = sequence_loss(lm_logits[:, :-1], x[:, 1:])
                    objective = task_loss + auxiliary_weight * lm_loss
                    if not torch.isfinite(objective):
                        raise FloatingPointError(f'Non-finite {name} fine-tuning loss at step {step}.')
                    objective.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    if step % finetune_eval_every == 0 or step == finetune_steps:
                        measure_classifier(step, float(lm_loss.detach()))
                checkpoint = lab.artifact_dir / f'transfer-{name}-seed{seed}.pt'
                _atomic_save(checkpoint, {'formatVersion': FORMAT_VERSION, 'config': CONFIG,
                                         'vocabSize': lab.vocab_size, 'seed': seed, 'step': finetune_steps,
                                         'model': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                                         'optimizer': optimizer.state_dict(), 'name': name, 'lambda': auxiliary_weight,
                                         'dataManifestSha256': records['dataManifestSha256'],
                                         'classifierInitializationSha256': head_hash, 'minibatchOrderSha256': order_hash})
                run['completed'], run['checkpoint'] = True, True
                run['checkpointSha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                _json_write(recordings_path, records)
                del model, optimizer
    records['elapsedSeconds'] = round(time.time() - started, 2)
    records['completed'] = True
    _json_write(recordings_path, records)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('record', 'benchmark'))
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).parent / 'data')
    parser.add_argument('--artifact-dir', type=Path, default=Path(__file__).parent / '.training')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pretrain-steps', type=int, default=200)
    parser.add_argument('--finetune-steps', type=int, default=100)
    parser.add_argument('--seeds', default='42')
    parser.add_argument('--eval-every', type=int, default=25)
    parser.add_argument('--finetune-eval-every', type=int, default=25)
    parser.add_argument('--learning-rate', type=float, default=0.0003)
    parser.add_argument('--finetune-learning-rate', type=float, default=0.0003)
    args = parser.parse_args()
    lab = TrainingLab(args.data_dir, args.artifact_dir, args.seed, args.device)
    print(json.dumps({'ready': True, 'architecture': lab.metadata()['architecture'], 'initial': lab.state()['metrics']}), flush=True)
    if args.command == 'benchmark':
        started = time.time()
        state = lab.train(10, args.learning_rate)
        print(json.dumps({'benchmarkSteps': 10, 'elapsedSeconds': time.time() - started, 'state': state}), flush=True)
    else:
        record_experiment(lab, args.pretrain_steps, args.finetune_steps,
                          [int(s) for s in args.seeds.split(',')], args.eval_every,
                          args.finetune_eval_every, args.learning_rate, args.finetune_learning_rate)


if __name__ == '__main__':
    main()
