"""Runtime-only Transfer objective compilation; original models own all state."""
from contextlib import nullcontext
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class _TransferObjective(nn.Module):
    def __init__(self, model, sequence_loss_fn, auxiliary_weight):
        super().__init__()
        self.original_model = model
        self.sequence_loss_fn = sequence_loss_fn
        self.auxiliary_weight = auxiliary_weight

    def forward(self, inputs, ends, labels):
        lm_logits, hidden, _ = self.original_model(inputs, lm_head=self.auxiliary_weight > 0)
        logits = self.original_model.classifier(hidden[torch.arange(len(labels), device=inputs.device), ends])
        objective = F.cross_entropy(logits, labels)
        if self.auxiliary_weight:
            objective = objective + self.auxiliary_weight * self.sequence_loss_fn(lm_logits[:, :-1], inputs[:, 1:])
        return objective


class PreparedTransferObjective:
    """One model/auxiliary-policy binding, independent of Adam and GradScaler.

    CUDA batch-eight inputs retain their actual lengths and padding. Dynamic
    Inductor compilation uses default mode, not a CUDA graph per input length.
    Other batch sizes and disabled execution use the same eager objective.
    """
    def __init__(self, model, sequence_loss_fn, *, auxiliary_weight=0., precision='fp16', enabled=False):
        if type(enabled) is not bool:
            raise ValueError('Transfer compilation enabled must be a boolean.')
        if precision not in ('fp16', 'fp32'):
            raise ValueError('Transfer precision must be fp16 or fp32.')
        if isinstance(auxiliary_weight, bool) or auxiliary_weight not in (0, .5):
            raise ValueError('Transfer auxiliary weight must be 0 or 0.5.')
        self._model = model
        self._precision = precision
        self._enabled = enabled
        self._device = next(model.parameters()).device
        if enabled and (self._device.type != 'cuda' or precision != 'fp16'):
            raise ValueError('Compiled Transfer requires CUDA FP16 AMP.')
        self._objective = _TransferObjective(model, sequence_loss_fn, float(auxiliary_weight))
        self._compiled = None
        self._warmup_seconds = None

    @property
    def model(self):
        return self._model

    def metadata(self):
        ready = self._enabled and self._compiled is not None
        return {'backend': 'compiled-dynamic' if ready else 'eager',
                'ready': bool(ready), 'enabled': self._enabled,
                'warmupSeconds': self._warmup_seconds if ready else None,
                'dynamicLength': bool(ready)}

    def _autocast(self):
        return (torch.autocast('cuda', dtype=torch.float16, enabled=self._precision == 'fp16')
                if self._device.type == 'cuda' else nullcontext())

    def prepare(self):
        """Compile/warm without optimizer/scaler calls or any persistent changes."""
        if not self._enabled or self._compiled is not None:
            return self.metadata()
        modes = [(module, module.training) for module in self._model.modules()]
        gradients = [(p, p.grad, p.grad.detach().clone() if p.grad is not None else None)
                     for p in self._model.parameters()]
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        torch_rng = torch.random.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state(self._device).clone()
        started = time.perf_counter()
        try:
            candidate = torch.compile(self._objective, dynamic=True, mode='default')
            self._model.train()
            vocabulary = self._model.token.num_embeddings
            maximum = self._model.position.num_embeddings
            # Inductor specializes aligned and unaligned sequence strides. Warm
            # both before a job; actual batches keep their original lengths.
            for length in sorted({min(40, maximum), min(41, maximum), min(64, maximum)}):
                inputs = torch.arange(8 * length, device=self._device).reshape(8, length).remainder(vocabulary)
                ends = length - 1 - torch.arange(8, device=self._device).remainder(max(1, min(8, length-1)))
                labels = torch.arange(8, device=self._device).remainder(2)
                inputs[:, 0] = 1
                inputs[torch.arange(8, device=self._device), ends] = 3
                inputs.masked_fill_(torch.arange(length, device=self._device)[None] > ends[:, None], 0)
                for _ in range(2):
                    for p in self._model.parameters():
                        p.grad = None
                    with self._autocast():
                        objective = candidate(inputs, ends, labels)
                    if not bool(torch.isfinite(objective)):
                        raise FloatingPointError('Non-finite Transfer compilation warmup objective.')
                    objective.backward()
                    if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in self._model.parameters()):
                        raise FloatingPointError('Non-finite Transfer compilation warmup gradient.')
            torch.cuda.synchronize(self._device)
            self._compiled = candidate
            self._warmup_seconds = time.perf_counter() - started
        except Exception as error:
            self._compiled = None
            self._warmup_seconds = None
            raise RuntimeError('Transfer compilation failed; no optimizer update was performed.') from error
        finally:
            for parameter, original, saved in gradients:
                if original is not None:
                    original.copy_(saved)
                parameter.grad = original
            for module, training in modes:
                module.training = training
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            torch.random.set_rng_state(torch_rng)
            torch.cuda.set_rng_state(cuda_rng, self._device)
        return self.metadata()

    def __call__(self, inputs, ends, labels):
        eligible = self._enabled and inputs.device.type == 'cuda' and inputs.shape[0] == 8
        if eligible and self._compiled is None:
            raise RuntimeError('Prepare the Transfer objective before starting optimizer updates.')
        with self._autocast():
            return (self._compiled if eligible else self._objective)(inputs, ends, labels)
