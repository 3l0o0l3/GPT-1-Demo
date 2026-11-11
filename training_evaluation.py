"""Prepared fixed evaluation inputs; no training, RNG or checkpoint ownership.

The caller chooses the exact rows, order, batch sizes and FP32 eager model.
Prepared tensor storage belongs to this cache and must not be mutated by callers.
Evaluation keeps the original model.eval() side effect and Python loss reduction.
"""
from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class LMBatch:
    inputs: torch.Tensor
    targets: torch.Tensor
    count: int


@dataclass(frozen=True, slots=True)
class ClassifierBatch:
    inputs: torch.Tensor
    ends: torch.Tensor
    labels: torch.Tensor
    row_indices: torch.Tensor
    cpu_labels: tuple[int, ...]
    count: int


def _parts(rows, batch_size):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('Evaluation batch size must be a positive integer.')
    if len(rows) == 0:
        raise ValueError('Evaluation requires at least one example.')
    for start in range(0, len(rows), batch_size):
        yield rows[start:start + batch_size]


def _tensor(value, count, name):
    if not isinstance(value, torch.Tensor) or value.dtype != torch.long:
        raise ValueError(f'Packed {name} must be an int64 tensor.')
    if value.ndim < 1 or value.shape[0] != count:
        raise ValueError(f'Packed {name} has the wrong example count.')
    # A packer may reuse scratch buffers. The cache must own stable storage.
    return value.detach().clone()


def prepare_lm(rows, pack, batch_size=8):
    """Pack each fixed batch once: pack(part) -> (inputs, targets), both [B,T]."""
    batches = []
    for part in _parts(rows, batch_size):
        inputs, targets = pack(part)
        inputs = _tensor(inputs, len(part), 'inputs')
        targets = _tensor(targets, len(part), 'targets')
        if inputs.ndim != 2 or targets.shape != inputs.shape or targets.device != inputs.device:
            raise ValueError('Packed LM inputs and targets must have matching [B,T] shapes and devices.')
        batches.append(LMBatch(inputs, targets, len(part)))
    return tuple(batches)


def prepare_classifier(rows, pack, batch_size=16):
    """Pack once: pack(part) -> (inputs[B,T], final_positions[B], labels[B])."""
    batches = []
    for part in _parts(rows, batch_size):
        inputs, ends, labels = pack(part)
        inputs = _tensor(inputs, len(part), 'inputs')
        ends = _tensor(ends, len(part), 'final positions')
        labels = _tensor(labels, len(part), 'labels')
        if (inputs.ndim != 2 or ends.ndim != 1 or labels.ndim != 1
                or ends.device != inputs.device or labels.device != inputs.device):
            raise ValueError('Packed classifier inputs, final positions and labels have incompatible shapes or devices.')
        cpu_labels = tuple(labels.cpu().tolist())
        if any(label not in (0, 1) for label in cpu_labels):
            raise ValueError('Classifier labels must be 0 or 1.')
        batches.append(ClassifierBatch(inputs, ends, labels,
            torch.arange(len(part), device=inputs.device), cpu_labels, len(part)))
    return tuple(batches)


def _weighted_loss(losses, batches):
    # One host transfer, then precisely the prior per-batch Python-float order.
    values = torch.stack(losses).cpu().tolist()
    return _weighted_values(values, batches)


def _weighted_values(values, batches):
    total = 0.0
    count = 0
    for loss, batch in zip(values, batches):
        total += loss * batch.count
        count += batch.count
    return total / count


@torch.no_grad()
def evaluate_lm(model, batches, loss_fn):
    """Return sequence-weighted mean loss; loss_fn returns a scalar batch mean."""
    if not batches:
        raise ValueError('Evaluation requires at least one prepared batch.')
    model.eval()
    losses = []
    for batch in batches:
        logits, _, _ = model(batch.inputs)
        losses.append(loss_fn(logits, batch.targets))
    return _weighted_loss(losses, batches)


@torch.no_grad()
def evaluate_classifier(model, batches, loss_fn=F.cross_entropy, *,
                        auxiliary_weight=None, lm_loss_fn=None):
    """Evaluate classification, optionally measuring its fine-tuning objective.

    ``None`` preserves the classification-only return contract. A weight of 0
    reports an inactive LM component without evaluating the vocabulary head.
    A positive weight needs the caller's original sequence-normalized LM loss.
    The caller adds measurement step/split metadata; this module owns no clock.
    """
    if not batches:
        raise ValueError('Evaluation requires at least one prepared batch.')
    if auxiliary_weight is not None:
        if isinstance(auxiliary_weight, bool) or auxiliary_weight not in (0, .5):
            raise ValueError('Evaluation auxiliary weight must be 0 or 0.5.')
        if auxiliary_weight and not callable(lm_loss_fn):
            raise ValueError('Auxiliary evaluation requires a sequence loss callable.')
    measure_lm = auxiliary_weight is not None and auxiliary_weight > 0
    model.eval()
    losses, lm_losses, predictions = [], [], []
    for batch in batches:
        lm_logits, hidden, _ = model(batch.inputs, lm_head=measure_lm)
        logits = model.classifier(hidden[batch.row_indices, batch.ends])
        losses.append(loss_fn(logits, batch.labels))
        if measure_lm:
            lm_losses.append(lm_loss_fn(lm_logits[:, :-1], batch.inputs[:, 1:]))
        predictions.append(logits.argmax(-1))
    if measure_lm:
        # Transfer both components together, never a scalar per batch or step.
        values = torch.stack((torch.stack(losses), torch.stack(lm_losses)), dim=1).cpu().tolist()
        loss = _weighted_values([row[0] for row in values], batches)
        lm_loss = _weighted_values([row[1] for row in values], batches)
    else:
        loss, lm_loss = _weighted_loss(losses, batches), None
    predicted = torch.cat(predictions).cpu().tolist()
    labels = (label for batch in batches for label in batch.cpu_labels)
    correct, confusion = 0, [[0, 0], [0, 0]]
    for label, prediction in zip(labels, predicted):
        correct += label == prediction
        confusion[label][prediction] += 1
    result = {'loss': loss, 'accuracy': correct / len(predicted), 'confusion': confusion}
    if auxiliary_weight is not None:
        contribution = float(auxiliary_weight) * lm_loss if measure_lm else 0.0
        result['objective'] = {'classificationLoss': loss, 'lmLoss': lm_loss,
                               'auxiliaryWeight': float(auxiliary_weight),
                               'auxiliaryContribution': contribution,
                               'totalLoss': loss + contribution,
                               'sampleCount': sum(batch.count for batch in batches)}
    return result
