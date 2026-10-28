"""Completion log-probs without a full-vocabulary logit tensor.

TRL 0.21 scores a batch by materializing ``[batch, time, vocab]`` logits.
Qwen2.5-3B's vocab is 151936, so eight ~3.8k-token episodes are about 17 GiB
in fp32, which is the allocation that OOMed the colocated A100 step.
``lm_head`` is a linear applied independently to every token. Scoring a
configurable number of token rows at a time, then dropping that chunk's
logits, matches one matmul followed by the same log-softmax.

``selective_log_softmax`` matches TRL 0.21 (``trl/trainer/utils.py``): float32
and float64 use ``x - logsumexp(x)``; other dtypes use ``log_softmax`` per row.
Both reductions are per token, so row chunks do not change them.
"""

from __future__ import annotations

# 256 rows of fp32 logits at vocab 151936 is about 156 MiB. The unchunked
# 8 x ~4k window is about 17 GiB.
DEFAULT_LOGPROB_CHUNK_SIZE = 256


def selective_log_softmax(logits, index):
    """Log-prob of ``index`` along the last axis. Same reduction as TRL 0.21."""
    import torch
    import torch.nn.functional as F

    if logits.dtype in (torch.float32, torch.float64):
        selected = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        # One row at a time, matching TRL, so the softmax is never stored.
        logsumexp_values = torch.stack([torch.logsumexp(row, dim=-1) for row in logits])
        return selected - logsumexp_values
    per_token = []
    for row_logits, row_labels in zip(logits, index, strict=True):
        row_logps = F.log_softmax(row_logits, dim=-1)
        per_token.append(row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1))
    return torch.stack(per_token)


def chunked_token_logps(hidden, target_ids, lm_head, temperature: float, chunk_size: int):
    """Log-probs of ``target_ids`` under ``lm_head(hidden) / temperature``.

    ``hidden`` is ``[batch, time, hidden]`` and already aligned with
    ``target_ids`` (the completion window, last position dropped). ``chunk_size``
    is the number of token rows whose logits exist at once. While autograd is
    on, each chunk is checkpointed so backward recomputes that chunk's logits
    instead of keeping every chunk until the loss step.
    """
    import torch
    from torch.utils.checkpoint import checkpoint

    if chunk_size < 1:
        raise ValueError(f"logprob_chunk_size must be >= 1, got {chunk_size}")
    if hidden.ndim != 3:
        raise ValueError(f"hidden must be [batch, time, hidden], got {tuple(hidden.shape)}")
    if target_ids.shape != hidden.shape[:2]:
        raise ValueError(f"target ids {tuple(target_ids.shape)} do not match hidden {tuple(hidden.shape[:2])}")
    batch, seq, width = hidden.shape
    if seq == 0:
        return hidden.new_zeros(batch, 0)
    flat_hidden = hidden.reshape(batch * seq, width)
    flat_ids = target_ids.reshape(batch * seq)
    # Checkpoint while a grad could flow. A frozen head still needs it when the
    # hidden state (the LoRA backbone) requires grad, and the reverse too.
    train = torch.is_grad_enabled() and (
        flat_hidden.requires_grad or any(param.requires_grad for param in lm_head.parameters())
    )

    def score(rows, ids):
        logits = lm_head(rows)
        logits = logits / temperature
        return selective_log_softmax(logits, ids)

    pieces = []
    for start in range(0, flat_hidden.shape[0], chunk_size):
        rows = flat_hidden[start : start + chunk_size]
        ids = flat_ids[start : start + chunk_size]
        if train:
            pieces.append(checkpoint(score, rows, ids, use_reentrant=False))
        else:
            pieces.append(score(rows, ids))
    return torch.cat(pieces, dim=0).reshape(batch, seq)
