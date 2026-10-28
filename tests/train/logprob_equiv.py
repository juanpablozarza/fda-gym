"""Chunked completion log-probs match one matmul on a tiny linear head.

Run with any interpreter that has torch:

    .venv/bin/python tests/train/logprob_equiv.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    try:
        import torch
        from torch import nn
    except ImportError:
        print("SKIP no torch")
        raise SystemExit(2) from None

    from fdagym.train.token_logps import chunked_token_logps, selective_log_softmax

    torch.manual_seed(0)
    _check_values_and_grads(torch, nn, chunked_token_logps, selective_log_softmax, torch.float32)
    _check_values_and_grads(torch, nn, chunked_token_logps, selective_log_softmax, torch.bfloat16)
    _check_no_grad_chunks(torch, nn, chunked_token_logps, selective_log_softmax)
    _check_edges(torch, nn, chunked_token_logps)
    print("PASS")


def _unchunked(hidden, target_ids, lm_head, temperature, selective_log_softmax):
    logits = lm_head(hidden) / temperature
    return selective_log_softmax(logits, target_ids)


def _close(torch, left, right, dtype) -> bool:
    """GEMM tiling changes with the chunk shape. fp32 stays within 1e-5; bf16 within a few ulps."""
    if dtype == torch.float32:
        return torch.allclose(left, right, rtol=1e-5, atol=1e-5)
    return torch.allclose(left.float(), right.float(), rtol=0.0, atol=0.0625)


def _check_values_and_grads(torch, nn, chunked_token_logps, selective_log_softmax, dtype):
    batch, time, width, vocab = 3, 13, 8, 32
    temperature = 0.8
    ids = torch.randint(0, vocab, (batch, time))
    source = torch.randn(batch, time, width, dtype=dtype)
    weight = torch.randn(vocab, width, dtype=dtype)
    head = nn.Linear(width, vocab, bias=False, dtype=dtype)
    with torch.no_grad():
        head.weight.copy_(weight)
    # The log-softmax itself does not depend on how the rows are grouped.
    logits = head(source)
    full_lp = selective_log_softmax(logits, ids)
    flat_logits = logits.reshape(-1, vocab)
    flat_ids = ids.reshape(-1)
    grouped = torch.cat(
        [
            selective_log_softmax(flat_logits[start : start + 4], flat_ids[start : start + 4])
            for start in range(0, flat_logits.shape[0], 4)
        ]
    ).reshape(batch, time)
    if not torch.equal(full_lp, grouped):
        raise SystemExit(f"{dtype} selective_log_softmax changed when rows were grouped")

    for chunk_size in (1, 4, 13, 100):
        full_hidden = source.detach().requires_grad_(True)
        full_head = nn.Linear(width, vocab, bias=False, dtype=dtype)
        hidden = source.detach().requires_grad_(True)
        chunk_head = nn.Linear(width, vocab, bias=False, dtype=dtype)
        with torch.no_grad():
            full_head.weight.copy_(weight)
            chunk_head.weight.copy_(weight)
        full = _unchunked(full_hidden, ids, full_head, temperature, selective_log_softmax)
        got = chunked_token_logps(hidden, ids, chunk_head, temperature, chunk_size)
        if not _close(torch, full, got, dtype):
            delta = (full.float() - got.float()).abs().max().item()
            raise SystemExit(f"{dtype} chunk_size={chunk_size} values differ, max abs {delta}")
        full.sum().backward()
        got.sum().backward()
        if not _close(torch, full_hidden.grad, hidden.grad, dtype):
            delta = (full_hidden.grad.float() - hidden.grad.float()).abs().max().item()
            raise SystemExit(f"{dtype} chunk_size={chunk_size} hidden grad differs, max abs {delta}")
        if not _close(torch, full_head.weight.grad, chunk_head.weight.grad, dtype):
            delta = (full_head.weight.grad.float() - chunk_head.weight.grad.float()).abs().max().item()
            raise SystemExit(f"{dtype} chunk_size={chunk_size} weight grad differs, max abs {delta}")


def _check_no_grad_chunks(torch, nn, chunked_token_logps, selective_log_softmax):
    torch.manual_seed(1)
    batch, time, width, vocab = 2, 9, 4, 16
    temperature = 1.7
    hidden = torch.randn(batch, time, width)
    ids = torch.randint(0, vocab, (batch, time))
    head = nn.Linear(width, vocab, bias=False)
    seen: list[int] = []
    original = head.forward

    def counted(rows):
        seen.append(int(rows.shape[0]))
        return original(rows)

    head.forward = counted
    with torch.no_grad():
        got = chunked_token_logps(hidden, ids, head, temperature, chunk_size=4)
    head.forward = original
    full = _unchunked(hidden, ids, head, temperature, selective_log_softmax)
    if not torch.equal(full, got):
        raise SystemExit("no-grad chunked log-probs differ from the full matmul")
    if seen != [4, 4, 4, 4, 2]:
        # 2 sequences * 9 tokens = 18 rows, chunks of 4 -> 4,4,4,4,2
        raise SystemExit(f"lm_head saw row batches {seen}, expected chunks of at most 4")
    if max(seen) > 4:
        raise SystemExit(f"a chunk materialized more than 4 token rows: {seen}")


def _check_edges(torch, nn, chunked_token_logps):
    head = nn.Linear(4, 6, bias=False)
    hidden = torch.randn(2, 0, 4)
    ids = torch.zeros(2, 0, dtype=torch.long)
    empty = chunked_token_logps(hidden, ids, head, 0.8, 256)
    if tuple(empty.shape) != (2, 0):
        raise SystemExit(f"empty window shape {tuple(empty.shape)}")
    try:
        chunked_token_logps(torch.randn(2, 3, 4), torch.zeros(2, 3, dtype=torch.long), head, 0.8, 0)
    except ValueError as exc:
        if "logprob_chunk_size" not in str(exc):
            raise
    else:
        raise SystemExit("chunk_size 0 did not raise")


if __name__ == "__main__":
    main()
