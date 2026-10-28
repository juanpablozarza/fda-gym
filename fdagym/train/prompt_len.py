"""Prompt length vs vLLM context. Torch-free so dry-run can measure before a GPU job.

TRL 0.28 dropped ``max_prompt_length`` from GRPOConfig. The window is the longest
prompt actually rendered for the training or eval pool, plus the completion,
rounded up and capped at 8192. Budget tokens with 3 chars/token (Qwen BPE on JSON
is typically denser than prose; this overestimates tokens). A prompt past the cap
fails at launch instead of being truncated.
"""

from __future__ import annotations

from dataclasses import replace

from fdagym.train.config import TrainConfig

CHARS_PER_TOKEN = 3
CONTEXT_CAP = 8192


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN)


def messages_text(messages: list[dict]) -> str:
    return "\n".join(str(item.get("content") or "") for item in messages)


def prompt_tokens(messages: list[dict]) -> int:
    return estimate_tokens(messages_text(messages))


def max_prompt_tokens(rows: list[dict]) -> int:
    if not rows:
        return 0
    return max(prompt_tokens(row["prompt"]) for row in rows)


def _ceil_context(needed: int) -> int:
    """Next power of two, at least 4096, never above ``CONTEXT_CAP``."""
    size = 4096
    while size < needed and size < CONTEXT_CAP:
        size *= 2
    return min(size, CONTEXT_CAP)


def prompt_token_limit(config: TrainConfig) -> int:
    """Tokens a rendered prompt may use. Completion must still fit in the vLLM window."""
    room = config.vllm_max_model_len - config.max_completion_length
    if room < 1:
        raise RuntimeError(
            f"vllm_max_model_len={config.vllm_max_model_len} is smaller than "
            f"max_completion_length={config.max_completion_length}"
        )
    if config.max_prompt_length is None:
        return room
    return min(config.max_prompt_length, room)


def assert_prompts_fit(rows: list[dict], config: TrainConfig) -> None:
    """Refuse to launch if TRL or vLLM would truncate a prompt and drop the question."""
    limit = prompt_token_limit(config)
    too_long = []
    for row in rows:
        tokens = prompt_tokens(row["prompt"])
        if tokens > limit:
            too_long.append((row.get("task_id", "?"), tokens))
    if not too_long:
        return
    too_long.sort(key=lambda item: item[1], reverse=True)
    sample = ", ".join(f"{task_id}={tokens}" for task_id, tokens in too_long[:8])
    raise RuntimeError(
        f"{len(too_long)} rendered prompt(s) exceed the launch limit of {limit} tokens "
        f"(chars/3 estimate; max_prompt_length={config.max_prompt_length}, "
        f"vllm_max_model_len={config.vllm_max_model_len}, "
        f"max_completion_length={config.max_completion_length}). "
        f"Train and eval share these prompts. Refusing to launch so the question is not dropped. "
        f"Longest: {sample}."
    )


def fit_context(config: TrainConfig, rows: list[dict] | None = None) -> tuple[TrainConfig, dict]:
    """Size ``vllm_max_model_len`` from the longest prompt in ``rows``.

    The window is that measurement plus ``max_completion_length``, rounded up to
    the next power of two and capped at 8192. ``rows`` is the training or eval
    pool that will actually be sent. The theoretical 4-document maximum is not
    consulted, and ``max_pack_chars`` is left alone. With no rows there is nothing
    to measure, so the configured window stays. A prompt that still exceeds the
    cap fails in ``assert_prompts_fit``.
    """
    measured = max_prompt_tokens(rows) if rows else 0
    if not rows:
        return config, _length_stats(config, measured, fits=True)
    needed = measured + config.max_completion_length
    trial = replace(config, vllm_max_model_len=_ceil_context(needed))
    fits = measured <= prompt_token_limit(trial)
    return trial, _length_stats(trial, measured, fits=fits)


def _length_stats(config: TrainConfig, measured: int, *, fits: bool) -> dict:
    return {
        "measured_prompt_tokens": measured,
        "vllm_max_model_len": config.vllm_max_model_len,
        "max_pack_chars": config.max_pack_chars,
        "max_completion_length": config.max_completion_length,
        "fits": fits,
    }
