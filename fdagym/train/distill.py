"""Load, filter, and nest teacher-trajectory JSONL for distill SFT.

No torch. The generator writes the JSONL; this module only reads ``passed``
from JP's already-computed reward. It does not import ``fdagym.rewards``.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from random import Random

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from fdagym.train.config import QWEN_3B
from fdagym.train.rollout import DEFAULT_PROMPT_VERSION, DEFAULT_TOOL_BUDGET, PROMPT_VERSIONS
from fdagym.train.sft_config import SFTConfig

_ROOT = Path(__file__).resolve().parents[2]
_SCHEMAS = _ROOT / "schemas"
_FIXTURE = _ROOT / "tests" / "train" / "fixtures" / "teacher_trajectories.jsonl"
DEFAULT_SIZES = ("100", "250", "500", "1000", "all")
DEFAULT_RUN_ID = "distill-sft-qwen3b"


def _load_schema(name: str) -> dict:
    return json.loads((_SCHEMAS / name).read_text(encoding="utf-8"))


def teacher_trajectory_validator() -> Draft202012Validator:
    names = ("answer.schema.json", "trajectory.schema.json", "teacher_trajectory.schema.json")
    schemas = [_load_schema(name) for name in names]
    registry = Registry().with_resources((schema["$id"], Resource.from_contents(schema)) for schema in schemas)
    return Draft202012Validator(_load_schema("teacher_trajectory.schema.json"), registry=registry)


_VALIDATOR = teacher_trajectory_validator()


@dataclass(frozen=True)
class TeacherTrajectory:
    """One generator row after schema validation."""

    id: str
    question_id: str
    template: str
    variant: str
    teacher_model: str
    attempt_index: int
    messages: tuple[dict, ...]
    reward: dict
    passed: bool
    prompt_version: str
    tool_budget: int
    raw: dict

    @property
    def dedupe_key(self) -> tuple[str, str, int]:
        return (self.question_id, self.teacher_model, self.attempt_index)


@dataclass(frozen=True)
class DistillConfig:
    """Warm-up SFT knobs plus the learning-curve sampler.

    LoRA, lr, batch, and epoch defaults copy ``SFTConfig``. There is no drug
    hold-out: the curve's eval is ``eval-tools``, not teacher-forced val.
    ``max_steps`` 0 means train for ``num_train_epochs`` (the warm-up default).
    """

    model: str = QWEN_3B
    learning_rate: float = 1e-4
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    max_length: int = 8192
    warmup_ratio: float = 0.03
    bf16: bool = True
    gradient_checkpointing: bool = True
    logging_steps: int = 1
    save_steps: int = 50
    save_total_limit: int = 2
    seed: int = 0
    tool_budget: int = DEFAULT_TOOL_BUDGET
    prompt_version: str = DEFAULT_PROMPT_VERSION
    max_steps: int = 0
    init_adapter: str | None = None
    sizes: tuple[str, ...] = DEFAULT_SIZES
    teachers: tuple[str, ...] = ()
    templates: tuple[str, ...] = ()
    max_attempts_per_question: int | None = None
    eval_after: bool = False
    eval_split: str = "hard"
    eval_limit: int | None = None
    run_id: str = DEFAULT_RUN_ID

    def __post_init__(self) -> None:
        object.__setattr__(self, "teachers", tuple(self.teachers or ()))
        object.__setattr__(self, "templates", tuple(self.templates or ()))
        object.__setattr__(self, "sizes", tuple(parse_sizes(self.sizes)))
        init = self.init_adapter.strip() if isinstance(self.init_adapter, str) else self.init_adapter
        object.__setattr__(self, "init_adapter", init or None)
        if self.max_attempts_per_question is not None and self.max_attempts_per_question < 1:
            raise ValueError("max_attempts_per_question must be >= 1")
        if self.max_steps < 0:
            raise ValueError("max_steps must be >= 0")
        if self.prompt_version not in PROMPT_VERSIONS:
            raise ValueError(f"prompt_version must be one of {', '.join(PROMPT_VERSIONS)}")

    def sft_config(self) -> SFTConfig:
        return SFTConfig(
            model=self.model,
            learning_rate=self.learning_rate,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            max_length=self.max_length,
            warmup_ratio=self.warmup_ratio,
            bf16=self.bf16,
            gradient_checkpointing=self.gradient_checkpointing,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            save_total_limit=self.save_total_limit,
            seed=self.seed,
            tool_budget=self.tool_budget,
            prompt_version=self.prompt_version,
            val_drug_fraction=0.0,
        )

    def hyperparameters(self) -> dict:
        return {
            "model": self.model,
            "learning_rate": self.learning_rate,
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_length": self.max_length,
            "warmup_ratio": self.warmup_ratio,
            "bf16": self.bf16,
            "gradient_checkpointing": self.gradient_checkpointing,
            "save_steps": self.save_steps,
            "seed": self.seed,
            "tool_budget": self.tool_budget,
            "prompt_version": self.prompt_version,
            "max_steps": self.max_steps or None,
            "lora": self.sft_config().lora_kwargs(),
        }


def parse_sizes(raw) -> list[str]:
    """``100,250,all`` or an iterable of those tokens. Order is kept."""
    if raw is None or raw == "":
        tokens = list(DEFAULT_SIZES)
    elif isinstance(raw, str):
        tokens = [part.strip() for part in raw.split(",") if part.strip()]
    else:
        tokens = [str(part).strip() for part in raw if str(part).strip()]
    if not tokens:
        raise ValueError("sizes is empty")
    parsed = []
    seen: set[str] = set()
    for token in tokens:
        name = _size_name(token)
        if name in seen:
            continue
        seen.add(name)
        parsed.append(name)
    return parsed


def _size_name(token: str) -> str:
    lowered = token.lower()
    if lowered == "all":
        return "all"
    if not token.isdigit() or int(token) < 1:
        raise ValueError(f"size must be a positive integer or 'all', got {token!r}")
    return str(int(token))


def size_n(name: str, n_pool: int) -> int:
    if name == "all":
        return n_pool
    return min(int(name), n_pool)


def fixture_path() -> Path:
    return _FIXTURE


def parse_teacher_trajectory(row: dict) -> TeacherTrajectory:
    _VALIDATOR.validate(row)
    messages = tuple(dict(message) for message in row["messages"])
    if not any(message.get("role") == "user" for message in messages):
        raise ValueError(f"{row.get('id')}: messages need a user turn")
    if not any(message.get("role") == "assistant" for message in messages):
        raise ValueError(f"{row.get('id')}: messages need an assistant turn")
    prompt_version = row.get("prompt_version") or DEFAULT_PROMPT_VERSION
    if prompt_version not in PROMPT_VERSIONS:
        raise ValueError(f"{row.get('id')}: unknown prompt_version {prompt_version!r}")
    budget = row.get("tool_budget", DEFAULT_TOOL_BUDGET)
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise ValueError(f"{row.get('id')}: tool_budget must be an integer >= 1")
    return TeacherTrajectory(
        id=str(row["id"]),
        question_id=str(row["question_id"]),
        template=str(row["template"]),
        variant="" if row.get("variant") is None else str(row["variant"]),
        teacher_model=str(row["teacher_model"]),
        attempt_index=int(row["attempt_index"]),
        messages=messages,
        reward=dict(row["reward"]),
        passed=bool(row["passed"]),
        prompt_version=str(prompt_version),
        tool_budget=int(budget),
        raw=row,
    )


def load_teacher_jsonl(paths: list[Path] | Path) -> list[TeacherTrajectory]:
    if isinstance(paths, (str, Path)):
        paths = [Path(paths)]
    rows: list[TeacherTrajectory] = []
    for path in paths:
        text = Path(path).read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({exc.msg})") from exc
            try:
                rows.append(parse_teacher_trajectory(payload))
            except Exception as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return rows


def filter_trajectories(
    rows: list[TeacherTrajectory],
    *,
    teachers: tuple[str, ...] | list[str] = (),
    templates: tuple[str, ...] | list[str] = (),
    max_attempts_per_question: int | None = None,
    prompt_version: str | None = None,
    tool_budget: int | None = None,
    passing_only: bool = True,
) -> tuple[list[TeacherTrajectory], dict]:
    """Keep passing rows, optional teacher/template filters, then dedupe and cap attempts."""
    skipped: Counter[str] = Counter()
    kept: list[TeacherTrajectory] = []
    teacher_set = {item for item in teachers if item}
    template_set = {item for item in templates if item}
    for row in rows:
        if passing_only and not row.passed:
            skipped["failed"] += 1
            continue
        if teacher_set and row.teacher_model not in teacher_set:
            skipped["teacher"] += 1
            continue
        if template_set and row.template not in template_set:
            skipped["template"] += 1
            continue
        if prompt_version and row.prompt_version != prompt_version:
            skipped["prompt_version"] += 1
            continue
        if tool_budget is not None and row.tool_budget != tool_budget:
            skipped["tool_budget"] += 1
            continue
        kept.append(row)
    unique, n_dupes = dedupe_trajectories(kept)
    skipped["duplicate"] += n_dupes
    capped, n_cap = cap_attempts(unique, max_attempts_per_question)
    skipped["max_attempts"] += n_cap
    stats = {
        "n_loaded": len(rows),
        "n_kept": len(capped),
        "n_skipped": dict(skipped),
        "n_questions": len({row.question_id for row in capped}),
        "per_template": dict(Counter(row.template for row in capped)),
        "teacher_mix": dict(Counter(row.teacher_model for row in capped)),
        "per_variant": dict(Counter(row.variant or row.template for row in capped)),
    }
    return capped, stats


def dedupe_trajectories(rows: list[TeacherTrajectory]) -> tuple[list[TeacherTrajectory], int]:
    """First occurrence of ``(question_id, teacher_model, attempt_index)`` wins."""
    seen: set[tuple[str, str, int]] = set()
    unique: list[TeacherTrajectory] = []
    n_dupes = 0
    for row in rows:
        key = row.dedupe_key
        if key in seen:
            n_dupes += 1
            continue
        seen.add(key)
        unique.append(row)
    return unique, n_dupes


def cap_attempts(
    rows: list[TeacherTrajectory], max_attempts_per_question: int | None
) -> tuple[list[TeacherTrajectory], int]:
    if max_attempts_per_question is None:
        return list(rows), 0
    counts: dict[str, int] = defaultdict(int)
    kept: list[TeacherTrajectory] = []
    dropped = 0
    ordered = sorted(rows, key=lambda row: (row.question_id, row.attempt_index, row.teacher_model, row.id))
    keep_ids = set()
    for row in ordered:
        if counts[row.question_id] >= max_attempts_per_question:
            dropped += 1
            continue
        counts[row.question_id] += 1
        keep_ids.add(row.dedupe_key)
    for row in rows:
        if row.dedupe_key in keep_ids:
            kept.append(row)
    return kept, dropped


def proportional_counts(weights: dict[str, int], n: int) -> dict[str, int]:
    """Largest-remainder allocation. Keys with weight 0 stay 0."""
    keys = sorted(weights)
    total = sum(max(0, int(weights[key])) for key in keys)
    if n <= 0 or total <= 0:
        return {key: 0 for key in keys}
    if n >= total:
        return {key: int(weights[key]) for key in keys}
    raw = {key: n * int(weights[key]) / total for key in keys}
    floors = {key: int(raw[key]) for key in keys}
    leftover = n - sum(floors.values())
    remainders = sorted(keys, key=lambda key: (-(raw[key] - floors[key]), key))
    out = dict(floors)
    for key in remainders[:leftover]:
        out[key] += 1
    return out


def unique_first_order(
    rows: list[TeacherTrajectory], seed: int
) -> dict[str, list[TeacherTrajectory]]:
    """Per template: one attempt per question (shuffled), then leftover attempts.

    Extra attempts are shuffled separately so a size that fits in unique
    questions never spends a slot on a repeat.
    """
    rng = Random(seed)
    by_template: dict[str, list[TeacherTrajectory]] = defaultdict(list)
    for row in rows:
        by_template[row.template].append(row)
    ordered: dict[str, list[TeacherTrajectory]] = {}
    for template in sorted(by_template):
        grouped: dict[str, list[TeacherTrajectory]] = defaultdict(list)
        for row in by_template[template]:
            grouped[row.question_id].append(row)
        questions = sorted(grouped)
        rng.shuffle(questions)
        first: list[TeacherTrajectory] = []
        extra: list[TeacherTrajectory] = []
        for question_id in questions:
            attempts = sorted(
                grouped[question_id],
                key=lambda row: (row.attempt_index, row.teacher_model, row.id),
            )
            first.append(attempts[0])
            extra.extend(attempts[1:])
        rng.shuffle(extra)
        ordered[template] = first + extra
    return ordered


def nested_subsets(
    rows: list[TeacherTrajectory],
    sizes,
    *,
    seed: int = 0,
) -> dict[str, list[TeacherTrajectory]]:
    """Nested, template-stratified prefixes. ``S_small`` is always a subset of ``S_large``.

    Quotas are allocated from the next-larger set (largest remainder), so a
    template's count is non-decreasing with size. Selection inside a template
    is the unique-first order for ``seed``.
    """
    names = parse_sizes(sizes)
    n_pool = len(rows)
    ordered = unique_first_order(rows, seed)
    templates = sorted(ordered)
    current = {template: len(ordered[template]) for template in templates}
    targets = sorted({size_n(name, n_pool) for name in names}, reverse=True)
    quotas_by_n: dict[int, dict[str, int]] = {}
    for target in targets:
        quota = proportional_counts(current, target)
        quotas_by_n[target] = quota
        current = quota
    subsets: dict[str, list[TeacherTrajectory]] = {}
    for name in names:
        target = size_n(name, n_pool)
        quota = quotas_by_n[target]
        chosen: list[TeacherTrajectory] = []
        for template in templates:
            chosen.extend(ordered[template][: quota[template]])
        subsets[name] = chosen
    return subsets


def subset_ids(rows: list[TeacherTrajectory]) -> list[str]:
    return [row.id for row in rows]


def question_ids(rows: list[TeacherTrajectory]) -> list[str]:
    return sorted({row.question_id for row in rows})


def per_template_counts(rows: list[TeacherTrajectory]) -> dict[str, int]:
    return dict(Counter(row.template for row in rows))


def teacher_mix(rows: list[TeacherTrajectory]) -> dict[str, int]:
    return dict(Counter(row.teacher_model for row in rows))


def data_hash(rows: list[TeacherTrajectory]) -> str:
    """SHA-256 of sorted ``id`` lines. Same rows, same hash, independent of list order."""
    payload = "\n".join(sorted(row.id for row in rows)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def subset_report(name: str, rows: list[TeacherTrajectory]) -> dict:
    return {
        "size": name,
        "n": len(rows),
        "n_questions": len({row.question_id for row in rows}),
        "per_template": per_template_counts(rows),
        "per_variant": dict(Counter(row.variant or row.template for row in rows)),
        "teacher_mix": teacher_mix(rows),
        "question_ids": question_ids(rows),
        "trajectory_ids": sorted(row.id for row in rows),
        "data_hash": data_hash(rows),
    }


def prepare_distill(
    paths: list[Path] | Path,
    config: DistillConfig | None = None,
) -> dict:
    """Load JSONL, filter, and build nested subsets. No tokenizer, no GPU."""
    config = config or DistillConfig()
    loaded = load_teacher_jsonl(paths)
    kept, filter_stats = filter_trajectories(
        loaded,
        teachers=config.teachers,
        templates=config.templates,
        max_attempts_per_question=config.max_attempts_per_question,
        prompt_version=config.prompt_version,
        tool_budget=config.tool_budget,
        passing_only=True,
    )
    subsets = nested_subsets(kept, config.sizes, seed=config.seed)
    reports = {name: subset_report(name, rows) for name, rows in subsets.items()}
    return {
        "config": config,
        "loaded": loaded,
        "kept": kept,
        "filter": filter_stats,
        "subsets": subsets,
        "reports": reports,
    }


def format_prepare_report(prepared: dict) -> str:
    filt = prepared["filter"]
    lines = [
        f"loaded={filt['n_loaded']} kept={filt['n_kept']} questions={filt['n_questions']} skipped={filt['n_skipped']}",
        f"pool_templates={filt['per_template']}",
        f"pool_teachers={filt['teacher_mix']}",
    ]
    for name, report in prepared["reports"].items():
        lines.append(
            f"size={name} n={report['n']} questions={report['n_questions']} "
            f"templates={report['per_template']} teachers={report['teacher_mix']} hash={report['data_hash'][:12]}"
        )
    return "\n".join(lines)


def size_run_id(prefix: str, size: str) -> str:
    return f"{prefix}-n{size}"
