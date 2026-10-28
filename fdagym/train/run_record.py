"""Portable metadata for a run, independent of its GPU provider."""

from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path

RECORD_NAME = "run.json"
RECORD_VERSION = 1
_PACKAGES = ("torch", "transformers", "trl", "peft", "vllm", "datasets")
_ACTIVE_LOCKS: dict[Path, object] = {}
_ACTIVE_LOCKS_GUARD = threading.Lock()


class RunConflictError(RuntimeError):
    """The requested job or config conflicts with the existing run identity."""


def validate_run_id(run_id: str) -> None:
    if not run_id or run_id in {".", ".."} or "/" in run_id or "\\" in run_id:
        raise ValueError(f"run_id must be one path segment, got {run_id!r}")


def start_run(
    runs_dir: Path | str,
    run_id: str,
    *,
    job: str,
    protocol: str,
    config: dict,
    resume: bool = False,
) -> Path:
    """Start an attempt, retaining its history and refusing incompatible resumes."""
    validate_run_id(run_id)

    run_dir = Path(runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = (run_dir / ".run.lock").resolve()
    _acquire_lock(lock_path, run_id)
    try:
        record_path = run_dir / RECORD_NAME
        digest = _config_digest(config)
        previous = _read(record_path)
        if previous:
            if previous.get("job") != job or previous.get("protocol") != protocol:
                raise RunConflictError(
                    f"run {run_id!r} belongs to {previous.get('job')!r}/{previous.get('protocol')!r}; "
                    f"requested {job!r}/{protocol!r}. Choose a new run_id."
                )
            if resume and previous.get("config_sha256") != digest:
                raise RunConflictError(
                    f"run {run_id!r} config differs from its recorded config; use a new run_id or start with resume=False"
                )

        for marker in ("DONE", "FAILED"):
            (run_dir / marker).unlink(missing_ok=True)

        now = _now()
        environment = _environment()
        source_revision = os.environ.get("FDAGYM_GIT_SHA", "")
        attempts = previous.get("attempts", []) if previous else []
        attempts.append(
            {
                "started_at": now,
                "finished_at": None,
                "status": "running",
                "source_revision": source_revision,
                "environment": environment,
                "config": config,
                "config_sha256": digest,
            }
        )
        record = {
            "record_version": RECORD_VERSION,
            "run_id": run_id,
            "job": job,
            "protocol": protocol,
            "status": "running",
            "config": config,
            "config_sha256": digest,
            "source_revision": source_revision,
            "environment": environment,
            "attempts": attempts,
        }
        _write(record_path, record)
    except BaseException:
        _release_lock(lock_path)
        raise
    return run_dir


def finish_run(run_dir: Path, status: str, error: str | None = None) -> None:
    """Record the final state for the most recent attempt."""
    path = Path(run_dir) / RECORD_NAME
    lock_path = (Path(run_dir) / ".run.lock").resolve()
    held = lock_path in _ACTIVE_LOCKS
    if not held:
        _acquire_lock(lock_path, Path(run_dir).name)
    try:
        record = _read(path)
        if not record:
            return
        now = _now()
        record["status"] = status
        record["environment"] = _environment()
        if record.get("attempts"):
            latest = record["attempts"][-1]
            latest["status"] = status
            latest["finished_at"] = now
            if error:
                latest["error"] = error
        _write(path, record)
    finally:
        _release_lock(lock_path)


def _config_digest(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _executor() -> str:
    return os.environ.get("FDAGYM_EXECUTOR", "unknown")


def _environment() -> dict:
    packages = {}
    for package in _PACKAGES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    info = {
        "executor": _executor(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
    }
    torch = sys.modules.get("torch")
    if torch is not None:
        info["cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
        try:
            if torch.cuda.is_available():
                info["accelerator"] = torch.cuda.get_device_name(0)
        except (RuntimeError, OSError):
            pass
    return info


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _read(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunConflictError(f"cannot read run record {path}: {type(exc).__name__}") from exc
    if not isinstance(value, dict):
        raise RunConflictError(f"run record {path} must contain a JSON object")
    return value


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _acquire_lock(lock_path: Path, run_id: str) -> None:
    with _ACTIVE_LOCKS_GUARD:
        if lock_path in _ACTIVE_LOCKS:
            raise RunConflictError(f"run {run_id!r} already has an active writer in this process")
        lock_file = lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            lock_file.close()
            raise RunConflictError(f"run {run_id!r} already has an active writer") from exc
        _ACTIVE_LOCKS[lock_path] = lock_file


def _release_lock(lock_path: Path) -> None:
    with _ACTIVE_LOCKS_GUARD:
        lock_file = _ACTIVE_LOCKS.pop(lock_path, None)
    if lock_file is not None:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()
