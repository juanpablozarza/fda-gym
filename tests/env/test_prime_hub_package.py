"""Hub package wheel ships public.jsonl + freeze 2026-10-15 and nothing else."""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from fdagym.demo import DemoError, load_public_tasks
from fdagym.openfda.store import SNAPSHOT_ENV, SNAPSHOT_VERSION, discover_snapshot_root
from fdagym.prime.episode import (
    PUBLIC_ENV,
    discover_public_path,
    make_public_env,
    play_public_adapter_episode,
)
from fdagym.rewards.version import GRADER_VERSION

ROOT = Path(__file__).resolve().parents[2]
HUB = ROOT / "environments" / "fda-gym"
PUBLIC = ROOT / "data" / "tasks" / "public.jsonl"
LEAK_MARKERS = ("hard_train", "hard_train_holdout", "hard.jsonl", "bio-rl")


def _isolate(monkeypatch, tmp_path: Path, site: Path) -> Path:
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.delenv(SNAPSHOT_ENV, raising=False)
    monkeypatch.delenv(PUBLIC_ENV, raising=False)
    monkeypatch.syspath_prepend(str(site))
    monkeypatch.chdir(cwd)
    importlib.invalidate_caches()
    sys.modules.pop("fda_gym", None)
    return cwd


def test_hub_package_is_v1_and_depends_on_fdagym_via_pep508():
    text = (HUB / "pyproject.toml").read_text()
    assert 'name = "fda-gym"' in text
    assert "verifiers==0.4.0" in text
    assert "fdagym @ git+https://github.com/juanpablozarza/fda-gym.git" in text
    assert "allow-direct-references" in text
    assert "[tool.uv.sources]" not in text
    init = (HUB / "fda_gym" / "__init__.py").read_text()
    assert "FDAGymTaskset" in init
    assert "load_environment" not in init
    assert "load_environment" not in text


def test_packaged_hub_data_is_discovered_without_a_checkout(tmp_path, monkeypatch):
    site = tmp_path / "site"
    pkg = site / "fda_gym"
    freeze = pkg / "data" / "snapshots" / SNAPSHOT_VERSION
    freeze.mkdir(parents=True)
    (freeze / "manifest.jsonl").write_text("")
    tasks = pkg / "data" / "tasks"
    tasks.mkdir(parents=True)
    shutil.copy(PUBLIC, tasks / "public.jsonl")
    (pkg / "__init__.py").write_text("")
    _isolate(monkeypatch, tmp_path, site)

    snap = discover_snapshot_root()
    public = discover_public_path()
    assert snap == freeze.parent
    assert public == tasks / "public.jsonl"
    rows = load_public_tasks(public)
    ids = sorted(rows)
    assert ids[0] == "FG-9101"
    assert ids[-1] == "FG-9193"
    assert len(ids) == 93
    with pytest.raises(DemoError, match="published public sheet"):
        make_public_env("FG-0001", public_path=public, snapshot_root=snap)
    with pytest.raises(DemoError, match="published public sheet"):
        make_public_env("FG-9194", public_path=public, snapshot_root=snap)


def test_hub_wheel_install_resolves_public_freeze_and_refuses_nonpublic_ids(tmp_path, monkeypatch):
    dist = tmp_path / "dist"
    built = subprocess.run(
        [
            "uv",
            "build",
            "--wheel",
            "--out-dir",
            str(dist),
            "--directory",
            str(HUB),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert built.returncode == 0, built.stdout + built.stderr
    wheels = list(dist.glob("*.whl"))
    assert len(wheels) == 1, wheels
    with zipfile.ZipFile(wheels[0]) as archive:
        names = archive.namelist()
        metadata = next(name for name in names if name.endswith("METADATA"))
        requires = [line for line in archive.read(metadata).decode().splitlines() if line.startswith("Requires-Dist:")]
    assert any("fdagym @ git+https://github.com/juanpablozarza/fda-gym.git" in line for line in requires)
    assert any("verifiers==0.4.0" in line for line in requires)
    assert "fda_gym/data/tasks/public.jsonl" in names
    assert any(
        name.startswith(f"fda_gym/data/snapshots/{SNAPSHOT_VERSION}/") and name.endswith("manifest.jsonl")
        for name in names
    )
    leaked = [name for name in names if any(marker in name for marker in LEAK_MARKERS)]
    assert leaked == []

    site = tmp_path / "site"
    with zipfile.ZipFile(wheels[0]) as archive:
        archive.extractall(site)
    _isolate(monkeypatch, tmp_path, site)

    public = discover_public_path()
    snap = discover_snapshot_root()
    assert public is not None and public.is_file()
    assert snap is not None and (snap / SNAPSHOT_VERSION / "manifest.jsonl").is_file()
    assert "fda_gym" in public.parts
    assert "fda_gym" in snap.parts
    rows = load_public_tasks(public)
    assert sorted(rows) == [f"FG-{n}" for n in range(9101, 9194)]
    with pytest.raises(DemoError, match="published public sheet"):
        make_public_env("FG-0001")
    with pytest.raises(DemoError, match="published public sheet"):
        make_public_env("FG-9194")
    reward = play_public_adapter_episode("FG-9101")
    assert reward["grader_version"] == GRADER_VERSION == "5"
    assert reward["total"] == 1.0
