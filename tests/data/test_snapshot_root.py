import importlib
from pathlib import Path

import pytest

from fdagym.openfda.store import (
    SNAPSHOT_ENV,
    SNAPSHOT_VERSION,
    DiskSnapshot,
    discover_snapshot_root,
    packaged_hub_snapshot_root,
    resolve_snapshot_root,
)


def test_constructor_root_wins_over_env(tmp_path, monkeypatch):
    monkeypatch.setenv(SNAPSHOT_ENV, str(tmp_path / "from-env"))
    snap = DiskSnapshot("2026-10-15", root=tmp_path)
    assert snap.dir == tmp_path / "2026-10-15"


def test_env_var_sets_snapshot_root(tmp_path, monkeypatch):
    monkeypatch.setenv(SNAPSHOT_ENV, str(tmp_path))
    snap = DiskSnapshot("2026-10-15")
    assert snap.dir == tmp_path / "2026-10-15"


def test_missing_snapshot_root_names_the_env_var_and_helper(tmp_path, monkeypatch):
    monkeypatch.delenv(SNAPSHOT_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    if packaged_hub_snapshot_root() is not None:
        pytest.skip("fda-gym Hub package is installed in this environment")
    with pytest.raises(FileNotFoundError, match="python -m fdagym.openfda"):
        resolve_snapshot_root()
    assert discover_snapshot_root() is None


def test_checkout_data_snapshots_is_discovered_from_cwd(monkeypatch):
    monkeypatch.delenv(SNAPSHOT_ENV, raising=False)
    monkeypatch.chdir(Path(__file__).resolve().parents[2])
    root = discover_snapshot_root()
    assert root is not None
    assert (root / "2026-10-15").is_dir()


def test_hub_wheel_freeze_is_discovered_when_cwd_has_no_checkout(tmp_path, monkeypatch):
    site = tmp_path / "site"
    freeze = site / "fda_gym" / "data" / "snapshots" / SNAPSHOT_VERSION
    freeze.mkdir(parents=True)
    (site / "fda_gym" / "__init__.py").write_text("")
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    monkeypatch.delenv(SNAPSHOT_ENV, raising=False)
    monkeypatch.syspath_prepend(str(site))
    monkeypatch.chdir(cwd)
    importlib.invalidate_caches()
    root = packaged_hub_snapshot_root()
    assert root == freeze.parent
    assert discover_snapshot_root() == freeze.parent
