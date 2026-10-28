import shutil
from pathlib import Path

from scripts.check_litellm_security import (
    APPROVED_HASHES,
    APPROVED_VERSION,
    check_container_tags,
    check_dependency,
    check_repository,
)

ROOT = Path(__file__).resolve().parents[2]


def test_current_dependency_pin_lock_hashes_and_repo_images_are_approved():
    assert check_repository(ROOT) == []


def test_guard_rejects_unpinned_dependency_and_tampered_artifact_hashes(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    lock = tmp_path / "uv.lock"
    shutil.copy(ROOT / "pyproject.toml", pyproject)
    shutil.copy(ROOT / "uv.lock", lock)

    pyproject.write_text(pyproject.read_text().replace("litellm==1.106.2", "litellm>=1.106.2"))
    errors = check_dependency(pyproject, lock)
    assert any("pyproject.toml must contain exactly" in error for error in errors)

    pyproject.write_text((ROOT / "pyproject.toml").read_text())
    expected_hash = min(APPROVED_HASHES)
    lock.write_text(lock.read_text().replace(expected_hash, "sha256:" + "0" * 64, 1))
    errors = check_dependency(pyproject, lock)
    assert any("artifact hashes differ" in error for error in errors)


def test_guard_rejects_floating_and_incident_docker_tags(tmp_path):
    (tmp_path / "compose.yml").write_text(
        "services:\n  proxy:\n    image: ghcr.io/berriai/litellm:latest\n"
        "  other:\n    image: litellm/litellm:1.107.0\n"
        f"  safe:\n    image: ghcr.io/berriai/litellm:v{APPROVED_VERSION}\n"
    )
    errors = check_container_tags(tmp_path)
    assert len(errors) == 2
    assert all("safe explicit tag" in error for error in errors)
