"""Fail when LiteLLM's dependency pin or locked artifacts drift from the approved release."""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

APPROVED_VERSION = "1.106.2"
APPROVED_REQUIREMENT = f"litellm=={APPROVED_VERSION}"
APPROVED_HASHES = frozenset(
    {
        "sha256:9f0e770cb007c8c6e78917ad7b5c4f059f7bdc4a3606f854831780ea53fd9e2d",
        "sha256:bf6f13445bb59d8d02131bbc1fb57161d1ae629bec65dc66f9add4c09fb6994a",
        "sha256:cc778d01ed65a8dd57d5211d581ec237e06df239a09c3d573ace33e0feea76c9",
        "sha256:34bfdc26c2c075726eda59ec0675e2a8a67b6e59a03980def7ef7651b849141a",
        "sha256:cb1a2a1e55a4a8350ae2167525dc0d06f59b078cd685fb3e0f5419f02da5420e",
        "sha256:ce18971e2bb2aa58ef8efae1cd12da1248e1c2e5ae3cf5903bfb9d563bb34d9d",
        "sha256:32e567ee53c812e36f79158ec280c8850973f84a9e5fdef50d31ca09306d148c",
        "sha256:836ab19110504ad7ba71d8ce73af6b9d1cf1dc6ffda2b49ebc1a51e348f754bc",
    }
)
_IMAGE = re.compile(
    r"(?:ghcr\.io/berriai/litellm|(?:docker\.io/)?litellm/litellm)"
    r"(?::(?P<tag>[\w.-]+))?(?:@(?P<digest>sha256:[a-f0-9]{64}))?"
)
_BAD_TAGS = frozenset({"latest", "main-latest", "main-stable", "1.107.0"})
_SKIP_DIRS = {".git", ".venv", ".next", "node_modules", "results", "__pycache__"}


def check_dependency(pyproject_path: Path, lock_path: Path) -> list[str]:
    project = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    dependencies = project.get("project", {}).get("dependencies", [])
    requirements = [item for item in dependencies if item.lower().startswith("litellm")]
    errors = []
    if requirements != [APPROVED_REQUIREMENT]:
        errors.append(f"pyproject.toml must contain exactly {APPROVED_REQUIREMENT!r}; found {requirements!r}")

    lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    packages = [package for package in lock.get("package", []) if package.get("name") == "litellm"]
    if len(packages) != 1:
        errors.append(f"uv.lock must contain exactly one litellm package; found {len(packages)}")
        return errors
    package = packages[0]
    if package.get("version") != APPROVED_VERSION:
        errors.append(f"uv.lock resolves litellm=={package.get('version')}; expected {APPROVED_VERSION}")
    if package.get("source", {}).get("registry") != "https://pypi.org/simple":
        errors.append("uv.lock must resolve LiteLLM from the official PyPI index")
    artifacts = [package.get("sdist", {}), *package.get("wheels", [])]
    hashes = {artifact.get("hash") for artifact in artifacts}
    if hashes != APPROVED_HASHES:
        errors.append("uv.lock LiteLLM artifact hashes differ from the approved v1.106.2 PyPI files")
    if any(not artifact.get("url", "").startswith("https://files.pythonhosted.org/") for artifact in artifacts):
        errors.append("uv.lock LiteLLM artifacts must come from files.pythonhosted.org")
    return errors


def check_container_tags(root: Path) -> list[str]:
    errors = []
    for path in root.rglob("*"):
        if not path.is_file() or any(part in _SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if path.suffix not in {".yml", ".yaml", ".toml", ".json", ".sh"} and not path.name.startswith("Dockerfile"):
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for line_number, line in enumerate(content.splitlines(), 1):
            for match in _IMAGE.finditer(line):
                tag, digest = match.group("tag"), match.group("digest")
                if tag in _BAD_TAGS or (tag is None and digest is None):
                    errors.append(f"{path.relative_to(root)}:{line_number}: LiteLLM image needs a safe explicit tag")
    return errors


def check_repository(root: Path) -> list[str]:
    errors = check_dependency(root / "pyproject.toml", root / "uv.lock")
    errors.extend(check_container_tags(root))
    return errors


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    errors = check_repository(root)
    if errors:
        print("LiteLLM security guard failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"LiteLLM security guard passed: {APPROVED_REQUIREMENT}; pinned artifact hashes verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
