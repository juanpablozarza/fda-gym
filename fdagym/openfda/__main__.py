"""Print how to point FDA-Gym at the 2026-10-15 snapshot. No network calls."""

from __future__ import annotations

from fdagym.openfda.store import SNAPSHOT_MISSING, discover_snapshot_root


def main() -> int:
    found = discover_snapshot_root()
    if found is not None:
        print(f"snapshot directory: {found}")
        freeze = found / "2026-10-15"
        print(f"2026-10-15 freeze: {'present' if freeze.is_dir() else 'missing'}")
        return 0
    print(SNAPSHOT_MISSING)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
