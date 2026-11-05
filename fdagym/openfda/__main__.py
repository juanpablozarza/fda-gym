"""Point FDA-Gym at the 2026-10-15 snapshot, or verify stored bodies. No network calls."""

from __future__ import annotations

import argparse
import sys

from fdagym.openfda.store import SNAPSHOT_MISSING, SNAPSHOT_VERSION, DiskSnapshot, discover_snapshot_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("verify",), default=None)
    parser.add_argument("--version", default=SNAPSHOT_VERSION)
    args = parser.parse_args(argv)
    found = discover_snapshot_root()
    if found is None:
        print(SNAPSHOT_MISSING, file=sys.stderr)
        return 1
    print(f"snapshot directory: {found}")
    freeze = found / args.version
    print(f"{args.version} freeze: {'present' if freeze.is_dir() else 'missing'}")
    if args.command == "verify":
        mismatches = DiskSnapshot(args.version, found).verify()
        if mismatches:
            print(f"{len(mismatches)} body sha256 mismatch(es)", file=sys.stderr)
            for item in mismatches[:20]:
                print(item, file=sys.stderr)
            return 1
        print("manifest body sha256 ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
