"""Run one frozen-snapshot tool call and print its JSON. No task file is read."""

import json
import sys

import fdagym  # noqa: F401
from fdagym.openfda import DiskSnapshot
from fdagym.tools import EpisodeState, execute


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print("usage: poke_tool.py <tool> <json-arguments>", file=sys.stderr)
        return 2
    name, raw = argv
    arguments = json.loads(raw)
    output, error = execute(EpisodeState(DiskSnapshot("2026-10-15")), name, arguments, step_index=1)
    json.dump({"output": output, "error": error}, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
