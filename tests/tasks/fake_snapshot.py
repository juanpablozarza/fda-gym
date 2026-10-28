"""In-memory Snapshot over the trimmed fixtures in tests/tasks/fixtures/."""

import json
from pathlib import Path

from fdagym.snapshot import SnapshotMiss

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE_DRUGS = tuple(sorted(path.stem for path in FIXTURES.glob("*.json")))


class MemorySnapshot:
    def __init__(self, mapping: dict, version: str):
        self._mapping = mapping
        self.version = version

    def get(self, endpoint, *, search=None, count=None, limit=1, skip=0):
        key = (endpoint, search, count, limit)
        if key not in self._mapping:
            raise SnapshotMiss(key)
        return self._mapping[key]


FIXTURE_VERSION = "2026-10-15"


def load() -> MemorySnapshot:
    mapping = {}
    for path in sorted(FIXTURES.glob("*.json")):
        for record in json.loads(path.read_text()):
            query = record["query"]
            mapping[(query["endpoint"], query["search"], query["count"], query["limit"])] = record["response"]
    return MemorySnapshot(mapping, FIXTURE_VERSION)
