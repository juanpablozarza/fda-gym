"""The interface between the data track and everything that reads openFDA data.

The data track implements `Snapshot`. The task generator and the env only depend on this
protocol, so their tests can use an in-memory fake built from recorded responses.
"""

from typing import Literal, Protocol

Endpoint = Literal["drug/label", "drug/event", "drug/drugsfda"]


class SnapshotMiss(KeyError):
    """The query isn't in the frozen snapshot. Offline code must never fall back to the live API."""


class Snapshot(Protocol):
    version: str

    def get(
        self,
        endpoint: Endpoint,
        *,
        search: str | None = None,
        count: str | None = None,
        limit: int = 1,
        skip: int = 0,
    ) -> dict:
        """Return the openFDA JSON body for this query, with records trimmed to the fields we use
        (see fdagym/openfda/trim.py). `meta` is kept intact. Count responses have no meta.results.total.

        A query with no matches returns {"meta": {...}, "results": []} (openFDA itself answers 404).
        Raises SnapshotMiss if the query was never recorded.
        """
        ...
