"""HTTP client for the three openFDA endpoints FDA-Gym uses.

Only the snapshot recorder talks to the network. Everything else reads through `DiskSnapshot`.
"""

import os
import time
from collections.abc import Callable

import httpx

from fdagym.snapshot import Endpoint

BASE_URL = "https://api.fda.gov"
KEYLESS_REQUESTS_PER_MINUTE = 240
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


class OpenFDAError(RuntimeError):
    def __init__(self, status: int, body: str, url: str):
        super().__init__(f"openFDA {status} for {url}: {body[:300]}")
        self.status = status
        self.body = body


def empty_response(limit: int, skip: int) -> dict:
    """What a query with no matches returns. openFDA itself answers 404 "No matches found!"."""
    return {"meta": {"results": {"skip": skip, "limit": limit, "total": 0}}, "results": []}


def _is_no_matches(response: httpx.Response) -> bool:
    try:
        return response.json()["error"]["code"] == "NOT_FOUND"
    except (ValueError, KeyError, TypeError):
        return False


def _is_daily_quota(response: httpx.Response) -> bool:
    """Keyless daily cap, not the per-minute 429. Retrying it only burns more of the same cap."""
    text = response.text.lower()
    return "daily" in text and "quota" in text


class RateLimiter:
    """Spaces calls evenly so a burst never exceeds `per_minute`."""

    def __init__(self, per_minute: int, *, clock: Callable[[], float], sleep: Callable[[float], None]):
        self.interval = 60.0 / per_minute
        self.clock = clock
        self.sleep = sleep
        self.next_at = 0.0

    def wait(self) -> None:
        now = self.clock()
        if now < self.next_at:
            self.sleep(self.next_at - now)
            now = self.next_at
        self.next_at = now + self.interval


class OpenFDAClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        requests_per_minute: int = KEYLESS_REQUESTS_PER_MINUTE,
        max_attempts: int = 4,
        backoff_base: float = 2.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.api_key = api_key if api_key is not None else os.environ.get("OPENFDA_API_KEY") or None
        self.max_attempts = max_attempts
        self.backoff_base = backoff_base
        self.sleep = sleep
        self.limiter = RateLimiter(requests_per_minute, clock=clock, sleep=sleep)
        # A full metformin label page is ~47 MB. 60s is tight on a slow link.
        self.http = httpx.Client(base_url=BASE_URL, transport=transport, timeout=httpx.Timeout(180.0, connect=15.0))
        self.requests_sent = 0
        self.last_raw_bytes = 0

    def get(
        self,
        endpoint: Endpoint,
        *,
        search: str | None = None,
        count: str | None = None,
        limit: int = 1,
        skip: int = 0,
    ) -> dict:
        params: dict[str, str | int] = {"limit": limit}
        if search is not None:
            params["search"] = search
        if count is not None:
            params["count"] = count
        if skip:
            params["skip"] = skip
        if self.api_key:
            params["api_key"] = self.api_key

        for attempt in range(1, self.max_attempts + 1):
            self.limiter.wait()
            self.requests_sent += 1
            try:
                response = self.http.get(f"/{endpoint}.json", params=params)
            except httpx.TransportError:
                if attempt == self.max_attempts:
                    raise
                self.sleep(self.backoff_base * 2 ** (attempt - 1))
                continue

            self.last_raw_bytes = len(response.content)
            if response.status_code == 200:
                return response.json()
            if response.status_code == 404 and _is_no_matches(response):
                return empty_response(limit, skip)
            if response.status_code == 429 and _is_daily_quota(response):
                url = response.request.url.copy_remove_param("api_key")
                raise OpenFDAError(response.status_code, response.text, str(url))
            if response.status_code in RETRY_STATUSES and attempt < self.max_attempts:
                retry_after = response.headers.get("retry-after", "")
                delay = float(retry_after) if retry_after.isdigit() else self.backoff_base * 2 ** (attempt - 1)
                self.sleep(delay)
                continue
            url = response.request.url.copy_remove_param("api_key")
            raise OpenFDAError(response.status_code, response.text, str(url))
        raise AssertionError("unreachable")
