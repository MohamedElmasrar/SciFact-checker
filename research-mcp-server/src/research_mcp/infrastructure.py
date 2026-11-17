import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from time import monotonic
from typing import Generic, TypeVar

import httpx
import requests
from cachetools import TTLCache
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from .config import Settings

T = TypeVar("T")
K = TypeVar("K")
logger = logging.getLogger(__name__)
TRANSIENT_STATUSES = {408, 425, 429, 500, 502, 503, 504}


class CooldownActive(RuntimeError):
    """The upstream requested a pause longer than this call can wait."""


class UpstreamRateLimit(RuntimeError):
    """A provider SDK reports rate limiting without the HTTP response."""


def status_code(error: BaseException) -> int | None:
    if isinstance(error, UpstreamRateLimit):
        return 429
    response = getattr(error, "response", None)
    return response.status_code if response is not None else None


def is_transient(error: BaseException) -> bool:
    return status_code(error) in TRANSIENT_STATUSES or isinstance(
        error,
        (
            httpx.TransportError,
            requests.ConnectionError,
            requests.Timeout,
            TimeoutError,
        ),
    )


def retry_after(error: BaseException) -> float:
    response = getattr(error, "response", None)
    header = response.headers.get("Retry-After") if response is not None else None
    if not header:
        return 0
    try:
        return max(0, int(header))
    except ValueError:
        try:
            when = parsedate_to_datetime(header)
            return max(0, (when - datetime.now(UTC)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return 0


class RequestGate:
    """Limit each upstream independently and share cooldowns across callers."""

    def __init__(self, interval: float, settings: Settings):
        self.interval = interval
        self.settings = settings
        self._lock = asyncio.Lock()
        self._next = 0.0
        self._cooldown_until = 0.0

    async def run(
        self, operation: Callable[[], Awaitable[T]], transient: Callable = is_transient
    ) -> T:
        backoff = wait_random_exponential(
            multiplier=1, max=self.settings.retry_max_delay
        )

        def can_retry(error: BaseException) -> bool:
            return (
                transient(error) and retry_after(error) <= self.settings.retry_max_delay
            )

        def delay(state) -> float:
            return max(backoff(state), retry_after(state.outcome.exception()))

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.settings.retry_attempts),
            retry=retry_if_exception(can_retry),
            wait=delay,
            reraise=True,
        ):
            with attempt:
                async with self._lock:
                    now = monotonic()
                    if self._cooldown_until - now > self.settings.retry_max_delay:
                        raise CooldownActive()
                    await asyncio.sleep(
                        max(0, max(self._next, self._cooldown_until) - now)
                    )
                    self._next = monotonic() + self.interval
                    try:
                        return await operation()
                    except Exception as error:
                        cooldown = retry_after(error)
                        if cooldown:
                            self._cooldown_until = monotonic() + cooldown
                        raise
        raise RuntimeError("Retry loop exited without a result")


class AsyncCache(Generic[K, T]):
    """Bounded TTL cache that coalesces concurrent loads for the same key."""

    def __init__(
        self, size: int, ttl: float, getsizeof: Callable[[T], int] | None = None
    ):
        self._cache = TTLCache(maxsize=size, ttl=ttl, getsizeof=getsizeof)
        self._pending: dict[K, asyncio.Task[T]] = {}

    async def get(
        self,
        key: K,
        load: Callable[[], Awaitable[T]],
        *,
        cache_if: Callable[[T], bool] = bool,
    ) -> T:
        try:
            return self._cache[key]
        except KeyError:
            pass

        if key not in self._pending:

            async def fetch() -> T:
                try:
                    result = await load()
                    if (
                        cache_if(result)
                        and self._cache.getsizeof(result) <= self._cache.maxsize
                    ):
                        self._cache[key] = result
                    return result
                finally:
                    self._pending.pop(key, None)

            task = asyncio.create_task(fetch())
            task.add_done_callback(self._observe)
            self._pending[key] = task
        return await asyncio.shield(self._pending[key])

    @staticmethod
    def _observe(task: asyncio.Task) -> None:
        if not task.cancelled():
            task.exception()


class TimeoutSession(requests.Session):
    """Supply a timeout to SDKs whose requests calls omit it."""

    def __init__(self, timeout: float):
        super().__init__()
        self.timeout = timeout

    def request(self, method, url, **kwargs):
        kwargs.setdefault("timeout", self.timeout)
        response = super().request(method, url, **kwargs)
        if response.status_code in TRANSIENT_STATUSES:
            response.raise_for_status()
        return response
