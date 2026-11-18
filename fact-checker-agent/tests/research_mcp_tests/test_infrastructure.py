import asyncio
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from unittest.mock import AsyncMock

import httpx
import pytest

from research_mcp.infrastructure import (
    AsyncCache,
    CooldownActive,
    RequestGate,
    retry_after,
)


def http_error(status, retry=None):
    request = httpx.Request("GET", "https://example.org")
    headers = {"Retry-After": retry} if retry is not None else {}
    response = httpx.Response(status, request=request, headers=headers)
    with pytest.raises(httpx.HTTPStatusError) as error:
        response.raise_for_status()
    return error.value


def test_retry_after_seconds_and_http_date():
    date = format_datetime(datetime.now(UTC) + timedelta(seconds=15), usegmt=True)
    assert retry_after(http_error(429, "4")) == 4
    assert 13 <= retry_after(http_error(503, date)) <= 15
    assert retry_after(http_error(429, "invalid")) == 0


async def test_retry_uses_bounded_attempts_and_skips_permanent_errors(settings):
    settings = settings.model_copy(
        update={"retry_attempts": 3, "retry_max_delay": 0.01}
    )
    gate = RequestGate(0, settings)
    failing = AsyncMock(side_effect=http_error(503))
    with pytest.raises(httpx.HTTPStatusError):
        await gate.run(failing)
    assert failing.await_count == 3

    permanent = AsyncMock(side_effect=http_error(404))
    with pytest.raises(httpx.HTTPStatusError):
        await gate.run(permanent)
    assert permanent.await_count == 1

    long_cooldown = AsyncMock(side_effect=http_error(429, "10"))
    with pytest.raises(httpx.HTTPStatusError):
        await gate.run(long_cooldown)
    assert long_cooldown.await_count == 1
    with pytest.raises(CooldownActive):
        await gate.run(long_cooldown)
    assert long_cooldown.await_count == 1

    transient = AsyncMock(side_effect=[TimeoutError(), "ok"])
    assert await RequestGate(0, settings).run(transient) == "ok"
    assert transient.await_count == 2


async def test_cache_coalesces_survives_cancellation_and_does_not_cache_errors():
    cache = AsyncCache[str, str](2, 60)
    entered = asyncio.Event()
    finish = asyncio.Event()

    async def load():
        entered.set()
        await finish.wait()
        return "shared"

    loader = AsyncMock(side_effect=load)
    task = asyncio.create_task(cache.get("key", loader))
    await entered.wait()
    other = asyncio.create_task(cache.get("key", loader))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    finish.set()
    assert await other == "shared"
    assert await cache.get("key", loader) == "shared"
    assert loader.await_count == 1

    retry = AsyncMock(side_effect=[TimeoutError(), "ok"])
    with pytest.raises(TimeoutError):
        await cache.get("retry", retry)
    assert await cache.get("retry", retry) == "ok"


async def test_cache_expires_and_rejects_oversized_entries():
    clock = 0

    def timer():
        return clock

    # Inject a clock through cachetools to test TTL without real sleeps.
    from cachetools import TTLCache

    cache = AsyncCache[str, str](4, 5, len)
    cache._cache = TTLCache(maxsize=4, ttl=5, getsizeof=len, timer=timer)
    loader = AsyncMock(return_value="text")
    assert await cache.get("key", loader) == "text"
    clock += 6
    assert await cache.get("key", loader) == "text"
    assert loader.await_count == 2
    large = AsyncMock(return_value="too large")
    await cache.get("large", large)
    await cache.get("large", large)
    assert large.await_count == 2
