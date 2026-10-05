"""API requests must share the bounded non-chat stop grace (#132989)."""
import asyncio

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from tests.gateway.restart_test_helpers import make_restart_runner


@pytest.mark.asyncio
@pytest.mark.parametrize("chat_timeout, background_timeout", [(2.0, None), (0.0, 2.0)])
async def test_real_api_task_completes_with_zero_chat_budget(chat_timeout, background_timeout):
    runner, _ = make_restart_runner()
    api = APIServerAdapter(PlatformConfig(enabled=True))
    runner.adapters = {Platform.API_SERVER: api}
    release = asyncio.Event()
    task = asyncio.create_task(release.wait())
    api._active_run_tasks["queued"] = task
    assert api._active_run_agents == {}
    assert runner._active_api_run_count() == 1
    # The callback cannot finish the task until drain yields. No negative
    # wall-clock assertions or process-wide asyncio.create_task patch.
    asyncio.get_running_loop().call_soon(release.set)
    try:
        _, timed_out = await runner._drain_active_agents(chat_timeout, background_timeout)
    finally:
        release.set()
        await task
    assert timed_out is False
    assert runner._active_api_run_count() == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", [None, 0.0, 0.01])
async def test_unfinished_api_work_still_times_out(budget):
    runner, _ = make_restart_runner()
    api = APIServerAdapter(PlatformConfig(enabled=True))
    runner.adapters = {Platform.API_SERVER: api}
    task = asyncio.create_task(asyncio.Event().wait())
    api._active_run_tasks["queued"] = task
    try:
        _, timed_out = await runner._drain_active_agents(0.0, budget)
        assert timed_out is True
        assert not task.done()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_empty_api_adapter_does_not_wait(monkeypatch):
    runner, _ = make_restart_runner()
    runner.adapters = {Platform.API_SERVER: APIServerAdapter(PlatformConfig(enabled=True))}
    async def unexpected_sleep(*args):
        pytest.fail("empty adapter must not consume its drain budget")
    monkeypatch.setattr(asyncio, "sleep", unexpected_sleep)
    _, timed_out = await runner._drain_active_agents(0.0, 30.0)
    assert timed_out is False


@pytest.mark.asyncio
async def test_stop_phase_uses_default_background_budget_for_api():
    import time
    from gateway.run_shutdown import GatewayShutdownMixin

    runner, _ = make_restart_runner()
    api = APIServerAdapter(PlatformConfig(enabled=True))
    runner.adapters = {Platform.API_SERVER: api}
    release = asyncio.Event()
    task = asyncio.create_task(release.wait())
    api._active_run_tasks["queued"] = task
    ctx = GatewayShutdownMixin._StopContext(
        deferred_count=lambda: 0, started_at=time.monotonic()
    )
    asyncio.get_running_loop().call_soon(release.set)
    try:
        await runner._stop_drain_active_work(runner._restart_drain_timeout, ctx)
    finally:
        release.set()
        await task
    assert ctx.timed_out is False
    assert runner._active_api_run_count() == 0

