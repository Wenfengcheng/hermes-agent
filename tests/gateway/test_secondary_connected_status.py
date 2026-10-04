"""Secondary install publishes health even when an adapter only returns True."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.status import flush_runtime_status


@pytest.fixture
def setup_runner(tmp_path, monkeypatch):
    home = tmp_path / 'launch'
    secondary = tmp_path / 'work'
    home.mkdir()
    secondary.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner._running = True
    runner._profile_adapters = {}
    runner._profile_failed_platforms = {}
    runner._sync_voice_mode_state_to_adapter = lambda adapter: None
    runner._schedule_planned_restart_replay = lambda: None
    runner._schedule_resume_pending_sessions = lambda **kw: None
    runner._redeliver_failed_obligations_for_platform = AsyncMock()
    runner._safe_adapter_disconnect = AsyncMock()
    runner._load_secondary_profile_config = AsyncMock(return_value=GatewayConfig(
        platforms={Platform.EMAIL: PlatformConfig(enabled=True)}))
    runner._adapter_credential_claim = lambda *args: None
    runner._adapter_listener_claim = lambda *args: None
    runner._configure_profile_adapter = lambda *args: None
    runner._schedule_secondary_profile_startup_reconnect = lambda *args: None
    monkeypatch.setattr('gateway.run._platform_has_bot_credential', lambda *args: True)
    runner._update_platform_runtime_status('email', platform_state='connected')
    runner._update_platform_runtime_status('other:email', platform_state='fatal', error_code='other_error')
    return runner, home, secondary


def read_rows(home):
    flush_runtime_status()
    return json.loads((home / 'gateway_state.json').read_text(encoding='utf8'))['platforms']


@pytest.mark.asyncio
@pytest.mark.parametrize('startup', [True, False])
@pytest.mark.parametrize('degraded', [False, True])
async def test_secondary_install_recovers_durable_status(setup_runner, startup, degraded):
    runner, home, secondary = setup_runner
    before = read_rows(home)
    if not startup:
        runner._update_platform_runtime_status('work:email', platform_state='fatal',
            error_code='email_imap_connect_error', error_message='offline',
            needs_attention=True, retrying_since='2026-01-01T00:00:00Z')
    adapter = SimpleNamespace(send_path_degraded=degraded, DEGRADED_STATUS_MESSAGE='receive unconfirmed')
    runner._create_adapter = lambda *args: adapter
    runner._connect_initial_adapter_with_timeout = AsyncMock(return_value=True)
    runner._secondary_reconnect_attempt = AsyncMock(return_value=(adapter, True))
    if startup:
        assert await runner._start_one_profile_adapters('work', secondary, {}) == 1
    else:
        await runner._run_secondary_profile_reconnect('work', Platform.EMAIL)
    assert runner._profile_adapters['work'][Platform.EMAIL] is adapter
    rows = read_rows(home)
    row = rows.get('work:email', {})
    assert row.get('state') == ('retrying' if degraded else 'connected'), row
    assert row.get('needs_attention') is False
    assert row.get('error_code') is None
    assert row.get('error_message') == ('receive unconfirmed' if degraded else None)
    assert row.get('retrying_since') is None
    assert rows['email'] == before['email']
    assert rows['other:email'] == before['other:email']
    assert not (secondary / 'gateway_state.json').exists()


@pytest.mark.asyncio
@pytest.mark.parametrize('outcome', ['false', 'exception', 'superseded', 'shutdown'])
async def test_rejected_reconnect_does_not_publish_health(setup_runner, outcome):
    runner, home, secondary = setup_runner
    runner._update_platform_runtime_status('work:email', platform_state='fatal', error_code='original')
    before = read_rows(home)
    adapter = SimpleNamespace(send_path_degraded=False, has_fatal_error=True, fatal_error_retryable=False)
    async def attempt(*args):
        if outcome == 'exception':
            runner._running = False
            raise OSError('local fixture failure')
        if outcome == 'shutdown':
            runner._running = False
        return adapter, outcome != 'false'
    runner._secondary_reconnect_attempt = attempt
    if outcome == 'superseded':
        runner._profile_adapters['work'] = {Platform.EMAIL: object()}
    await runner._run_secondary_profile_reconnect('work', Platform.EMAIL)
    assert read_rows(home) == before
