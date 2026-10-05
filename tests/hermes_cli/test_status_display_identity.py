"""The statusbar's REST snapshot retains the same label as health discovery."""
import json
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize('base,derived,distance', [
    ('1.2.3', '1.2.3+4.gabcdef0', 4),
    ('1.2.3', '1.2.3', 0),
    ('unknown', 'git.abcdef0', None),
    ('unknown', 'unknown', None),
])
async def test_status_preserves_display_identity(tmp_path, monkeypatch, base, derived, distance):
    from hermes_cli import version_info
    from hermes_cli.web_routers import status

    stamp = tmp_path / 'install-stamp.json'
    stamp.write_text(json.dumps({'schemaVersion': 2, 'baseVersion': base,
        'displayVersion': derived, 'distance': distance, 'commit': 'a' * 40,
        'source': 'git', 'updateMechanism': 'self'}), encoding='utf-8')
    monkeypatch.setattr(version_info, '_resolve_stamp_file', lambda: stamp)
    monkeypatch.setattr(version_info, '_cached_version_info', None)
    monkeypatch.setattr(status, 'check_config_version', lambda: (1, 1))
    gateway = dict(runtime=None, gateway_running=False, gateway_state='stopped',
        gateway_platforms={}, gateway_exit_reason=None, gateway_updated_at=None,
        gateway_heartbeat_stale_s=None, gateway_shared_with=None, gateway_pid=None)
    monkeypatch.setattr(status, '_resolve_gateway_status', AsyncMock(return_value=gateway))
    monkeypatch.setattr(status, '_collect_profile_gateway_topology_cached', lambda: {
        'profiles': [], 'gateway_mode': 'standalone', 'gateways': []})
    monkeypatch.setattr(status, '_status_active_sessions', AsyncMock(return_value=0))
    monkeypatch.setattr(status, '_resolve_restart_drain_timeout', lambda: 30)
    monkeypatch.setattr(status, '_dashboard_local_update_managed_externally', lambda: False)
    monkeypatch.setattr(status, '_auth_gate_status', lambda: {'auth_required': True})
    monkeypatch.setattr(status, '_nous_session_validity', lambda: 'unknown')
    monkeypatch.setattr(status, 'get_install_id', lambda: None)
    monkeypatch.setattr('hermes_cli.shared_profile_warning.shared_profile_warning', lambda: None)
    monkeypatch.setattr(status, '_component_health', AsyncMock(return_value={}))
    monkeypatch.setattr(status, '_advisory_pressure', AsyncMock())

    health = await status.get_health()
    snapshot = await status.get_status()
    assert snapshot['version'] == health['version'] == base
    assert snapshot.get('displayVersion') == health['displayVersion']
