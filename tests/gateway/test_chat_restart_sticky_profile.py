"""A detached chat restart must retain the host's settled root."""
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


@pytest.mark.platforms("windows")
@pytest.mark.asyncio
@pytest.mark.parametrize('host', [True, False])
@pytest.mark.parametrize('retry', [True, False])
async def test_chat_restart_does_not_adopt_sticky_profile(tmp_path, monkeypatch, host, retry):
    from agent import secret_scope
    from gateway.run_shutdown import GatewayShutdownMixin
    from hermes_cli import main

    root = tmp_path / 'root'
    worker = root / 'profiles' / 'worker'
    worker.mkdir(parents=True)
    (worker / 'config.yaml').write_text('{}\n', encoding='utf8')
    (root / 'config.yaml').write_text('gateway:\n  multiplex_profiles: true\n', encoding='utf8')
    (root / 'active_profile').write_text('worker', encoding='utf8')
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setattr('hermes_constants._get_platform_default_hermes_home', lambda: root)
    expected_home = root if host else worker
    monkeypatch.setenv('HERMES_HOME', str(expected_home))
    monkeypatch.setenv('HERMES_GATEWAY_LOCK_DIR', str(tmp_path / 'locks'))
    for key in ('HERMES_SUPERVISED_CHILD', 'HERMES_S6_SUPERVISED_CHILD', 'INVOCATION_ID', 'HERMES_GATEWAY_EXTERNAL_SUPERVISOR'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr('gateway.run._resolve_hermes_bin', lambda: ['hermes'])
    calls = []
    def capture(argv, **kw):
        calls.append((argv, kw))
        if retry and len(calls) == 1:
            raise OSError('synthetic job breakaway refusal')

    monkeypatch.setattr('subprocess.Popen', capture)
    secret_scope.set_multiplex_active(host)
    try:
        runner = SimpleNamespace(_detached_restart_helper_started=False, _restart_drain_timeout=0)
        await GatewayShutdownMixin._launch_detached_restart_command(runner)
        assert len(calls) == (2 if retry else 1)
        argv, kwargs = calls[-1]
        restart_argv = argv[argv.index('hermes'):]
        assert kwargs['env']['HERMES_HOME'] == str(expected_home)
        if retry:
            assert calls[0][0] == calls[1][0]
            assert calls[0][1]['env'] == calls[1][1]['env']
        with patch.dict(os.environ, kwargs['env'], clear=True):
            monkeypatch.setattr(sys, 'argv', restart_argv)
            main._apply_profile_override()
            assert Path(os.environ['HERMES_HOME']).resolve() == expected_home.resolve(), 'restart was redirected by sticky active_profile'
    finally:
        secret_scope.set_multiplex_active(False)


@pytest.mark.parametrize('home_kind', ['host', 'named', 'missing'])
def test_restart_argv_preserves_resolved_launcher_and_environment(tmp_path, monkeypatch, home_kind):
    from gateway.run_shutdown import GatewayShutdownMixin

    root = tmp_path / 'root'
    monkeypatch.setattr('hermes_constants.get_default_hermes_root', lambda: root)
    env = {} if home_kind == 'missing' else {'HERMES_HOME': str(root if home_kind == 'host' else root / 'profiles' / 'worker')}
    cmd = ['python with spaces', '-m', 'hermes_cli.main']
    original_env = dict(env)
    result = GatewayShutdownMixin._detached_restart_argv(cmd, env)
    assert result[:len(cmd)] == cmd
    assert result[len(cmd):] == (['--profile', 'default'] if home_kind == 'host' else []) + ['gateway', 'restart']
    assert env == original_env
    assert cmd == ['python with spaces', '-m', 'hermes_cli.main']
