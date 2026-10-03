"""Deferred source repair must reach the previously committed dependencies."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("mode", ["cap", "backoff", "sync-error"])
def test_bootstrap_c_program_keeps_previous_generation(tmp_path, mode):
    """Run the real bootstrap, launch decision, and activation in a fresh process.

    The dependency currency and store interpreter are controlled boundaries;
    the retry records, committed generation, and its import are real. Deny
    process creation so the unfixed self-relaunch fails safely instead of
    growing a process tree (Windows) or replacing the probe (POSIX).
    """
    source = Path(__file__).resolve().parents[2]
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / ".git").mkdir()
    (checkout / "pyproject.toml").write_text("[project]\nname='probe'\n", encoding="utf-8")
    (checkout / "install-stamp.json").write_text(
        json.dumps({"updateMechanism": "self"}), encoding="utf-8")
    shutil.copy2(source / "hermes_bootstrap.py", checkout / "hermes_bootstrap.py")
    home = tmp_path / "home"
    home.mkdir()
    env = {key: value for key, value in os.environ.items() if key.upper() in {
        "PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP",
    }}
    for key in ("HOME", "USERPROFILE", "HERMES_HOME", "APPDATA", "LOCALAPPDATA"):
        env[key] = str(home)
    env["PYTHONIOENCODING"] = "utf-8"
    program = textwrap.dedent(f"""
        import json, os, sys
        from pathlib import Path
        sys.path.insert(0, {str(source)!r})
        import pm
        from pm.environments import install_state_dir, runtime_facts_path, site_packages
        from hermes_cli import venv_sync, _launchers
        root = Path({str(checkout)!r})
        mode = {mode!r}
        pending = venv_sync.arm_completion(root)
        count = (venv_sync.COMPLETION_RETRY_MAX_ATTEMPTS if mode == 'cap'
                 else venv_sync.COMPLETION_RETRY_BACKOFF_ATTEMPTS if mode == 'backoff'
                 else 0)
        attempts = venv_sync._completion_attempts_path(root)
        attempts.write_text(str(count), encoding='utf-8')
        generation = install_state_dir(root) / 'environments' / 'previous'
        generation.mkdir(parents=True)
        (generation / 'pyvenv.cfg').write_text(
            'version = ' + '.'.join(map(str, sys.version_info[:3])) + '\\n', encoding='utf-8')
        packages = site_packages(generation)
        packages.mkdir(parents=True)
        (packages / 'previous_generation_probe.py').write_text(
            'VALUE = "previous dependency generation"\\n', encoding='utf-8')
        facts = runtime_facts_path(root)
        facts.write_text(json.dumps({{'packages': {{'venv': {{'environment': str(generation)}}}}}}),
                         encoding='utf-8')
        before = facts.read_bytes()
        pm.venv_is_current = lambda **kw: False
        _launchers.resolve_store_python = lambda root: Path(sys.executable)
        syncs = []
        def sync(*args, **kwargs):
            syncs.append('sync')
            raise RuntimeError('fixture sync unavailable')
        pm.sync_venv = sync
        import pm.client
        pm.client.ensure_tools_for_sync = lambda: None
        publications = []
        venv_sync.publish_launchers = lambda root: publications.append(str(root))
        launches = []
        def audit(event, args):
            if event in ('subprocess.Popen', 'os.exec', 'os.posix_spawn'):
                launches.append(event)
                raise RuntimeError('fixture forbids process creation')
            if event == 'socket.connect':
                raise RuntimeError('fixture forbids network')
        sys.addaudithook(audit)
        sys.path.insert(0, str(root))
        import hermes_bootstrap
        import previous_generation_probe
        assert Path(previous_generation_probe.__file__).parent == packages
        assert facts.read_bytes() == before
        assert pending.is_file()
        if mode != 'sync-error':
            assert attempts.read_text(encoding='utf-8') == str(count)
        print(json.dumps({{'value': previous_generation_probe.VALUE, 'launches': launches,
                           'publications': publications, 'syncs': syncs, 'argv': sys.argv}}))
    """)
    result = subprocess.run(
        [sys.executable, "-I", "-c", program, "status"], cwd=checkout, env=env,
        capture_output=True, text=True, encoding="utf-8", timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    observed = json.loads(result.stdout)
    assert observed["launches"] == [], result.stderr
    assert observed["publications"] == []
    assert observed["value"] == "previous dependency generation"
    assert observed["argv"] == ["-c", "status"]
    assert observed["syncs"] == (["sync"] if mode == "sync-error" else [])
    assert result.stderr.count("could not be finished automatically") == int(mode == "cap")
    if mode == "sync-error":
        assert "fixture sync unavailable" in result.stderr
        assert "running with the previous dependencies" in result.stderr
    else:
        assert "source-update completion failed" not in result.stderr
