"""Sandbox desktops take their startup lock on the host, not inside Linux."""
import subprocess
import sys
from pathlib import Path

import pytest

from tools.bot_desktop import placement, runtime, sandbox_host
from tools.environments import streams

pytestmark = pytest.mark.platforms("linux", "macos", "windows")


@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text("bot_desktop:\n  geometry: 800x600\n", encoding="utf-8")
    monkeypatch.setattr(placement, "_terminal_backend", lambda: "docker")
    monkeypatch.setattr(runtime, "_profile_name", lambda: "fixture")

    class Transport:
        _container_id = "fixture-container"
        _docker_exe = "docker"
        running = False
        spawns = 0
        failure = None

        def get_temp_dir(self):
            return "/sandbox-tmp"

        def run(self, env, argv, **kwargs):
            assert env is self
            script = argv[-1]
            code, output = 0, b""
            if argv == ["id", "-u", "pn"]:
                output = b"1000"
            elif script == sandbox_host._CHROMIUM_PROBE:
                code = 1  # A screen need not have a dock browser.
            elif script.startswith("kill -0 "):
                code = 0 if self.running else 1
                output = b"DISPLAY=:20\nXAUTHORITY=/sandbox-tmp/fixture/Xauthority\n" if self.running else b""
            elif script.startswith("\nlive=0"):
                self.running = False
            elif script.startswith("for b in "):
                pass
            elif script.startswith("set -e; rm -rf "):
                if self.failure:
                    raise self.failure
            elif "setsid -f bash -c" in script:
                assert kwargs["child_env"]["HERMES_BD_GEOMETRY"] == "800x600"
                self.spawns += 1
                self.running = True
            else:
                pytest.fail(f"Unexpected sandbox command: {argv}")
            return subprocess.CompletedProcess(argv, code, output, b"")

    transport = Transport()
    monkeypatch.setattr(placement, "terminal_environment", lambda **kwargs: transport)
    monkeypatch.setattr(streams, "run_in", transport.run)
    monkeypatch.setattr(runtime, "_spawn_and_wait", lambda *a: pytest.fail("host launcher used"))
    return transport


def test_windows_tool_start_reaches_sandbox_and_publishes_state(sandbox):
    # Real tool boundary, start, lock, sandbox orchestration and status. Only
    # transport I/O is simulated; no Docker daemon or host desktop is touched.
    try:
        where = runtime.tool_placement()
    except ImportError as exc:
        pytest.fail(f"Sandbox startup must not require a POSIX host: {exc}")
    assert where == placement.TERMINAL
    assert sandbox.spawns == 1
    assert runtime.status().running
    assert runtime.status().placement == "terminal:docker"
    assert sandbox_host._read_marker()["container"] == sandbox._container_id
    assert "DISPLAY=:20" in (runtime.state_dir() / "env").read_text(encoding="utf-8")
    assert runtime.tool_placement() == placement.TERMINAL
    assert runtime.start().running
    assert sandbox.spawns == 1


def test_windows_sandbox_error_propagates_and_start_can_retry(sandbox):
    sandbox.failure = RuntimeError("fixture transport failed")
    with pytest.raises(RuntimeError, match="fixture transport failed"):
        runtime.tool_placement()
    assert sandbox.spawns == 0
    assert not sandbox_host._read_marker()
    sandbox.failure = None
    assert runtime.tool_placement() == placement.TERMINAL
    assert sandbox.spawns == 1


def test_browser_preflight_starts_sandbox_without_host_browser(sandbox, monkeypatch):
    from tools import browser_tool_session
    from tools import interrupt

    monkeypatch.setattr(interrupt, "is_interrupted", lambda: False)
    assert browser_tool_session._browser_command_preflight() == {"browser_cmd": "agent-browser"}
    assert sandbox.spawns == 1
    sandbox.running = False
    sandbox.failure = RuntimeError("fixture transport failed")
    result = browser_tool_session._browser_command_preflight()
    assert result["success"] is False
    assert "fixture transport failed" in result["error"]


def test_explicit_gateway_placement_does_not_start_sandbox(sandbox, monkeypatch):
    monkeypatch.setattr(placement, "_setting", lambda: "gateway")
    assert runtime.tool_placement() == placement.GATEWAY
    assert sandbox.spawns == 0
    assert not sandbox_host._read_marker()


def _peer_can_lock(path):
    result = subprocess.run(
        [sys.executable, "-c", "from pm.filesystem import lock_fd; import sys; "
         "f=open(sys.argv[1], 'a+'); print(lock_fd(f.fileno(), wait=False))", str(path)],
        cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True,
        timeout=30, check=True,
    )
    return result.stdout.strip() == "True"


@pytest.mark.parametrize("raise_inside", [False, True])
def test_start_lock_excludes_another_process_and_releases(tmp_path, raise_inside):
    path = tmp_path / "start.lock"
    try:
        with runtime._flocked(path):
            assert not _peer_can_lock(path)
            if raise_inside:
                raise RuntimeError("inside critical section")
    except RuntimeError as exc:
        assert raise_inside and str(exc) == "inside critical section"
    assert _peer_can_lock(path)


def test_lock_acquisition_failure_closes_handle(monkeypatch, tmp_path):
    import builtins
    from pm import filesystem

    opened = []
    real_open = builtins.open

    def capture(*args, **kwargs):
        handle = real_open(*args, **kwargs)
        opened.append(handle)
        return handle

    def fail(*args, **kwargs):
        raise OSError("fixture lock failure")

    monkeypatch.setattr(runtime, "open", capture, raising=False)
    monkeypatch.setattr(filesystem, "lock_fd", fail)
    with pytest.raises(OSError, match="fixture lock failure"):
        with runtime._flocked(tmp_path / "start.lock"):
            pytest.fail("unlocked critical section entered")
    assert opened and opened[0].closed
