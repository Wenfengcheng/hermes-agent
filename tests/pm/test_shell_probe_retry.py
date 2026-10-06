"""Transient probe timeouts must not discard a healthy staged shell."""
import subprocess
from unittest.mock import Mock

import pytest

from pm import shell


@pytest.mark.parametrize("final", [0, 1, "timeout", "missing"])
def test_staged_shell_retries_timeout_once(monkeypatch, final):
    candidate = "staged-bash.exe"
    results = [subprocess.TimeoutExpired(candidate, 5)]
    if final == "timeout":
        results.append(subprocess.TimeoutExpired(candidate, 5))
    elif final == "missing":
        results.append(FileNotFoundError(candidate))
    else:
        results.append(subprocess.CompletedProcess(candidate, final))
    probe = Mock(side_effect=results)
    monkeypatch.setattr(shell, "_staged_bash", lambda: candidate)
    monkeypatch.setattr(shell.subprocess, "run", probe)
    monkeypatch.setattr(shell.shutil, "which", lambda _: None)
    monkeypatch.setattr(shell.os.path, "isfile", lambda _: False)
    monkeypatch.setattr(shell.os, "access", lambda *_: False)

    assert shell.bash() == (candidate if final == 0 else None)
    assert probe.call_count == 2
    assert probe.call_args_list[0] == probe.call_args_list[1]


@pytest.mark.parametrize("result", [0, 1, "missing"])
def test_completed_or_unlaunchable_probe_is_not_retried(monkeypatch, result):
    probe = Mock()
    if result == "missing":
        probe.side_effect = FileNotFoundError("bash")
    else:
        probe.return_value = subprocess.CompletedProcess("bash", result)
    monkeypatch.setattr(shell.subprocess, "run", probe)
    assert shell._bash_starts("bash") is (result == 0)
    assert probe.call_count == 1


@pytest.mark.platforms("windows")
def test_exhausted_staged_probe_preserves_candidate_fallback(monkeypatch):
    probe = Mock(side_effect=[
        subprocess.TimeoutExpired("staged", 5),
        subprocess.TimeoutExpired("staged", 5),
        subprocess.CompletedProcess("override", 0),
    ])
    monkeypatch.setattr(shell, "_staged_bash", lambda: "staged")
    monkeypatch.setenv("HERMES_GIT_BASH_PATH", "override")
    monkeypatch.setattr(shell.subprocess, "run", probe)
    monkeypatch.setattr(shell.shutil, "which", lambda _: None)
    monkeypatch.setattr(shell.os.path, "isfile", lambda p: p == "override")
    assert shell.bash() == "override"
    assert [call.args[0][0] for call in probe.call_args_list] == [
        "staged", "staged", "override",
    ]


@pytest.mark.platforms("windows")
def test_transient_probe_recovers_to_real_git_bash(monkeypatch):
    # Resolve only installed candidates; never provision a shell or read PM state.
    candidate = next(
        (p for p in shell.windows_bash_candidates(None, shell.os.environ)
         if shell.os.path.isfile(p)),
        None,
    )
    if candidate is None:
        pytest.skip("requires installed Git Bash")
    real_run = subprocess.run
    calls = []

    def transient_start(argv, **kwargs):
        calls.append(argv)
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        return real_run(argv, **kwargs)

    monkeypatch.setattr(shell, "_staged_bash", lambda: candidate)
    monkeypatch.setattr(shell.subprocess, "run", transient_start)
    resolved = shell.bash()
    assert resolved == candidate
    assert len(calls) == 2
    assert real_run([resolved, "-c", "exit 0"], timeout=15).returncode == 0
