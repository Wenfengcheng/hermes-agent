"""Canonical container removal needs the same consent as lifecycle operations (#132483)."""
import pytest

from tools.approval_detection import detect_dangerous_command, detect_hardline_command


@pytest.mark.parametrize("engine", ["docker", "podman"])
@pytest.mark.parametrize("flags", [
    "", "--log-level debug ", "--log-level  debug ", "--log-level\t\tdebug ",
    "--debug ", "--log-level=debug ", "--opt val --flag=x -q - " * 58,
])
@pytest.mark.parametrize("operation", ["rm app", "rm -f app", "prune", "prune -f"])
def test_container_removal_requires_consent(engine, flags, operation, monkeypatch):
    command = f"{engine} {flags}container {operation}"
    dangerous, key, description = detect_dangerous_command(command)
    assert dangerous is True, command
    assert key and description
    assert "container" in description and "destr" in description
    # These are intentional operator actions, not an unconditional hardline prohibition.
    assert detect_hardline_command(command) == (False, None)
    from hermes_cli.approvals_suggest import build_proposals

    assert build_proposals([(command, description)] * 3) == []
    # Remote payloads must remain detectable, without executing a container command.
    assert detect_dangerous_command(f"ssh host '{command}'")[0] is True

    from tools import approval, approval_context

    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(approval_context, "_get_approval_config", lambda: {"mode": "manual"})
    monkeypatch.setattr(
        "tools.tirith_security.check_command_security",
        lambda _: {"action": "allow", "findings": [], "summary": ""},
    )
    prompts = []

    def deny(*args, **kwargs):
        prompts.append((args, kwargs))
        return "deny"

    result = approval.check_all_command_guards(command, "local", approval_callback=deny)
    assert len(prompts) == 1
    assert result["approved"] is False


@pytest.mark.parametrize("engine", ["docker", "podman"])
@pytest.mark.parametrize("operation", ["container ls", "container inspect app", "container logs app", "container rename old new", "image rm img", "network rm net", "rmi img", "ps", "run --rm alpine true"])
def test_nondestructive_container_queries_and_excluded_resources_stay_clean(engine, operation):
    assert detect_dangerous_command(f"{engine} {operation}") == (False, None, None)
