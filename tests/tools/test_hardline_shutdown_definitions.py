"""Definitions are not shutdown invocations; fixtures are never executed."""

import pytest

from tools.approval_detection import detect_hardline_command


@pytest.mark.parametrize("verb", ["shutdown", "reboot", "halt", "poweroff"])
@pytest.mark.parametrize("template", [
    "{verb}() {{ echo stop; }}",
    "set -e\n{verb} () {{ printf x; }}\necho ok",
    "{verb}( ) {{ printf x; }}",
    "{verb}\t() {{ echo ok; }}",
    "if true; then {verb}() {{ echo ok; }}; fi\nhalt() {{ echo ok; }}",
    "{verb}(\n) {{ echo ok; }}",
])
def test_shutdown_name_definition_is_not_invocation(verb, template):
    from tools.approval import (
        check_all_command_guards, disable_session_yolo, enable_session_yolo,
    )
    from tools.approval_context import reset_current_session_key, set_current_session_key

    command = template.format(verb=verb)
    assert detect_hardline_command(command) == (False, None)
    token = set_current_session_key("shutdown_definition")
    enable_session_yolo("shutdown_definition")
    try:
        assert check_all_command_guards(command, "local")["approved"]
        assert not check_all_command_guards(command + "\nreboot", "local")["approved"]
    finally:
        disable_session_yolo("shutdown_definition")
        reset_current_session_key(token)


@pytest.mark.parametrize("verb", ["shutdown", "reboot", "halt", "poweroff"])
@pytest.mark.parametrize("template", [
    "{verb}",
    "{verb} -f",
    "echo ok; {verb}",
    "echo ok && {verb}",
    "echo ok || {verb}",
    "echo ok | {verb}",
    "sudo {verb}",
    "{verb}() {{ echo ok; }}; {verb}",
    "helper() {{ {verb}; }}",
    "halt=$({verb})",
    "echo $({verb})",
    "sh -c '{verb}'",
    "{verb}\\(\\)",
    "{verb} '()'",
    "{verb}\n() {{ echo ok; }}",
    "halt() {{ echo ok; }}\n{verb}",
])
def test_shutdown_invocations_remain_blocked(verb, template):
    assert detect_hardline_command(template.format(verb=verb)) == (
        True, "system shutdown/reboot",
    )
