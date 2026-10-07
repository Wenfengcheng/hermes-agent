"""CLI inspection must disclose tools removed from a surviving toolset."""
from argparse import Namespace

import pytest

from hermes_cli import tools_config
from hermes_cli.tools_config_mcp import tools_disable_enable_command


def test_list_discloses_partially_disabled_web(monkeypatch, capsys):
    config = {
        "platform_toolsets": {"cli": ["web"]},
        "agent": {"disabled_toolsets": ["search"]},
    }
    monkeypatch.setattr(tools_config, "load_config", lambda: config)
    tools_disable_enable_command(Namespace(tools_action="list", platform="cli"))
    output = capsys.readouterr().out
    web_row = next(line for line in output.splitlines() if "  web  " in line)
    assert "partial" in web_row
    assert "web_search" in web_row
    assert "agent.disabled_toolsets" in web_row
    assert config["agent"]["disabled_toolsets"] == ["search"]


@pytest.mark.parametrize("disabled", [None, [], ["browser"], ["web"], ["not-a-toolset"]])
def test_no_partial_note_without_partial_suppression(disabled):
    assert tools_config._toolset_policy_note("web", disabled) == ""
    assert tools_config._toolset_policy_note("unregistered-server", disabled) == ""


@pytest.mark.parametrize("disabled", [["search"], '["search"]', "['search']"])
def test_summary_discloses_same_policy_and_preserves_extraction(disabled, capsys):
    from model_tools import _select_tool_names

    config = {"platform_toolsets": {"cli": ["web"]}, "agent": {"disabled_toolsets": disabled}}
    tools_config._print_tools_summary(config, ["cli"])
    output = capsys.readouterr().out
    assert "partial: web_search disabled by agent.disabled_toolsets" in output
    assert "Web Search & Scraping" in output
    enabled = tools_config._get_platform_tools(config, "cli", include_default_mcp_servers=False)
    runtime = _select_tool_names(sorted(enabled), ["search"], quiet_mode=True)
    assert "web_extract" in runtime
    assert "web_search" not in runtime


def test_original_list_caller_without_policy_keeps_enabled_row(capsys):
    from hermes_cli.tools_config_mcp import _print_tools_list

    _print_tools_list({"web"}, {})
    row = next(line for line in capsys.readouterr().out.splitlines() if "  web  " in line)
    assert "enabled" in row and "partial" not in row
