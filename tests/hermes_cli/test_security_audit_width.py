"""Regression for #134652: wrap human prose without clipping actionable data."""

import argparse
import json

import pytest

from hermes_cli import security_audit as sa


@pytest.mark.parametrize("columns", [40, 80, 140])
@pytest.mark.parametrize("output_json", [False, True])
def test_command_respects_width_without_losing_description(tmp_path, monkeypatch, capsys, columns, output_json):
    plugin = tmp_path / "plugins" / "fixture"
    plugin.mkdir(parents=True)
    (plugin / "requirements.txt").write_text("fixture==1.0\n", encoding="utf-8")
    monkeypatch.setattr(sa, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setenv("COLUMNS", str(columns))
    summary = "Investigate this affected installation and apply the recommended mitigation. " * 5 + "Final detail."

    def http_fixture(url, payload=None):
        if payload is not None:
            return {"results": [{"vulns": [{"id": "PYSEC-2099-1"}]}]}
        return {"summary": summary, "database_specific": {"severity": "HIGH"}}

    monkeypatch.setattr(sa, "_http_json", http_fixture)
    args = argparse.Namespace(skip_venv=True, skip_plugins=False, skip_mcp=True,
                              json=output_json, fail_on="high")
    assert sa.cmd_security_audit(args) == 1
    output = capsys.readouterr().out
    if output_json:
        assert json.loads(output)["findings"][0]["summary"] == summary
    else:
        prose = [line for line in output.splitlines() if line.startswith("           ")
                 and not line.strip().startswith("https://")]
        assert " ".join(summary.split()) == " ".join(" ".join(prose).split())
        assert all(len(line) <= columns for line in prose)
        assert "https://osv.dev/vulnerability/PYSEC-2099-1" in output


@pytest.mark.parametrize("columns", [1, 40, 120])
def test_human_wrap_preserves_long_tokens_and_explicit_paragraphs(monkeypatch, columns):
    monkeypatch.setenv("COLUMNS", str(columns))
    token = "identifier_" + "x" * 160
    summary = "First paragraph.\n\n" + token + " final mitigation."
    finding = sa.Finding(sa.Component("fixture", "1", "PyPI", "venv"),
                         sa.Vulnerability("PYSEC-2099-1", summary=summary, fixed_versions=["2", "3"]))
    output = sa._render_human([finding], 1)
    assert token in output
    assert " ".join(output.split("\n\n")[1].split()).endswith("First paragraph.")
    assert " ".join(summary.split()) in " ".join(output.split())
    assert "fixed in: 2, 3" in output
    assert "https://osv.dev/vulnerability/PYSEC-2099-1" in output
