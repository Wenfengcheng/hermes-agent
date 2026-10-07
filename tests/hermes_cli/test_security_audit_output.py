"""Regression for #134652: actionable advisory text survives CLI rendering."""

import argparse
import json
import urllib.error

import pytest

from hermes_cli import security_audit as sa


@pytest.mark.parametrize("output_json", [False, True])
@pytest.mark.parametrize("details_available", [False, True])
def test_audit_keeps_advisory_details_and_link(tmp_path, monkeypatch, capsys, output_json, details_available):
    plugin = tmp_path / "plugins" / "audit-fixture"
    plugin.mkdir(parents=True)
    (plugin / "requirements.txt").write_text("fixture-package==1.0\n", encoding="utf-8")
    monkeypatch.setattr(sa, "get_hermes_home", lambda: tmp_path)
    summary = "A condition requiring investigation. " * 8 + "Important final mitigation."
    advisory_id = "PYSEC-2099-123"
    fixes = ["1.1", "2.1", "3.1", "4.1"]

    def http_fixture(url, payload=None):
        if payload is not None:
            assert payload == {"queries": [{"package": {"name": "fixture-package", "ecosystem": "PyPI"}, "version": "1.0"}]}
            return {"results": [{"vulns": [{"id": advisory_id}]}]}
        if not details_available:
            raise urllib.error.URLError("fixture unavailable")
        return {"summary": summary, "database_specific": {"severity": "HIGH"},
                "affected": [{"ranges": [{"events": [{"fixed": v} for v in fixes]}]}]}

    monkeypatch.setattr(sa, "_http_json", http_fixture)
    args = argparse.Namespace(skip_venv=True, skip_plugins=False, skip_mcp=True,
                              json=output_json, fail_on="high")
    assert sa.cmd_security_audit(args) == int(details_available)
    output = capsys.readouterr().out
    link = "https://osv.dev/vulnerability/" + advisory_id
    if output_json:
        row = json.loads(output)["findings"][0]
        assert row["url"] == link
        assert row["summary"] == (summary if details_available else "")
        assert row["fixed_versions"] == (fixes if details_available else [])
    else:
        assert link in output
        if details_available:
            assert " ".join(summary.split()) in " ".join(output.split())
            assert ", ".join(fixes) in output


def test_advisory_link_encodes_id_without_changing_id(monkeypatch):
    finding = sa.Finding(sa.Component("fixture", "1", "PyPI", "venv"),
                         sa.Vulnerability("custom/id?x#y", summary=""))
    row = json.loads(sa._render_json([finding], 1))["findings"][0]
    assert row["vuln_id"] == "custom/id?x#y"
    assert row["url"] == "https://osv.dev/vulnerability/custom%2Fid%3Fx%23y"
    assert row["url"] in sa._render_human([finding], 1)
