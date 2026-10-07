"""The audit reports discovered distribution roots, not guessed venvs."""

import argparse
import importlib.metadata as metadata
import json

import pytest

from hermes_cli import security_audit as sa


def installed_fixture(root):
    info = root / "fixture_package-1.0.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: fixture-package\nVersion: 1.0\n", encoding="utf-8")
    return metadata.PathDistribution(info)


@pytest.mark.parametrize("output_json", [False, True])
def test_command_reports_all_discovered_roots_without_duplicate_findings(tmp_path, monkeypatch, capsys, output_json):
    roots = [tmp_path / "environment one" / "site-packages", tmp_path / "lazy target"]
    for root in roots:
        installed_fixture(root)
    # Exercise actual metadata discovery over only the disposable installation roots.
    distributions = metadata.distributions
    monkeypatch.setattr(metadata, "distributions", lambda: distributions(path=[str(p) for p in [*roots, roots[0]]]))

    def http_fixture(url, payload=None):
        if payload is not None:
            assert payload == {"queries": [{"package": {"name": "fixture-package", "ecosystem": "PyPI"}, "version": "1.0"}]}
            return {"results": [{"vulns": [{"id": "PYSEC-fixture"}]}]}
        return {"summary": "Fixture advisory", "database_specific": {"severity": "HIGH"}}

    monkeypatch.setattr(sa, "_http_json", http_fixture)
    args = argparse.Namespace(skip_venv=False, skip_plugins=True, skip_mcp=True, json=output_json, fail_on="high")
    assert sa.cmd_security_audit(args) == 1
    output = capsys.readouterr().out
    expected = [str(p.absolute()) for p in roots]
    if output_json:
        result = json.loads(output)
        assert result["total_components_scanned"] == result["finding_count"] == 1
        assert result["findings"][0].get("locations") == expected
        assert result["findings"][0]["source"] == "venv"
    else:
        for root in expected:
            assert root in output
        assert output.count("fixture-package==1.0") == 1


@pytest.mark.parametrize("failure", [OSError, NotImplementedError])
def test_unavailable_location_preserves_component_and_threshold(tmp_path, monkeypatch, capsys, failure):
    dist = installed_fixture(tmp_path / "site-packages")

    def unavailable(path):
        raise failure("metadata finder cannot locate files")

    monkeypatch.setattr(dist, "locate_file", unavailable)
    monkeypatch.setattr(metadata, "distributions", lambda: [dist])
    monkeypatch.setattr(sa, "_http_json", lambda url, payload=None: (
        {"results": [{"vulns": [{"id": "PYSEC-fixture"}]}]} if payload is not None
        else {"database_specific": {"severity": "HIGH"}}
    ))
    args = argparse.Namespace(skip_venv=False, skip_plugins=True, skip_mcp=True, json=True, fail_on="high")
    assert sa.cmd_security_audit(args) == 1
    row = json.loads(capsys.readouterr().out)["findings"][0]
    assert row["package"] == "fixture-package"
    assert row.get("locations") == []


def test_location_does_not_change_query_identity(tmp_path):
    from dataclasses import replace

    component = sa.Component("fixture", "1", "PyPI", "venv")
    located = replace(component, locations=(str(tmp_path),))
    assert located == component
    assert {component: "finding"}[located] == "finding"


def test_skip_venv_never_discovers_locations(monkeypatch, capsys):
    def forbidden():
        raise AssertionError("skipped discovery ran")

    monkeypatch.setattr(metadata, "distributions", forbidden)
    args = argparse.Namespace(skip_venv=True, skip_plugins=True, skip_mcp=True, json=True, fail_on="high")
    assert sa.cmd_security_audit(args) == 0
    assert json.loads(capsys.readouterr().out)["findings"] == []


def test_legacy_component_has_no_invented_location():
    finding = sa.Finding(sa.Component("fixture", "1", "PyPI", "plugin:fixture"), sa.Vulnerability("PYSEC-fixture"))
    row = json.loads(sa._render_json([finding], 1))["findings"][0]
    assert row.get("locations") == []
    assert "installed at:" not in sa._render_human([finding], 1)
