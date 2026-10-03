"""PyPI releases are information, not an update channel Hermes can apply."""
import importlib.metadata
import json
from types import SimpleNamespace

import pytest


@pytest.fixture
def plugin_environment(tmp_path, monkeypatch):
    from hermes_cli import plugin_catalog, plugins_cmd, plugins_updates

    plugins = tmp_path / "plugins"
    target = plugins / "demo"
    target.mkdir(parents=True)
    (target / "plugin.yaml").write_text("name: demo\nversion: 1.0\n", encoding="utf-8")
    sidecar = plugins / ".install-metadata.json"
    sidecar.write_text(json.dumps({"demo": {
        "catalog": {"name": "demo"}, "revision": "a" * 40,
    }}), encoding="utf-8")
    dist = tmp_path / "demo_package-1.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Name: demo-package\nVersion: 1.0\n", encoding="utf-8")
    (dist / "entry_points.txt").write_text(
        "[hermes_agent.plugins]\ndemo = never_import_this:register\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    eps = importlib.metadata.distribution("demo-package").entry_points
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: eps)
    monkeypatch.setattr(plugins_cmd, "_plugins_dir", lambda: plugins)
    monkeypatch.setattr(plugin_catalog, "find_removed", lambda name: None)
    monkeypatch.setattr(plugin_catalog, "get_live_catalog_entry", lambda name: SimpleNamespace(sha="a" * 40))
    monkeypatch.setattr(plugins_updates, "_default_pypi_latest", lambda name: "1.1")
    monkeypatch.setenv("COLUMNS", "200")
    return plugins_updates, plugin_catalog, sidecar


@pytest.mark.parametrize("latest, catalog_sha", [
    ("1.1", "a" * 40), ("1.1", "b" * 40), ("1.1", None),
    ("1.0", "a" * 40), ("0.9", "a" * 40),
    (None, "a" * 40), ("not-a-version", "a" * 40),
])
def test_only_catalog_updates_are_presented_as_hermes_actionable(
    plugin_environment, capsys, monkeypatch, latest, catalog_sha,
):
    from hermes_cli.plugins_cmd import cmd_check_updates

    updates, catalog, sidecar = plugin_environment
    monkeypatch.setattr(updates, "_default_pypi_latest", lambda name: latest)
    monkeypatch.setattr(catalog, "get_live_catalog_entry", lambda name: SimpleNamespace(sha=catalog_sha))
    if catalog_sha is None:
        (sidecar.parent / "demo" / "plugin.yaml").unlink()
        (sidecar.parent / "demo").rmdir()
    before = sidecar.read_bytes()
    cmd_check_updates()
    text = capsys.readouterr().out
    assert ("update available" in text) == (catalog_sha == "b" * 40)
    assert ("newer on PyPI (informational)" in text) == (latest == "1.1")
    assert "not applied by Hermes" in text
    if catalog_sha is None:
        assert "hermes plugins update" not in text
    if latest is None or latest == "not-a-version":
        assert "unknown" in text
    assert sidecar.read_bytes() == before


def test_json_retains_version_comparison_but_explains_external_channel(plugin_environment, capsys):
    from hermes_cli.plugins_cmd import cmd_check_updates

    cmd_check_updates(SimpleNamespace(json=True))
    rows = json.loads(capsys.readouterr().out)
    catalog, pip = rows
    assert catalog["update_available"] is False
    assert pip["update_available"] is True
    assert (pip["current"], pip["latest"]) == ("1.0", "1.1")
    assert "not applied by Hermes" in pip["reason"]
