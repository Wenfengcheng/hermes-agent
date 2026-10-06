"""Opt-in isolation for built-in error verdict tests (Part of #133719)."""

import pytest


@pytest.fixture(autouse=True)
def _builtin_only_plugins(_hermetic_environment, monkeypatch):
    """Fresh real registry per case, without importing every bundled plugin.

    Only modules importing this fixture opt in. Plugin discovery and verdict
    integration tests retain the production lazy-discovery path.
    """
    from hermes_cli import plugins

    manager = plugins.PluginManager()
    manager._discovered = True
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    return manager
