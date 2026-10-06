"""Built-in classifier fixtures avoid repeated discovery without sharing state."""

from unittest.mock import patch

import pytest

from agent.error_classifier import FailoverReason, classify_api_error
from hermes_cli.plugins import get_plugin_manager
from tests.agent.classifier_fixtures import _builtin_only_plugins  # noqa: F401


@pytest.mark.parametrize("case", ["first", "second"])
def test_builtin_registry_is_fresh_and_does_not_discover(case):
    manager = get_plugin_manager()
    assert manager._hooks == {}
    with patch.object(
        manager, "_discover_and_load_inner", wraps=manager._discover_and_load_inner
    ) as discover:
        manager.discover_and_load()
        result = classify_api_error(Exception("unrecognized synthetic failure"))
    assert result.reason is FailoverReason.unknown
    assert discover.call_count == 0
    # The next invocation must not inherit either callbacks or mutable registry state.
    manager._hooks[case] = [lambda: None]
