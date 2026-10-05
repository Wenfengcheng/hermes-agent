"""First writes must not silently split model IDs (#133459)."""

import copy

import hermes_yaml as yaml
import pytest

from hermes_cli.config import _set_nested, set_config_value
from hermes_constants import resolve_reasoning_config


@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize("key", [
    "agent.reasoning_overrides.glm-5.3-flash",
    r"agent.reasoning_overrides.vendor/model\.2.1",
])
@pytest.mark.parametrize("overrides", [None, {}, {"other-model": "low"}, {"glm-5": "low"}])
def test_ambiguous_first_write_refuses_without_mutation(tmp_path, monkeypatch, overrides, key, force):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = {"agent": {"reasoning_effort": "medium"}}
    if overrides is not None:
        config["agent"]["reasoning_overrides"] = overrides
    before = copy.deepcopy(config)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    original = path.read_bytes()
    with pytest.raises(SystemExit) as exc:
        set_config_value(key, "high", force=force)
    assert exc.value.code == 1
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="escape"):
        _set_nested(config, key, "high")
    assert config == before


@pytest.mark.parametrize("model", ["glm-5.3-flash", "vendor/model.2.1", "plain-model"])
def test_explicit_creation_updates_and_fields_reach_runtime(tmp_path, monkeypatch, model):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "config.yaml"
    path.write_text("agent:\n  reasoning_effort: medium\n", encoding="utf-8")
    prefix = "agent.reasoning_overrides."
    set_config_value(prefix + model.replace(".", "\\."), "high")
    saved = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert saved["agent"]["reasoning_overrides"] == {model: "high"}
    assert resolve_reasoning_config(saved, model) == {"enabled": True, "effort": "high"}
    set_config_value(prefix + model, '{"enabled": true, "effort": "fast"}')
    set_config_value(prefix + model + ".enabled", "false")
    saved = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert saved["agent"]["reasoning_overrides"][model] == {"enabled": False, "effort": "fast"}
    assert resolve_reasoning_config(saved, model) == {"enabled": False}
    # The dot-free field spelling creates a dictionary for a new model too.
    set_config_value(prefix + "new-model.effort", "fast")
    saved = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert resolve_reasoning_config(saved, "new-model") == {"enabled": True, "effort": "fast"}
    # Longest literal wins even when a shorter model name is already present.
    set_config_value(prefix + "prefix-model", "low")
    set_config_value(prefix + r"prefix-model\.3-flash", "high")
    set_config_value(prefix + "prefix-model.3-flash", "max")
    saved = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert resolve_reasoning_config(saved, "prefix-model.3-flash") == {"enabled": True, "effort": "max"}
    assert resolve_reasoning_config(saved, "prefix-model") == {"enabled": True, "effort": "low"}
    # An error's suggested field path must configure the original model.
    config = {}
    with pytest.raises(ValueError) as exc:
        _set_nested(config, prefix + "new.3-model.effort", "fast")
    suggestion = str(exc.value).split("(e.g. ", 1)[1].split(").", 1)[0]
    _set_nested(config, suggestion, "fast")
    assert resolve_reasoning_config(config, "new.3-model") == {"enabled": True, "effort": "fast"}
