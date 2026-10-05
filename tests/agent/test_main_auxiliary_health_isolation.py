"""Main routing ignores side-task quarantine without clearing it (#133196)."""
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("lane", ["explicit", "fallback", "auto", "blank"])
def test_main_route_keeps_aux_marked_custom_endpoint(monkeypatch, lane):
    from agent import auxiliary_client as ac
    from agent.agent_init import _routed_client_kwargs

    from hermes_constants import get_hermes_home
    home = get_hermes_home()
    endpoint = "http://127.0.0.1:19876/v1"
    (home / "config.yaml").write_text(
        "model:\n  provider: custom\n  default: fixture-model\n"
        f"  base_url: {endpoint}\n", encoding="utf8"
    )
    ac._reset_aux_unhealthy_cache()
    ac.shutdown_cached_clients()
    try:
        if lane != "fallback":
            monkeypatch.setenv("OPENAI_BASE_URL", endpoint)
            monkeypatch.setenv("OPENAI_API_KEY", "fixture-key")
            agent = SimpleNamespace(provider={"explicit": "custom", "auto": "auto", "blank": ""}[lane], model="fixture-model")
            fallback = None
        else:
            agent = SimpleNamespace(provider="missing-scout-provider", model="missing-model")
            fallback = [{"provider": "custom", "model": "fixture-model", "base_url": endpoint, "api_key": "fixture-key"}]
        from hermes_cli.config import load_config_readonly
        assert load_config_readonly()["model"]["provider"] == "custom"
        assert ac._read_main_provider() == "custom"
        assert ac._read_main_model() == "fixture-model"
        control = _routed_client_kwargs(SimpleNamespace(provider=agent.provider, model=agent.model), fallback, None)
        assert str(control["base_url"]).rstrip("/") == endpoint
        ac._mark_provider_unhealthy("custom", base_url=endpoint)
        assert ac._is_provider_unhealthy("custom", endpoint)
        try:
            kwargs = _routed_client_kwargs(agent, fallback, None)
        except Exception as exc:
            pytest.fail(f"Main route {lane} lost available provider after aux-only mark: {exc}")
        assert str(kwargs["base_url"]).rstrip("/") == endpoint
        assert kwargs["api_key"] == "fixture-key"
        assert ac._is_provider_unhealthy("custom", endpoint)
        assert ac.resolve_provider_client("auto", model="fixture-model")[0] is None
        if lane == "fallback":
            assert agent._fallback_activated
            assert agent.provider == "custom"
    finally:
        ac.shutdown_cached_clients()
        ac._reset_aux_unhealthy_cache()


@pytest.mark.parametrize("outcome", ["empty", "raises", "nested"])
def test_main_resolution_scope_restores_and_does_not_change_other_threads(monkeypatch, outcome):
    from concurrent.futures import ThreadPoolExecutor
    from agent import auxiliary_client as ac
    from agent.auxiliary_health import resolve_main_provider_client

    endpoint = "http://127.0.0.1:19876/v1"
    ac._reset_aux_unhealthy_cache()
    ac._mark_provider_unhealthy("custom", base_url=endpoint)
    seen = []

    def resolve(provider, **kwargs):
        assert kwargs == {"model": "caller-model", "explicit_base_url": endpoint}
        assert not ac._is_provider_unhealthy("custom", endpoint)
        with ThreadPoolExecutor(max_workers=1) as executor:
            assert executor.submit(ac._is_provider_unhealthy, "custom", endpoint).result()
        seen.append(provider)
        if outcome == "raises":
            raise ValueError("fixture resolution error")
        if outcome == "nested" and provider == "outer":
            assert resolve_main_provider_client("inner", **kwargs) == (None, None)
            assert not ac._is_provider_unhealthy("custom", endpoint)
        return None, None

    monkeypatch.setattr(ac, "resolve_provider_client", resolve)
    try:
        if outcome == "raises":
            with pytest.raises(ValueError, match="fixture resolution error"):
                resolve_main_provider_client("outer", model="caller-model", explicit_base_url=endpoint)
        else:
            assert resolve_main_provider_client("outer", model="caller-model", explicit_base_url=endpoint) == (None, None)
        assert seen == (["outer", "inner"] if outcome == "nested" else ["outer"])
        assert ac._is_provider_unhealthy("custom", endpoint)
    finally:
        ac._reset_aux_unhealthy_cache()
