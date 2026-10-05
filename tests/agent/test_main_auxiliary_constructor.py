"""Public constructor integration for auxiliary-only quarantine (#133196)."""
import pytest


@pytest.mark.parametrize("provider", ["auto", ""])
def test_constructor_preserves_main_route_after_auxiliary_quarantine(monkeypatch, provider):
    from hermes_constants import get_hermes_home
    from agent import auxiliary_client as ac
    from run_agent import AIAgent

    endpoint = "http://127.0.0.1:19876/v1"
    (get_hermes_home() / "config.yaml").write_text(
        "model:\n  provider: custom\n  default: fixture-model\n"
        f"  base_url: {endpoint}\n  context_length: 65536\n", encoding="utf8"
    )
    monkeypatch.setenv("OPENAI_BASE_URL", endpoint)
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-key")
    ac.shutdown_cached_clients()
    ac._reset_aux_unhealthy_cache()
    agents = []
    try:
        for marked in (False, True):
            if marked:
                ac._mark_provider_unhealthy("custom", base_url=endpoint)
            agent = AIAgent(provider=provider, model="fixture-model", quiet_mode=True,
                            skip_context_files=True, skip_memory=True,
                            skip_background_review=True, enabled_toolsets=[])
            agents.append(agent)
            assert str(agent.client.base_url).rstrip("/") == endpoint
            assert agent.client.api_key == "fixture-key"
        assert ac._is_provider_unhealthy("custom", endpoint)
        assert ac.resolve_provider_client("auto", model="fixture-model")[0] is None
    finally:
        for agent in agents:
            agent.client.close()
        ac.shutdown_cached_clients()
        ac._reset_aux_unhealthy_cache()
