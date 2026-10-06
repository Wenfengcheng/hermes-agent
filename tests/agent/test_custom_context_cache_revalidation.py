"""Custom routing aliases must recover from persisted windows (#133606)."""
from contextlib import contextmanager

import httpx
import pytest

from agent import model_metadata as mm


@pytest.fixture
def catalog(monkeypatch):
    state = {"length": 128_000, "calls": []}
    mm._endpoint_model_metadata_cache.clear()
    mm._endpoint_model_metadata_cache_time.clear()

    @contextmanager
    def stream(url, **kwargs):
        state["calls"].append(url)
        if state.get("error"):
            raise httpx.ConnectError("offline")
        rows = [] if state["length"] is None else [{"id": "auto", "context_length": state["length"]}]
        yield httpx.Response(200, json={"data": rows}, request=httpx.Request("GET", url))

    monkeypatch.setattr(mm.model_metadata_http, "stream", stream)
    return state


def resolve(**kwargs):
    return mm.get_model_context_length("auto", base_url="https://router.example/v1", provider="custom", custom_providers=[], **kwargs)


def test_remote_alias_revalidates_persisted_window(catalog):
    mm.save_context_length("auto", "https://router.example/v1", 24_000)
    assert resolve() == 128_000
    assert catalog["calls"]


def test_remote_alias_refreshes_after_metadata_ttl(catalog, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(mm.time, "time", lambda: clock[0])
    mm.save_context_length("auto", "https://router.example/v1", 24_000)
    assert resolve() == 128_000
    calls = len(catalog["calls"])
    catalog["length"] = 96_000
    assert resolve() == 128_000
    assert len(catalog["calls"]) == calls
    clock[0] += mm._ENDPOINT_MODEL_CACHE_TTL + 1
    assert resolve() == 96_000


@pytest.mark.parametrize("length", [None, 0, -1])
def test_missing_or_invalid_catalog_preserves_cached_fallback(catalog, length):
    catalog["length"] = length
    mm.save_context_length("auto", "https://router.example/v1", 96_000)
    assert resolve() == 96_000


def test_unreachable_catalog_preserves_cached_fallback(catalog):
    catalog["error"] = True
    mm.save_context_length("auto", "https://router.example/v1", 96_000)
    assert resolve() == 96_000


def test_explicit_override_does_not_probe(catalog):
    mm.save_context_length("auto", "https://router.example/v1", 24_000)
    assert resolve(config_context_length=64_000) == 64_000
    assert catalog["calls"] == []


def test_revalidation_does_not_rewrite_unscoped_scalar(catalog):
    mm.save_context_length("auto", "https://router.example/v1", 24_000)
    assert resolve() == 128_000
    assert mm.get_cached_context_length("auto", "https://router.example/v1") == 24_000


def test_raised_resolver_preserves_last_known_window(catalog, monkeypatch):
    def failed(*args, **kwargs):
        raise RuntimeError("catalog plugin failed")

    monkeypatch.setattr(mm, "_resolve_endpoint_context_length", failed)
    mm.save_context_length("auto", "https://router.example/v1", 96_000)
    assert resolve() == 96_000


def test_known_provider_keeps_existing_cache_precedence(catalog):
    mm.save_context_length("auto", "https://api.openai.com/v1", 96_000)
    assert mm.get_model_context_length("auto", base_url="https://api.openai.com/v1", provider="openai") == 96_000
    assert catalog["calls"] == []


def test_route_credential_reaches_catalog(catalog, monkeypatch):
    seen = []
    original = mm.model_metadata_http.stream

    @contextmanager
    def capture(url, **kwargs):
        seen.append(kwargs.get("headers", {}).get("Authorization"))
        with original(url, **kwargs) as response:
            yield response

    monkeypatch.setattr(mm.model_metadata_http, "stream", capture)
    mm.save_context_length("auto", "https://router.example/v1", 24_000)
    assert resolve(api_key="fixture-credential") == 128_000
    assert seen == ["Bearer fixture-credential"]


def test_known_provider_proxy_does_not_revalidate(catalog):
    mm.save_context_length("auto", "https://router.example/v1", 96_000)
    assert mm.get_model_context_length("auto", base_url="https://router.example/v1", provider="openai") == 96_000
    assert catalog["calls"] == []


def test_revalidation_does_not_promote_other_credentials_disk_catalog(catalog):
    url = "https://router.example/v1"
    mm._endpoint_disk_cache_put(url, {"auto": {"context_length": 1_000_000}})
    # Exercise the existing disk -> credential-keyed memory promotion first.
    assert mm.fetch_endpoint_model_metadata(url, api_key="fixture-B")["auto"]["context_length"] == 1_000_000
    mm.save_context_length("auto", url, 24_000)
    assert resolve(api_key="fixture-B") == 128_000
    assert mm.get_cached_context_length("auto", url) == 24_000


def test_revalidation_keeps_credentials_separate(catalog):
    mm.save_context_length("auto", "https://router.example/v1", 24_000)
    assert resolve(api_key="fixture-A") == 128_000
    catalog["length"] = 64_000
    assert resolve(api_key="fixture-B") == 64_000
    assert resolve(api_key="fixture-A") == 128_000


def test_expired_other_credential_does_not_change_offline_fallback(catalog, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(mm.time, "time", lambda: clock[0])
    mm.save_context_length("auto", "https://router.example/v1", 96_000)
    assert resolve(api_key="fixture-A") == 128_000
    catalog["length"] = 64_000
    assert resolve(api_key="fixture-B") == 64_000
    clock[0] += mm._ENDPOINT_MODEL_CACHE_TTL + 1
    catalog["error"] = True
    assert resolve(api_key="fixture-A") == 96_000
