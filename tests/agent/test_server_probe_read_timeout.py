"""A hanging fingerprint must not stall every leg or suppress /models (#126567)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from agent import model_metadata as mm


@pytest.fixture(autouse=True)
def isolated_probe_caches(monkeypatch):
    for name in (
        "_endpoint_probe_path_cache", "_endpoint_blackhole_cache",
        "_endpoint_model_metadata_cache", "_endpoint_model_metadata_cache_time",
    ):
        monkeypatch.setattr(mm, name, {})


def test_hanging_fingerprint_preserves_catalog_and_recovers_after_ttl():
    requests = []
    release = threading.Event()
    recovered = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append((self.path, self.headers.get("Authorization")))
            if self.path == "/v1/models":
                payload = {"data": [{"id": "fixture-model", "context_length": 8192}]}
            elif recovered.is_set() and self.path == "/api/v1/models":
                payload = {"models": []}
            else:
                release.wait(30)
                return
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root = f"http://127.0.0.1:{server.server_port}"
    try:
        assert mm.detect_local_server_type(root + "/v1", api_key="fixture-key") is None
        assert requests == [("/api/v1/models", "Bearer fixture-key")]
        assert not mm._endpoint_blackholed(root)
        assert mm.detect_local_server_type(root + "/v1") is None
        assert len(requests) == 1  # existing negative TTL avoids another stalled request

        catalog = mm.fetch_endpoint_model_metadata(root + "/v1", api_key="fixture-key")
        assert catalog["fixture-model"]["context_length"] == 8192
        assert requests[-1] == ("/v1/models", "Bearer fixture-key")
        assert len(requests) == 2

        recovered.set()
        verdict, stamp = mm._endpoint_probe_path_cache[root]
        mm._endpoint_probe_path_cache[root] = (
            verdict, stamp - mm._ENDPOINT_PROBE_FAILURE_TTL_SECONDS - 1,
        )
        assert mm.detect_local_server_type(root + "/v1") == "lm-studio"
        assert len(requests) == 3
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("failure", ["read-first", "read-later", "connect", "refused", "malformed", "empty", "success"])
def test_probe_stops_only_for_timeout_and_preserves_fallbacks(monkeypatch, failure):
    paths = []
    real_client = httpx.Client

    def respond(request):
        path = request.url.path
        paths.append(path)
        if failure == "read-first" or (failure == "read-later" and path == "/api/tags"):
            raise httpx.ReadTimeout("fixture read timeout", request=request)
        if failure == "connect":
            raise httpx.ConnectTimeout("fixture connect timeout", request=request)
        if failure == "refused":
            raise httpx.ConnectError("fixture refused", request=request)
        if failure == "malformed" and path == "/api/tags":
            return httpx.Response(200, text="not JSON")
        if path == "/props" and failure in ("malformed", "success"):
            return httpx.Response(200, json={"default_generation_settings": {}})
        return httpx.Response(404)

    monkeypatch.setattr(httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(respond), **kw))
    root = "http://127.0.0.1:18099"
    result = mm.detect_local_server_type(root + "/v1")
    if failure in ("read-first", "connect"):
        assert paths == ["/api/v1/models"]
    elif failure == "read-later":
        assert paths == ["/api/v1/models", "/api/tags"]
    elif failure in ("success", "malformed"):
        assert result == "llamacpp"
        assert paths[-1] == "/props"
    else:
        assert result is None
        assert paths[-1] == "/version"
    assert mm._endpoint_blackholed(root) == (failure == "connect")
