"""LONG RPC overload rejects unstarted work rather than queueing forever (#132546)."""

from concurrent.futures import ThreadPoolExecutor
import queue
import threading

import pytest


class Transport:
    def __init__(self):
        self.frames = queue.Queue()

    def write(self, frame):
        self.frames.put(frame)
        return True


@pytest.fixture
def runtime(monkeypatch):
    from tui_gateway import server
    from hermes_cli import backend_retirement

    fence = backend_retirement.RetirementFence()
    monkeypatch.setattr(backend_retirement, "retirement", fence)
    monkeypatch.setattr(server, "_LONG_HANDLERS", {"test.pool"})
    return server, fence


def test_saturation_rejects_without_running_and_recovers(runtime, monkeypatch):
    server, fence = runtime
    workers = server._rpc_pool_workers
    entered = queue.Queue()
    release = threading.Event()
    transport = Transport()

    def handler(rid, params):
        entered.put(rid)
        assert release.wait(10)
        return server._ok(rid, {"done": True})

    monkeypatch.setitem(server._methods, "test.pool", handler)
    monkeypatch.setitem(server._methods, "test.fast", lambda rid, params: server._ok(rid, {}))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        monkeypatch.setattr(server, "_pool", pool)
        try:
            for i in range(workers):
                assert server.dispatch({"id": i, "method": "test.pool"}, transport) is None
            assert {entered.get(timeout=10) for _ in range(workers)} == set(range(workers))
            response = server.dispatch({"id": "overflow", "method": "test.pool"}, transport)
            assert response is not None, "saturated LONG pool silently queued the request"
            assert response["id"] == "overflow"
            assert "busy" in response["error"]["message"].lower()
            assert fence.active_count() == workers
            assert server.dispatch({"id": "fast", "method": "test.fast"}, transport)["result"] == {}
        finally:
            release.set()
    assert fence.active_count() == 0
    assert entered.empty(), "rejected request ran later"
    assert {transport.frames.get_nowait()["id"] for _ in range(workers)} == set(range(workers))
    assert transport.frames.empty()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        monkeypatch.setattr(server, "_pool", pool)
        assert server.dispatch({"id": "retry", "method": "test.pool"}, transport) is None
    assert transport.frames.get_nowait()["id"] == "retry"
    assert fence.active_count() == 0


@pytest.mark.parametrize("workers", [1, 3])
def test_websocket_client_receives_busy_and_can_retry(runtime, monkeypatch, workers):
    from fastapi import FastAPI, WebSocket
    from fastapi.testclient import TestClient
    from tui_gateway.ws import handle_ws

    server, fence = runtime
    monkeypatch.setattr(server, "_rpc_pool_slots", threading.BoundedSemaphore(workers))
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0)
    entered = queue.Queue()
    release = threading.Event()

    def handler(rid, params):
        entered.put(rid)
        assert release.wait(10)
        return server._ok(rid, {})

    monkeypatch.setitem(server._methods, "test.pool", handler)
    app = FastAPI()

    @app.websocket("/ws")
    async def endpoint(ws: WebSocket):
        await handle_ws(ws)

    with ThreadPoolExecutor(max_workers=workers) as pool, ThreadPoolExecutor(max_workers=1) as reader:
        monkeypatch.setattr(server, "_pool", pool)
        with TestClient(app) as client, client.websocket_connect("/ws") as ws:
            def receive():
                return reader.submit(ws.receive_json).result(timeout=5)

            assert receive()["params"]["type"] == "gateway.ready"
            try:
                for i in range(workers):
                    ws.send_json({"jsonrpc": "2.0", "id": i, "method": "test.pool", "params": {}})
                assert {entered.get(timeout=5) for _ in range(workers)} == set(range(workers))
                ws.send_json({"jsonrpc": "2.0", "id": "overflow", "method": "test.pool", "params": {}})
                rejected = receive()
                assert rejected["id"] == "overflow"
                assert "busy" in rejected["error"]["message"].lower()
                ws.send_json({"jsonrpc": "2.0", "id": "ping", "method": "gateway.ping", "params": {}})
                assert receive() == {"jsonrpc": "2.0", "id": "ping", "result": {"ok": True}}
            finally:
                release.set()
            pool.shutdown(wait=True)
            assert {receive()["id"] for _ in range(workers)} == set(range(workers))
            assert entered.empty(), "rejected websocket request executed later"
            assert fence.active_count() == 0
            with ThreadPoolExecutor(max_workers=workers) as replacement:
                monkeypatch.setattr(server, "_pool", replacement)
                ws.send_json({"jsonrpc": "2.0", "id": "retry", "method": "test.pool", "params": {}})
                assert receive()["id"] == "retry"
    assert fence.active_count() == 0


@pytest.mark.parametrize("outcome", ["empty", "raised", "write_error", "cancelled", "submit_error"])
def test_every_terminal_path_releases_admission(runtime, monkeypatch, outcome):
    server, fence = runtime
    workers = server._rpc_pool_workers
    transport = Transport()
    release = threading.Event()

    def handler(rid, params):
        if outcome == "raised":
            raise ValueError("fixture failure")
        return None if outcome == "empty" else server._ok(rid, {})

    monkeypatch.setitem(server._methods, "test.pool", handler)
    if outcome == "write_error":
        def fail_write(frame):
            raise OSError("closed fixture transport")
        monkeypatch.setattr(transport, "write", fail_write)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        monkeypatch.setattr(server, "_pool", pool)
        if outcome == "cancelled":
            started = queue.Queue()
            def blocker():
                started.put(True)
                release.wait(10)
            for _ in range(workers):
                pool.submit(blocker)
            for _ in range(workers):
                started.get(timeout=10)
        if outcome == "submit_error":
            pool.shutdown()
        try:
            for i in range(workers):
                if outcome == "submit_error":
                    with pytest.raises(RuntimeError):
                        server.dispatch({"id": i, "method": "test.pool"}, transport)
                else:
                    assert server.dispatch({"id": i, "method": "test.pool"}, transport) is None
            if outcome == "cancelled":
                pool.shutdown(wait=False, cancel_futures=True)
        finally:
            release.set()
    assert fence.active_count() == 0
    # A full second wave must be admitted: no exception/cancellation leaked a slot.
    monkeypatch.setitem(server._methods, "test.pool", lambda rid, params: None)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        monkeypatch.setattr(server, "_pool", pool)
        for i in range(workers):
            assert server.dispatch({"id": i, "method": "test.pool"}, transport) is None
    assert fence.active_count() == 0
