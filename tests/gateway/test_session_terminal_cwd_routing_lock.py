"""A blocked end-of-turn cwd write must not freeze session.create (#158204)."""
import asyncio
import json
import threading

import pytest

from gateway import session_local_recovery
from gateway.session_terminal_cwd import persist_turn_cwd
from tests.gateway.test_session_create_tier_cwd import _create, _owner
from tools.terminal_tool import clear_session_cwd, record_session_cwd


@pytest.mark.usefixtures('authority_cwd_in_tmp')
@pytest.mark.asyncio
async def test_session_create_keeps_loop_responsive_during_cwd_write(tmp_path, monkeypatch):
    store, runner, authority, owner = _owner(tmp_path / 'home', monkeypatch, {'model': {'default': 'm'}})
    worker = None
    release = threading.Event()
    try:
        live, _ = await _create(owner, authority, 'first', tmp_path)
        route = live.route
        changed = tmp_path / 'changed'
        changed.mkdir()
        record_session_cwd(route, str(changed))
        token = store._entries[route].created_at
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()
        timed_out = threading.Event()
        persist_mirror = store._save_sessions_json
        routing = store._routing_persistence.routing
        persist_entry = type(routing).upsert_routing_entry

        def wait_for_loop():
            loop.call_soon_threadsafe(entered.set)
            if not release.wait(5):
                timed_out.set()

        def slow_mirror(data):
            wait_for_loop()
            persist_mirror(data)

        def slow_upsert(self, key, entry_json, **kwargs):
            if key == route:
                wait_for_loop()
            return persist_entry(self, key, entry_json, **kwargs)

        monkeypatch.setattr(store, '_save_sessions_json', slow_mirror)
        monkeypatch.setattr(type(routing), 'upsert_routing_entry', slow_upsert)
        worker = asyncio.create_task(asyncio.to_thread(persist_turn_cwd, runner, route, token))
        await asyncio.wait_for(entered.wait(), 5)
        restore = session_local_recovery.restore_local_session

        def restore_with_probe(authority, sid):
            # This callback is queued just before the route install takes the lock on the loop.
            loop.call_soon(release.set)
            return restore(authority, sid)

        monkeypatch.setattr(session_local_recovery, 'restore_local_session', restore_with_probe)
        second, _ = await _create(owner, authority, 'second', tmp_path)
        assert second.route != route
        await worker
        assert not timed_out.is_set(), 'session.create blocked the owner loop until cwd persistence returned'
        saved = routing.load_routing_entries(routing_scope=store._routing_scope())
        assert json.loads(saved[route])['metadata']['terminal_cwd'] == str(changed)
        assert second.route in routing.load_routing_entries()  # authority creation uses the default scope
    finally:
        release.set()
        if worker is not None:
            await worker
            clear_session_cwd(route)
        await owner.close()
        store.close_all_db_handles()
