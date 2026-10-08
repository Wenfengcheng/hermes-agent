"""Synthetic wake routing must not overwrite observed Slack status anchors."""
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.relay.adapter import RelayAdapter
from gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor
from gateway.session import SessionSource
from gateway.wake import deliver_wake


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["prime", "deliver"])
@pytest.mark.parametrize("synthetic_id", [None, "old-wake-id"])
async def test_synthetic_wake_preserves_observed_status(entry, synthetic_id, monkeypatch):
    frames = []

    class Transport:
        async def send_outbound(self, action, *, platform=None):
            frames.append((action, platform))
            return {"success": True, "message_id": "reply"}

    adapter = RelayAdapter(PlatformConfig(), CapabilityDescriptor(
        contract_version=CONTRACT_VERSION, platform="slack", label="Slack",
        max_message_length=2000, supports_draft_streaming=False, supports_edit=True,
        supports_threads=True, markdown_dialect="markdown", len_unit="codepoints",
        emoji="", platform_hint="", pii_safe=False,
    ), transport=Transport())
    observed = MessageEvent(text="hello", message_type=MessageType.TEXT,
        message_id="1700.0001", source=SessionSource(platform=Platform.SLACK,
            chat_id="D1", chat_type="dm", user_id="u1"))
    # This is the same capture called by real relay inbound, without a network client.
    adapter._capture_scope(observed)
    source = SessionSource(platform=Platform.SLACK, chat_id="D1", chat_type="group",
                           user_id="u1", profile="worker")

    async def handle(event):
        event._gateway_accepted = True
    monkeypatch.setattr(adapter, "handle_message", handle)
    if entry == "deliver":
        await deliver_wake(adapter, text="done", source=source)
    else:
        adapter.prime_routing_cache(MessageEvent(text="done", source=source,
            message_type=MessageType.TEXT, internal=True, message_id=synthetic_id))
    await adapter.send_typing("D1")
    await adapter.stop_typing("D1")
    assert len(frames) == 2
    for action, platform in frames:
        assert platform == "slack"
        assert action["metadata"].get("thread_id") == "1700.0001"
        assert action["metadata"].get("user_id") == "u1"
        assert action["metadata"].get("profile") == "worker"
    assert frames[-1][0]["content"] == ""
    # Explicit caller-selected anchors still win and metadata is not mutated.
    explicit = {"thread_id": "chosen"}
    await adapter.send_typing("D1", explicit)
    assert frames[-1][0]["metadata"]["thread_id"] == "chosen"
    assert explicit == {"thread_id": "chosen"}
    # Later observed inbound must still advance the status anchor.
    observed.message_id = "1700.0002"
    adapter._capture_scope(observed)
    await adapter.send_typing("D1")
    assert frames[-1][0]["metadata"]["thread_id"] == "1700.0002"
    assert adapter._with_status_thread_anchor("other-chat", None) == {}


def test_cold_synthetic_prime_does_not_invent_observations():
    adapter = RelayAdapter.__new__(RelayAdapter)
    adapter._platform_by_chat = {}
    adapter._dm_user_by_chat = {}
    adapter._scope_by_chat = {}
    adapter._chat_type_by_chat = {}
    adapter._last_inbound_ts_by_chat = {}
    event = SimpleNamespace(source=SessionSource(platform=Platform.SLACK, chat_id="D1",
        chat_type="group", user_id="u1", scope_id="team", profile="worker"),
        message_id="synthetic-id")
    adapter.prime_routing_cache(event)
    assert adapter._with_scope("D1", None) == {"user_id": "u1", "scope_id": "team", "profile": "worker"}
    assert adapter._platform_by_chat == {"D1": "slack"}
    assert adapter._chat_type_by_chat == {}
    assert adapter._last_inbound_ts_by_chat == {}
    adapter.prime_routing_cache(None)
    adapter.prime_routing_cache(SimpleNamespace())
    assert adapter._with_status_thread_anchor("D1", None) == {}
