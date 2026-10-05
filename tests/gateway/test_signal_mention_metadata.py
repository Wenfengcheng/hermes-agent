"""Signal mention metadata survives text cleanup at ingress (#133247)."""
import json
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.signal import SignalAdapter


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "allowed")
    result = SignalAdapter(PlatformConfig(enabled=True, extra={
        "account": "+15550001234", "http_url": "http://localhost:8080",
        "require_mention": False,
    }))
    result.handle_message = AsyncMock()
    return result


def envelope(data):
    return {"envelope": {
        "sourceNumber": "+15550005678", "timestamp": 1000000000,
        "dataMessage": data,
    }}


@pytest.mark.asyncio
@pytest.mark.parametrize("edited", [False, True])
@pytest.mark.parametrize("require_mention", [False, True])
async def test_mentions_survive_real_sse_ingress(adapter, edited, require_mention):
    adapter.require_mention = require_mention
    mentions = [{"start": 0, "length": 1, "number": adapter.account},
                {"start": 6, "length": 1, "uuid": "other-person"}]
    quote = {"id": 99, "author": adapter.account, "text": "Earlier question"}
    data = {"message": "\ufffc ask \ufffc", "mentions": mentions,
            "groupInfo": {"groupId": "allowed"}, "quote": quote}
    payload = envelope(data)
    if edited:
        payload["envelope"]["editMessage"] = {
            "dataMessage": payload["envelope"].pop("dataMessage")}
    await adapter._handle_sse_line("data: " + json.dumps(payload))
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "ask @other-person"
    assert event.raw_message.get("mentions") == mentions
    assert event.raw_message["quote"] == quote
    assert event.reply_to_is_own_message is True
    assert adapter._extract_reaction_target(event) == ("+15550005678", 1000000000)


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata", [{}, {"mentions": None}, {"mentions": []}])
async def test_no_mentions_and_admission_policy_are_preserved(adapter, metadata):
    data = {"message": "Johnny said the market closes at 8", **metadata,
            "groupInfo": {"groupId": "allowed"}}
    await adapter._handle_envelope(envelope(data))
    event = adapter.handle_message.await_args.args[0]
    assert event.text == data["message"]
    assert event.raw_message.get("mentions") == []

    adapter.handle_message.reset_mock()
    adapter.require_mention = True
    await adapter._handle_envelope(envelope(data))
    adapter.handle_message.assert_not_awaited()

    adapter.require_mention = False
    data["groupInfo"] = {"groupId": "denied"}
    await adapter._handle_envelope(envelope(data))
    adapter.handle_message.assert_not_awaited()

    data.pop("groupInfo")
    await adapter._handle_envelope(envelope(data))
    event = adapter.handle_message.await_args.args[0]
    assert event.source.chat_type == "dm"
    assert event.raw_message.get("mentions") == []
