"""SVG receive regression: real adapter/cache, only the download is substituted."""
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.event import MessageType
from plugins.platforms.slack.adapter import SlackAdapter

SVG = b'<svg xmlns="http://www.w3.org/2000/svg"><text>fixture</text></svg>'


@pytest.fixture
def adapter():
    a = SlackAdapter(PlatformConfig(enabled=True, token="fixture"))
    a._app = MagicMock()
    a._app.client = AsyncMock()
    a._app.client.users_info.return_value = {
        "user": {"is_bot": False, "profile": {"display_name": "Fixture"}}}
    a._bot_user_id = "U_BOT"
    a._running = True
    a.handle_message = AsyncMock()
    a._download_slack_file_bytes = AsyncMock(return_value=SVG)
    return a


def event(mime="image/svg+xml", name="drawing.svg", size=len(SVG)):
    return {"type": "message", "user": "U1", "text": "read this",
            "channel": "D1", "channel_type": "im", "ts": "123.001",
            "files": [{"id": "F1", "name": name, "mimetype": mime,
                       "size": size, "url_private_download": "https://files.slack.com/fixture"}]}


@pytest.mark.asyncio
@pytest.mark.parametrize("mime,name", [
    ("image/svg+xml", "drawing.svg"),
    ("image/svg+xml; charset=utf-8", "drawing.SVG"),
    ("image/svg+xml", ""),
])
async def test_svg_reaches_message_as_readable_document(adapter, mime, name):
    await adapter._handle_slack_message(event(mime, name))
    adapter.handle_message.assert_awaited_once()
    message = adapter.handle_message.call_args.args[0]
    assert len(message.media_urls) == 1, message.text
    cached = Path(message.media_urls[0])
    assert cached.read_bytes() == SVG
    assert cached.suffix.lower() == ".svg"
    assert message.message_type == MessageType.DOCUMENT
    assert message.media_types == ["application/xml"]  # downstream image dispatch must not reopen it
    assert "scope" not in message.text.lower()


@pytest.mark.asyncio
async def test_svg_image_hint_also_uses_document_cache(adapter):
    f = event()["files"][0]
    cached = await adapter._cache_slack_file("image", f, f["url_private_download"], f["mimetype"], "T2")
    assert Path(cached[0]).read_bytes() == SVG
    assert cached[1] == "application/xml"
    adapter._download_slack_file_bytes.assert_awaited_once_with(f["url_private_download"], team_id="T2")


@pytest.mark.asyncio
async def test_invalid_image_notice_does_not_blame_permissions(adapter):
    adapter._download_slack_file_bytes.return_value = b"not a supported bitmap"
    await adapter._handle_slack_message(event("image/png", "bad.png"))
    message = adapter.handle_message.call_args.args[0]
    assert not message.media_urls
    assert "unsupported" in message.text.lower()
    assert "scope" not in message.text.lower()
    assert "auth" not in message.text.lower()


@pytest.mark.asyncio
async def test_png_still_uses_image_cache(adapter):
    data = b"\x89PNG\r\n\x1a\nfixture"
    adapter._download_slack_file_bytes.return_value = data
    await adapter._handle_slack_message(event("image/png", "image.png", len(data)))
    message = adapter.handle_message.call_args.args[0]
    assert message.message_type == MessageType.PHOTO
    assert message.media_types == ["image/png"]
    assert Path(message.media_urls[0]).read_bytes() == data


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [0, 20 * 1024 * 1024 + 1])
async def test_svg_keeps_document_size_gate(adapter, size):
    await adapter._handle_slack_message(event(size=size))
    assert not adapter.handle_message.call_args.args[0].media_urls
    adapter._download_slack_file_bytes.assert_not_awaited()


@pytest.mark.asyncio
async def test_svg_empty_download_keeps_bytes_without_image_validation(adapter):
    adapter._download_slack_file_bytes.return_value = b""
    await adapter._handle_slack_message(event())
    message = adapter.handle_message.call_args.args[0]
    assert message.message_type == MessageType.DOCUMENT
    assert Path(message.media_urls[0]).read_bytes() == b""


@pytest.mark.asyncio
async def test_svg_download_failure_preserves_notice_and_no_attachment(adapter):
    adapter._download_slack_file_bytes.side_effect = ValueError("Slack returned HTML instead of media")
    await adapter._handle_slack_message(event())
    message = adapter.handle_message.call_args.args[0]
    assert not message.media_urls
    assert "scope" in message.text
    assert "read this" in message.text


def test_actual_html_failure_keeps_auth_diagnostic(adapter):
    detail = adapter._describe_slack_download_failure(ValueError("Slack returned HTML instead of media"))
    assert "scope" in detail


def test_unknown_failure_keeps_existing_fallback(adapter):
    assert adapter._describe_slack_download_failure(RuntimeError("fixture")) is None
