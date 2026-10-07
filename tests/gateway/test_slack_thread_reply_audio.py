"""Earlier Slack reply voice notes must survive cold hydration (#134486)."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageType
from plugins.platforms.slack.adapter import SlackAdapter


async def _cold_reply_event(tmp_path, mimetype, subtype):
    import importlib
    from pathlib import Path
    from types import SimpleNamespace
    from hermes_cli.plugins import PluginManager
    manager = PluginManager()
    module = manager._load_directory_module(
        SimpleNamespace(path=Path(__file__).resolve().parents[2] / "plugins/platforms/slack"),
        module_name="hermes_plugins.scout_slack_audio",
    )
    loaded_adapter = importlib.import_module(module.__name__ + ".adapter").SlackAdapter
    adapter = loaded_adapter(PlatformConfig(enabled=True, token="fixture"))
    adapter._app = MagicMock()
    adapter._app.client = AsyncMock()
    adapter._bot_user_id = "U_BOT"
    adapter._team_bot_user_ids = {"T_TEAM": "U_BOT"}
    adapter._running = True
    adapter.handle_message = AsyncMock()
    store = MagicMock()
    store._entries = {}
    store.config.group_sessions_per_user = True
    store.get_session_metadata.return_value = ""
    adapter.set_session_store(store)
    adapter._has_active_session_for_thread = MagicMock(return_value=False)
    adapter._register_mentioned_thread = MagicMock()
    adapter._user_name_cache = {("T_TEAM", "U_ALICE"): "Alice", ("T_TEAM", "U_USER"): "User"}
    audio = tmp_path / "voice.m4a"
    audio.write_bytes(b"fixture-audio")
    adapter._download_slack_file = AsyncMock(return_value=str(audio))
    adapter._app.client.conversations_replies.return_value = {"messages": [
        {"ts": "123.000", "user": "U_ALICE", "text": "root"},
        {"ts": "123.100", "user": "U_ALICE", "text": "voice reply", "files": [
            {"id": "F_AUDIO", "name": "voice.m4a", "mimetype": mimetype, "subtype": subtype,
             "url_private_download": "https://files.slack.com/fixture/voice.m4a"}]},
        {"ts": "123.456", "user": "U_USER", "text": "<@U_BOT> summarize"}]}
    await adapter._handle_slack_message({
        "text": "<@U_BOT> summarize", "user": "U_USER", "channel": "C123",
        "ts": "123.456", "thread_ts": "123.000", "channel_type": "channel", "team": "T_TEAM"})
    adapter.handle_message.assert_awaited_once()
    return adapter.handle_message.call_args.args[0], audio


@pytest.mark.asyncio
@pytest.mark.parametrize("mimetype,subtype", [("audio/mp4", ""), ("video/mp4", "slack_audio")])
async def test_cold_reply_audio_reaches_inbound_media(tmp_path, mimetype, subtype):
    event, audio = await _cold_reply_event(tmp_path, mimetype, subtype)
    assert "voice.m4a]" in event.channel_context
    assert event.media_urls == [str(audio)]
    assert event.media_types == ["audio/mp4"]
    assert event.message_type == MessageType.VOICE
    from gateway.run import _event_media_is_stt_input
    assert _event_media_is_stt_input(event, 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "fallback", "empty", "error", "disabled"])
async def test_cold_reply_audio_runs_gateway_transcription(tmp_path, monkeypatch, outcome):
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    from tools import transcription_tools

    event, audio = await _cold_reply_event(tmp_path, "audio/mp4", "")
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=outcome != "disabled")
    runner._should_echo_stt_transcripts = lambda: False
    # Keep reference expansion out of this fixture, not media preparation or STT.
    runner._expand_inbound_context_references = AsyncMock(side_effect=lambda source, key, text: text)
    transcript = "The launch is on Friday."
    result = {"success": True, "transcript": transcript}
    if outcome == "fallback":
        result = {"success": False, "error": "configured provider unavailable"}
    elif outcome == "empty":
        result = {"success": True, "transcript": "  "}
    transcribe = MagicMock(return_value=result)
    if outcome == "error":
        transcribe.side_effect = RuntimeError("fixture provider failure")
    fallback = MagicMock(return_value={"success": True, "transcript": transcript})
    monkeypatch.setattr(transcription_tools, "transcribe_audio", transcribe)
    monkeypatch.setattr(transcription_tools, "transcribe_audio_local_fallback", fallback)
    monkeypatch.setattr("gateway.run._probe_audio_duration", AsyncMock(return_value=None))

    prepared = await runner._prepare_inbound_message_text(
        event=event, source=event.source, history=[], session_key="fixture-thread",
    )
    assert "summarize" in prepared
    assert "Alice" in prepared
    assert "voice.m4a]" in prepared
    if outcome == "disabled":
        transcribe.assert_not_called()
        assert str(audio) in prepared
    else:
        transcribe.assert_called_once_with(str(audio), None, "gateway")
        if outcome in {"success", "fallback"}:
            assert transcript in prepared
        elif outcome == "empty":
            assert "empty or inaudible" in prepared
        else:
            assert "could not be transcribed automatically" in prepared
    if outcome == "fallback":
        fallback.assert_called_once_with(str(audio))
    else:
        fallback.assert_not_called()


@pytest.mark.asyncio
async def test_reply_audio_boundaries_and_download_failure():
    from types import SimpleNamespace
    from plugins.platforms.slack.thread_audio import collect_reply_audio

    def message(ts, name, user="human", mime="audio/mp4"):
        return {"ts": ts, "user": user, "files": [{"id": name, "name": name,
            "mimetype": mime, "url_private_download": "https://files.slack.com/" + name}]}

    adapter = SlackAdapter(PlatformConfig(enabled=True, token="fixture"))
    adapter._bot_user_id = "self"
    key = adapter._thread_cache_key("channel", "1.0", "team")
    messages = [message("1.0", "root"), message("9.0", "trigger"),
                message("10.0", "future"), message("NaN", "invalid"),
                message("8.9", "bot", "self"), message("8.8", "image", mime="image/png"),
                {"ts": "8.7", "files": None}, None]
    messages += [message(str(i), str(i)) for i in range(2, 8)]
    adapter._thread_context_cache[key] = SimpleNamespace(messages=messages)
    async def cache(kind, file, url, mimetype, team):
        assert team == "team"
        if file["id"] == "5":
            raise OSError("fixture failure")
        return file["id"], mimetype, ""
    adapter._cache_slack_file = AsyncMock(side_effect=cache)
    paths, types = await collect_reply_audio(adapter, "channel", "1.0", "9.0", "team")
    assert paths == ["4", "6", "7"]
    assert types == ["audio/mp4"] * len(paths)
    assert adapter._cache_slack_file.await_count == 4
    adapter._cache_slack_file.reset_mock()
    assert await collect_reply_audio(adapter, "channel", "1.0", "9.0", "other-team") == ([], [])
    assert await collect_reply_audio(adapter, "channel", "1.0", "invalid", "team") == ([], [])
    adapter._cache_slack_file.assert_not_awaited()
