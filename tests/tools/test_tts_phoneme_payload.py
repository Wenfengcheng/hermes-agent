"""Phoneme control spans must not be rewritten as ordinary spoken prose."""
import json
import wave

import pytest

from tools.tts_text_normalize import prepare_spoken_text

SPAN = "<|phoneme_start|>t ah0 m ey1 t ow0<|phoneme_end|>"


def test_phoneme_payload_and_delimiters_survive_cleanup():
    assert prepare_spoken_text(f"**Say** {SPAN} at 2m.") == f"Say {SPAN} at 2 metres."


@pytest.mark.parametrize("payload", ["", "t ah0 m ey1", "&amp; _x_ 20%"])
def test_payload_is_opaque_and_repeatable(payload):
    span = f"<|phoneme_start|>{payload}<|phoneme_end|>"
    text = f"Say {span} and {span}."
    assert prepare_spoken_text(text) == text
    assert prepare_spoken_text(prepare_spoken_text(text)) == text


@pytest.mark.parametrize("wrapper", ["<think>{}</think>Visible.", "```\n{}\n```Visible."])
def test_phoneme_span_does_not_resurrect_nonspoken_blocks(wrapper):
    assert prepare_spoken_text(wrapper.format(SPAN)) == "Visible."


@pytest.mark.parametrize("prefix", [
    "ZZHERMESPHONEMEHOLD0ZZ", "ZZHERMESPHONEME&#72;OLD0ZZ",
    "ZZHERMESPHONEME**H**OLD0ZZ", "\ue000A\ue001", "&#57344;A&#57345;",
])
def test_placeholder_like_prose_is_not_replaced(prefix):
    expected_prefix = prepare_spoken_text(prefix)
    text = f"{prefix} {SPAN}"
    assert prepare_spoken_text(text) == f"{expected_prefix} {SPAN}"


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
def test_phoneme_newlines_become_spaces_not_provider_truncation(newline):
    raw = f"<|phoneme_start|>ah0{newline}m<|phoneme_end|>"
    assert prepare_spoken_text(raw) == "<|phoneme_start|>ah0 m<|phoneme_end|>"


@pytest.mark.parametrize("cap", [1, 4, 10, 30, 60, 200, None, 0])
def test_explicit_cap_never_splits_phoneme_span(cap):
    text = f"Say {SPAN} tomato."
    result = prepare_spoken_text(text, max_chars=cap)
    if cap and cap > 0:
        assert len(result) <= cap
    if "<|" in result:
        assert SPAN in result
    if cap is None or cap == 0 or cap >= len(text):
        assert result == text


def test_plain_text_and_empty_input_keep_existing_cleanup():
    assert prepare_spoken_text("") == ""
    assert prepare_spoken_text("**Hi** 2m | 20%") == "Hi 2 metres; 20 percent"


@pytest.mark.parametrize("introduction", ["A short introduction. ", "A short introduction "])
def test_chunk_boundary_keeps_a_complete_phoneme_span(introduction):
    from tools.tts_tool_delivery import _split_text_for_tts

    text = introduction + SPAN + " tomato."
    chunks = _split_text_for_tts(text, len(SPAN) + 2)
    assert SPAN in chunks
    assert all(len(chunk) <= len(SPAN) + 2 for chunk in chunks)
    assert " ".join(chunks) == text


def test_oversized_phoneme_span_reports_provider_limit():
    from tools.tts_tool_delivery import _split_text_for_tts

    with pytest.raises(ValueError, match="phoneme.*limit"):
        _split_text_for_tts(SPAN, len(SPAN) - 1)


@pytest.mark.parametrize("span", [SPAN, "<|phoneme_start|>t  ah0\tm<|phoneme_end|>"])
def test_registered_tool_passes_phonemes_to_selected_plugin(tmp_path, monkeypatch, span):
    SPAN = span
    from agent import tts_registry
    from agent.tts_provider import TTSProvider
    from tools import tts_tool  # registers the real tool handler
    from tools.registry import registry

    received = []

    class RecordingProvider(TTSProvider):
        name = "phoneme-fixture"

        def synthesize(self, text, output_path, **kwargs):
            received.append(text)
            with wave.open(str(output_path), "wb") as out:
                out.setnchannels(1)
                out.setsampwidth(2)
                out.setframerate(8000)
                out.writeframes(b"\x00\x00" * 80)
            return str(output_path)

    tts_registry._reset_for_tests()
    try:
        tts_registry.register_provider(RecordingProvider())
        monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "edge"})
        result = json.loads(registry.dispatch("text_to_speech", {
            "text": f"Say {SPAN} tomato.",
            "provider": "phoneme-fixture",
            "output_path": str(tmp_path / "speech.wav"),
        }))
        assert result["success"], result
        assert received == [f"Say {SPAN} tomato."]
        received.clear()
        monkeypatch.setattr(tts_tool, "_resolve_max_text_length", lambda *args: len(SPAN) + 2)
        monkeypatch.setattr(tts_tool, "_build_audio_delivery_files", lambda paths, *args, **kwargs: (paths, False))
        result = json.loads(registry.dispatch("text_to_speech", {
            "text": "A short introduction " + SPAN + " tomato.",
            "provider": "phoneme-fixture",
            "output_path": str(tmp_path / "chunks.wav"),
        }))
        assert result["success"], result
        assert SPAN in received
        assert all(len(chunk) <= len(SPAN) + 2 for chunk in received)
        received.clear()
        monkeypatch.setattr(tts_tool, "_resolve_max_text_length", lambda *args: len(SPAN) - 1)
        result = json.loads(tts_tool.text_to_speech_tool(SPAN, provider="phoneme-fixture"))
        assert result["success"] is False
        assert "phoneme" in result["error"]
        assert not received
    finally:
        tts_registry._reset_for_tests()
