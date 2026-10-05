"""Phoneme control spans must not be rewritten as ordinary spoken prose."""
import json
import wave

import pytest

from tools.tts_text_normalize import prepare_spoken_text

SPAN = "<|phoneme_start|>t ah0 m ey1 t ow0<|phoneme_end|>"


def test_phoneme_payload_and_delimiters_survive_cleanup():
    assert prepare_spoken_text(f"**Say** {SPAN} at 2m.") == f"Say {SPAN} at 2 metres."


@pytest.mark.parametrize("payload", ["", "t ah0 m ey1", "a  b\nc", "&amp; _x_ 20%"])
def test_payload_is_opaque_and_repeatable(payload):
    span = f"<|phoneme_start|>{payload}<|phoneme_end|>"
    text = f"Say {span} and {span}."
    assert prepare_spoken_text(text) == text
    assert prepare_spoken_text(prepare_spoken_text(text)) == text


@pytest.mark.parametrize("wrapper", ["<think>{}</think>Visible.", "```\n{}\n```Visible."])
def test_phoneme_span_does_not_resurrect_nonspoken_blocks(wrapper):
    assert prepare_spoken_text(wrapper.format(SPAN)) == "Visible."


def test_placeholder_like_prose_is_not_replaced():
    prefix = "ZZHERMESPHONEMEHOLD0ZZ"
    text = f"{prefix} {SPAN}"
    assert prepare_spoken_text(text) == text


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


def test_registered_tool_passes_phonemes_to_selected_plugin(tmp_path, monkeypatch):
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
    finally:
        tts_registry._reset_for_tests()
