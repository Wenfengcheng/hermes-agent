"""OpenCode Go's Responses wire must retain native tool images (#132837)."""
import asyncio
import json

import pytest

from agent.auxiliary_client import scoped_runtime_main
from agent.vision_message_prep import VisionMessagePrepMixin
from tools import vision_tools
from tools.registry import registry


@pytest.mark.parametrize('mode,native', [('codex_responses', True), ('chat_completions', False), ('anthropic_messages', False), ('', False)])
def test_go_tool_image_reaches_wire_without_aux(tmp_path, monkeypatch, mode, native):
    from PIL import Image
    from agent.codex_responses_adapter import _chat_messages_to_responses_input
    image = tmp_path / 'pixel.png'
    Image.new('RGB', (1, 1), 'red').save(image)
    model = 'fixture-vision-model'
    cfg = {'model': {'provider': 'opencode-go', 'model': model, 'supports_vision': True}}
    monkeypatch.setattr('hermes_cli.config.load_config', lambda: cfg)
    async def aux(*args, **kwargs):
        return json.dumps({'analysis': 'auxiliary description'})
    monkeypatch.setattr(vision_tools, 'vision_analyze_tool', aux)
    with scoped_runtime_main({'provider': 'opencode-go', 'model': model, 'api_mode': mode}):
        result = asyncio.run(registry.get_entry('vision_analyze').handler({'image_url': str(image), 'question': 'Color?'}))
    if not native:
        assert json.loads(result)['analysis'] == 'auxiliary description'
        return
    assert isinstance(result, dict) and result.get('_multimodal') is True
    agent = VisionMessagePrepMixin()
    agent.provider, agent.model, agent.api_mode = 'opencode-go', model, mode
    content = agent._tool_result_content_for_active_model('vision_analyze', result)
    assert isinstance(content, list)
    wire = _chat_messages_to_responses_input([
        {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'call_image', 'type': 'function', 'function': {'name': 'vision_analyze', 'arguments': '{}'}}]},
        {'role': 'tool', 'tool_call_id': 'call_image', 'content': content},
    ])
    output = next(x['output'] for x in wire if x['type'] == 'function_call_output')
    assert any(x['type'] == 'input_image' for x in output)


def test_transport_exception_is_route_local_and_preserves_fallbacks(tmp_path, monkeypatch):
    from providers import ProviderProfile, routed_model_rejects_vision_tool_messages
    from tools.computer_use.vision_routing import should_route_capture_to_aux_vision
    model = 'fixture-vision-model'
    cfg = {'model': {'provider': 'opencode-go', 'model': model, 'supports_vision': True}}
    monkeypatch.setattr('hermes_cli.config.load_config', lambda: cfg)
    route = {'provider': 'opencode-go', 'model': model, 'api_mode': 'codex_responses'}
    assert ProviderProfile(name='legacy', supports_vision_tool_messages=False).supports_vision_tool_messages_by_api_mode == {}
    for provider in ('xiaomi', 'openrouter'):
        assert routed_model_rejects_vision_tool_messages(provider, 'xiaomi/mimo-v2.5', api_mode='codex_responses')
    with scoped_runtime_main(route):
        assert not vision_tools._profile_rejects_tool_media('opencode-go', model)
        assert vision_tools._profile_rejects_tool_media('opencode-go', 'other-model')
        assert not should_route_capture_to_aux_vision('opencode-go', model, cfg)
        with scoped_runtime_main({**route, 'api_mode': 'chat_completions'}):
            assert vision_tools._profile_rejects_tool_media('opencode-go', model)
        assert not vision_tools._profile_rejects_tool_media('opencode-go', model)
        cfg['auxiliary'] = {'vision': {'provider': 'chosen', 'model': 'chosen-vision'}}
        assert not vision_tools._should_use_native_vision_fast_path()
        assert should_route_capture_to_aux_vision('opencode-go', model, cfg)
        cfg.pop('auxiliary')
        cfg['model']['supports_vision'] = False
        assert not vision_tools._should_use_native_vision_fast_path()
        cfg['model']['supports_vision'] = True
        missing = asyncio.run(registry.get_entry('vision_analyze').handler({'image_url': str(tmp_path / 'missing.png'), 'question': '?'}))
        assert json.loads(missing)['success'] is False
    with scoped_runtime_main({'provider': 'opencode-go', 'model': model}):
        assert vision_tools._profile_rejects_tool_media('opencode-go', model)
    from agent import auxiliary_client
    monkeypatch.setattr(auxiliary_client, '_RUNTIME_MAIN_PROVIDER', 'opencode-go')
    monkeypatch.setattr(auxiliary_client, '_RUNTIME_MAIN_MODEL', model)
    monkeypatch.setattr(auxiliary_client, '_RUNTIME_MAIN_API_MODE', 'codex_responses')
    token = auxiliary_client._RUNTIME_MAIN_CONTEXT.set(None)
    try:
        assert vision_tools._profile_rejects_tool_media('opencode-go', model)
    finally:
        auxiliary_client._RUNTIME_MAIN_CONTEXT.reset(token)
