"""Toolset inspection must disclose profile-local partial policy suppression."""
import json
from pathlib import Path

import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    from fastapi import FastAPI
    from starlette.testclient import TestClient
    from hermes_cli import profiles, tools_config
    from hermes_cli.web_routers.tools import router
    from hermes_constants import get_hermes_home

    default = get_hermes_home()
    root = default / 'profiles'
    worker = root / 'worker'
    worker.mkdir(parents=True)
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setattr(profiles, '_get_default_hermes_home', lambda: default)
    monkeypatch.setattr(profiles, '_get_profiles_root', lambda: root)
    for home, disabled in [(default, []), (worker, ['search'])]:
        (home / 'config.yaml').write_text(json.dumps({
            'platform_toolsets': {'cli': ['web']},
            'agent': {'disabled_toolsets': disabled},
        }), encoding='utf8')
    # No entitlement/network/credential discovery is needed to inspect policy.
    monkeypatch.setattr(tools_config, 'get_nous_subscription_features', lambda config: {})
    monkeypatch.setattr(tools_config, '_toolset_has_keys', lambda *a, **k: True)
    app = FastAPI()
    app.include_router(router)
    return TestClient(app), default, worker


@pytest.mark.parametrize('disabled', [['search'], '["search"]'])
def test_partial_policy_description_is_profile_local(client, disabled):
    http, default, worker = client
    config = json.loads((worker / 'config.yaml').read_text(encoding='utf8'))
    config['agent']['disabled_toolsets'] = disabled
    (worker / 'config.yaml').write_text(json.dumps(config), encoding='utf8')
    original = [(p / 'config.yaml').read_bytes() for p in (default, worker)]
    rows = []
    for profile in ('default', 'worker', 'default'):
        response = http.get('/api/tools/toolsets', params={'profile': profile})
        assert response.status_code == 200
        rows.append(next(row for row in response.json() if row['name'] == 'web'))
    assert rows[1]['enabled'] is True
    assert 'partial' in rows[1]['description']
    assert 'web_search' in rows[1]['description']
    assert 'agent.disabled_toolsets' in rows[1]['description']
    assert rows[0] == rows[2]
    assert 'partial' not in rows[0]['description']
    assert [(p / 'config.yaml').read_bytes() for p in (default, worker)] == original


@pytest.mark.parametrize('disabled, enabled', [([], True), (['web'], False), (['unknown-scout-toolset'], True)])
def test_nonpartial_policy_preserves_description_and_selection(client, disabled, enabled):
    http, _, worker = client
    config = json.loads((worker / 'config.yaml').read_text(encoding='utf8'))
    config['agent']['disabled_toolsets'] = disabled
    (worker / 'config.yaml').write_text(json.dumps(config), encoding='utf8')
    response = http.get('/api/tools/toolsets', params={'profile': 'worker'})
    assert response.status_code == 200
    row = next(row for row in response.json() if row['name'] == 'web')
    assert row['enabled'] is enabled
    assert 'partial' not in row['description']


def test_unknown_profile_is_not_replaced_by_default(client):
    http, _, _ = client
    assert http.get('/api/tools/toolsets', params={'profile': 'missing'}).status_code == 404

