"""Budget refusals identify their subject without disclosing command content."""
import hashlib
import logging

import pytest

from cron import lifecycle_guard as guard


def assert_subject(caplog, text, path=None):
    encoded = text.encode('utf-8', errors='replace')
    assert 'sha256=' + hashlib.sha256(encoded).hexdigest() in caplog.text
    assert f'bytes={len(encoded)}' in caplog.text
    assert f'lines={text.count(chr(10)) + 1}' in caplog.text
    if path is not None:
        assert repr(str(path)) in caplog.text
    assert 'private-token-123' not in caplog.text


@pytest.mark.parametrize('cap', [
    '_MAX_LIFECYCLE_SCAN_BYTES', '_MAX_LIFECYCLE_SCAN_LINES',
    '_MAX_LIFECYCLE_SCAN_LINE_BYTES',
])
def test_root_budget_subject(monkeypatch, caplog, cap):
    monkeypatch.setattr(guard, cap, 1)
    text = 'TOKEN=private-token-123 echo café\nnext\n'
    with caplog.at_level(logging.WARNING):
        unsafe, refusal = guard.scan_gateway_lifecycle(text)
    assert unsafe and 'text at depth 0' in refusal
    assert_subject(caplog, text)


def test_referenced_text_subject(monkeypatch, tmp_path, caplog):
    text = 'echo private-token-123 ' + 'x' * 80
    script = tmp_path / 'payload.sh'
    script.write_text(text, encoding='utf-8')
    monkeypatch.setattr(guard, '_MAX_LIFECYCLE_SCAN_LINE_BYTES', 40)
    with caplog.at_level(logging.WARNING):
        unsafe, refusal = guard.scan_gateway_lifecycle('bash payload.sh', cwd=str(tmp_path))
    assert unsafe and 'text at depth 1' in refusal
    assert_subject(caplog, text, script)


@pytest.mark.parametrize('cap,what', [
    ('_MAX_LIFECYCLE_SCAN_PATHS', 'paths'),
    ('_MAX_LIFECYCLE_SCAN_REMOTE_READS', 'remote reads'),
])
def test_unread_candidate_subject(monkeypatch, tmp_path, caplog, cap, what):
    monkeypatch.setattr(guard, cap, 0)
    calls = []
    text = 'bash missing.sh'
    with caplog.at_level(logging.WARNING):
        unsafe, refusal = guard.scan_gateway_lifecycle(
            text, cwd=str(tmp_path), read_remote_script=lambda path: calls.append(path))
    assert unsafe and f'{what} at depth 0' in refusal
    assert calls == []
    assert_subject(caplog, text, tmp_path / 'missing.sh')


def test_python_cron_subject(monkeypatch, tmp_path, caplog):
    text = '# private-token-123\n' + 'x' * 80
    script = tmp_path / 'payload.py'
    script.write_text(text, encoding='utf-8')
    monkeypatch.setattr(guard, '_MAX_LIFECYCLE_SCAN_LINE_BYTES', 40)
    with caplog.at_level(logging.WARNING), pytest.raises(guard.GatewayLifecycleBlocked):
        guard.check_gateway_lifecycle('run job', str(script))
    assert_subject(caplog, 'run job\n' + text, script)


def test_nested_payload_keeps_source_path(monkeypatch, tmp_path, caplog):
    payload = 'echo private-token-123'
    script = tmp_path / 'nested.sh'
    script.write_text(f"bash -c '{payload}'", encoding='utf-8')
    # Root and file fit individually; the inline payload exhausts the shared bytes.
    monkeypatch.setattr(guard, '_MAX_LIFECYCLE_SCAN_BYTES', 55)
    with caplog.at_level(logging.WARNING):
        unsafe, refusal = guard.scan_gateway_lifecycle('bash nested.sh', cwd=str(tmp_path))
    assert unsafe and 'text at depth 2' in refusal
    assert_subject(caplog, payload, script)


def test_chunked_unicode_fingerprint(monkeypatch, caplog):
    text = 'é' * 65535 + '\ud800' + 'private-token-123\n'
    monkeypatch.setattr(guard, '_MAX_LIFECYCLE_SCAN_BYTES', 1)
    with caplog.at_level(logging.WARNING):
        unsafe, refusal = guard.scan_gateway_lifecycle(text)
    assert unsafe and refusal
    assert_subject(caplog, text)


def test_remote_failure_preserves_direct_fallback(tmp_path, caplog):
    def broken_reader(path):
        raise OSError('fixture remote failure')
    with caplog.at_level(logging.WARNING):
        assert guard.scan_gateway_lifecycle(
            'bash missing.sh', cwd=str(tmp_path), read_remote_script=broken_reader,
        ) == (False, None)
    assert 'falling back to direct-scan verdict' in caplog.text
    assert 'sha256=' not in caplog.text


@pytest.mark.parametrize('text', ['', 'echo ok', 'hermes gateway stop'])
def test_normal_verdicts_do_not_emit_subject(caplog, text):
    with caplog.at_level(logging.WARNING):
        unsafe, refusal = guard.scan_gateway_lifecycle(text)
    assert unsafe == (text == 'hermes gateway stop')
    assert refusal is None
    assert 'sha256=' not in caplog.text
