"""Mount-target findings stay reviewable; host password-file access stays blocked."""
import pytest

from tools.plugin_guard import scan_plugin, should_allow_plugin_install


@pytest.mark.parametrize('option', ['--bind', '--ro-bind', '--bind-try', '--ro-bind-try'])
@pytest.mark.parametrize('path', ['/etc/passwd', '/etc/shadow'])
def test_python_mount_target_is_confirmable(tmp_path, option, path):
    (tmp_path / 'contained.py').write_text(
        f'identity_binds = ["{option}", str(identity / "passwd"), "{path}"]\n',
        encoding='utf-8',
    )
    result = scan_plugin(tmp_path)
    findings = [f for f in result.findings if f.pattern_id == 'system_passwd_access']
    assert findings and all(f.severity == 'high' for f in findings)
    assert result.verdict == 'caution'
    assert should_allow_plugin_install(result)[0] is None
    assert should_allow_plugin_install(result, force=True)[0] is True


@pytest.mark.parametrize('line', [
    'identity_binds = ["--ro-bind", str(identity / "passwd"), "/etc/passwd", "--ro-bind", str(identity / "group"), "/etc/group"]',
    'identity_binds = ("--bind", src, "/etc/passwd", "--bind", other, "/etc/shadow")',
    '路径 = ["--ro-bind", "临时文件", "/etc/passwd"]',
    'identity_binds = ["--ro-bind", str(identity / "passwd"), "/etc/passwd",\n                  "--ro-bind", str(identity / "group"), "/etc/group"]',
    '路径 = [\n    "--ro-bind", "临时文件", "/etc/passwd"\n]',
    '    args = ["--ro-bind", src, "/etc/passwd"]',
])
def test_complete_bind_groups_keep_findings_and_locations(tmp_path, line):
    (tmp_path / 'contained.py').write_text(line + '\n', encoding='utf-8')
    result = scan_plugin(tmp_path)
    findings = [f for f in result.findings if f.pattern_id == 'system_passwd_access']
    assert findings and all(f.severity == 'high' for f in findings)
    assert all(f.file == 'contained.py' and '/etc/' in line.splitlines()[f.line - 1] for f in findings)
    assert should_allow_plugin_install(result)[0] is None


@pytest.mark.parametrize('line', [
    'args = ["--ro-bind", "/etc/passwd", "/tmp/passwd"]',
    'args = ["--ro-bind", "/etc/passwd", "/etc/passwd"]',
    'args = ["--ro-bind", src, "/etc/passwd", "/etc/shadow"]',
    'args = ["--ro-bind", open("data").read(), "/etc/passwd"]',
    'args = ["--ro-bind", src, "/etc/passwd"]; open("/etc/shadow")',
    'args = ["--ro-bind", src, "/etc/passwd"]; run(args)',
    'args = ["--ro-bind", src, "/etc/passwd", "--ro-bind", "/etc/shadow", dest]',
    'args = ["--ro-bind", src, "/etc/passwd", "extra /etc/shadow"]',
    'args = ["--ro-bind", src, "/etc/passwd"] # /etc/shadow',
    'args = ["--ro-bind", *src, "/etc/passwd"]',
    'args = ["--ro-bind", src, "/etc/passwd"] + other',
    'args = ["--ro-bind", src, "/etc/passwd"',
    'args = ["--ro-bind", src, "/etc/passwd/backup"]',
    'args = ["--ro-bind", src, "/etc/passwd",\n        "--ro-bind", "/etc/shadow", dst]',
    'args = ["--ro-bind", src, "/etc/passwd",\n        "--bind", open("secret"), "/tmp/out"]',
    'args = ["--ro-bind", src, "/etc/passwd",\n        "--bind", other, "/tmp/out"] + extra',
])
def test_ambiguous_or_host_access_is_still_blocked(tmp_path, line):
    (tmp_path / 'contained.py').write_text(line + '\n', encoding='utf-8')
    result = scan_plugin(tmp_path)
    assert any(f.pattern_id == 'system_passwd_access' and f.severity == 'critical'
               for f in result.findings)
    assert should_allow_plugin_install(result, force=True)[0] is False
