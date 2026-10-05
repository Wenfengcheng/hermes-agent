"""Full-backup failures remain inspectable beyond the console cap (#105868)."""
import json
import zipfile
from types import SimpleNamespace


def test_backup_records_every_failed_member(tmp_path, monkeypatch, capsys):
    from hermes_cli import backup

    home = tmp_path / 'home'
    home.mkdir()
    (home / 'good.txt').write_text('recover me', encoding='utf8')
    failed = {f'failed-{i}.txt' for i in range(17)}
    for name in failed:
        (home / name).write_text('fixture', encoding='utf8')
    archive = tmp_path / 'backup.zip'
    report = tmp_path / 'failures.json'
    monkeypatch.setattr(backup, 'get_default_hermes_root', lambda: home)
    monkeypatch.setattr(backup, '_collect_external_entries', lambda: ([], []))
    real_write = zipfile.ZipFile.write

    def fail_selected(self, filename, arcname=None, **kwargs):
        if str(arcname) in failed:
            raise PermissionError('synthetic locked file')
        return real_write(self, filename, arcname, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, 'write', fail_selected)
    result = backup.run_backup(SimpleNamespace(output=str(archive), keep=0, error_report=str(report)))
    assert result is False
    with zipfile.ZipFile(archive) as zf:
        assert zf.read('good.txt') == b'recover me'
        assert not failed.intersection(zf.namelist())
    assert '7 more' in capsys.readouterr().out
    assert report.is_file(), 'full failure details must survive the console cap'
    data = json.loads(report.read_text(encoding='utf8'))
    assert {item['path'] for item in data['errors']} == failed
    assert all(item['reason'] == 'synthetic locked file' for item in data['errors'])
    assert data['archive'] == str(archive)
    assert data['complete'] is False


def test_report_boundaries_and_dispatch(tmp_path, monkeypatch):
    from argparse import ArgumentParser
    import pytest
    from hermes_cli import backup
    from hermes_cli.main import cmd_backup
    from hermes_cli.subcommands.backup import build_backup_parser

    home = tmp_path / 'home'
    home.mkdir()
    monkeypatch.setattr(backup, 'get_default_hermes_root', lambda: home)
    monkeypatch.setattr(backup, '_collect_external_entries', lambda: ([], []))
    parser = ArgumentParser()
    build_backup_parser(parser.add_subparsers(), cmd_backup=cmd_backup)

    def run(report, *extra):
        args = parser.parse_args(['backup', '-o', str(tmp_path / 'backup.zip'),
                                 '--error-report', str(report), *extra])
        args.func(args)

    empty_report = tmp_path / 'empty.json'
    run(empty_report)
    assert json.loads(empty_report.read_text())['archive'] is None
    (home / 'good.txt').write_text('recover me')
    good_report = tmp_path / 'good.json'
    run(good_report)
    assert json.loads(good_report.read_text())['errors'] == []
    for protected in [good_report, home / 'new.json', tmp_path / 'backup.zip']:
        with pytest.raises(SystemExit):
            run(protected)
    with pytest.raises(SystemExit):
        run(tmp_path / 'quick.json', '--quick')
    assert not (tmp_path / 'quick.json').exists()
    assert not (home / 'new.json').exists()
    assert json.loads(good_report.read_text())['errors'] == []

    # Publication failure preserves the archive and causes a failing CLI status.
    import hermes_cli.backup_error_report as reporting
    def deny_link(*args):
        raise PermissionError('fixture report failure')
    monkeypatch.setattr(reporting.os, 'link', deny_link)
    with pytest.raises(SystemExit) as error:
        run(tmp_path / 'unwritten.json')
    assert error.value.code == 1
    with zipfile.ZipFile(tmp_path / 'backup.zip') as zf:
        assert zf.read('good.txt') == b'recover me'
    assert not list(tmp_path.glob('.backup-errors-*'))


def test_report_cannot_use_managed_backup_name(tmp_path):
    import pytest
    from hermes_cli.backup_error_report import BackupErrorReport

    home = tmp_path / 'home'
    home.mkdir()
    with pytest.raises(ValueError, match='retention'):
        BackupErrorReport(tmp_path / 'hermes-backup-errors.zip', tmp_path / 'backup.zip', home)


def test_sqlite_and_external_failures_share_report(tmp_path, monkeypatch):
    from hermes_cli import backup
    home = tmp_path / 'home'
    home.mkdir()
    (home / 'state.db').write_bytes(b'fixture snapshot failure')
    external = tmp_path / 'missing.txt'
    monkeypatch.setattr(backup, 'get_default_hermes_root', lambda: home)
    monkeypatch.setattr(backup, '_collect_external_entries',
                        lambda: ([(external, '_external/missing.txt')], []))
    monkeypatch.setattr(backup, '_zip_sqlite_snapshot', lambda *args: None)
    report = tmp_path / 'failures.json'
    assert backup.run_backup(SimpleNamespace(output=str(tmp_path / 'backup.zip'),
                                             keep=0, error_report=str(report))) is False
    data = json.loads(report.read_text())
    assert {item['path'] for item in data['errors']} == {'state.db', '_external/missing.txt'}
    assert any(item['reason'] == 'SQLite safe copy failed' for item in data['errors'])
