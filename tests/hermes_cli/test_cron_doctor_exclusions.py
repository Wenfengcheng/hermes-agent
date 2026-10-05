"""Watchdog recovery and bounded diagnostics for #133135."""

from argparse import ArgumentParser, Namespace

import pytest

from cron import jobs
from hermes_cli.cron import cron_command


@pytest.fixture
def watchdog(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(jobs, "CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr(jobs, "JOBS_FILE", tmp_path / "cron/jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", tmp_path / "cron/output")
    job = jobs.create_job(prompt="watch other jobs", schedule="every 1h")
    jobs.mark_job_run(job["id"], success=False, error="previous watchdog findings")
    return job


def test_watchdog_can_recover_without_hiding_other_failures(watchdog, capsys):
    from hermes_cli.subcommands.cron import build_cron_parser

    parser = ArgumentParser()
    build_cron_parser(parser.add_subparsers(), cmd_cron=cron_command)
    args = parser.parse_args(["cron", "doctor", "--exclude", watchdog["id"],
                              "--exclude", watchdog["id"]])
    assert cron_command(args) == 0
    assert "previous watchdog findings" not in capsys.readouterr().out
    # Exclusion is read-only: the normal operator view still reports the failure.
    assert jobs.get_job(watchdog["id"])["last_status"] != "ok"
    assert cron_command(Namespace(cron_command="doctor")) == 1
    assert cron_command(Namespace(cron_command="doctor", exclude=[watchdog["id"][:4],
                                                                watchdog["name"]])) == 1
    capsys.readouterr()
    other = jobs.create_job(prompt="work", schedule="every 1h")
    jobs.mark_job_run(other["id"], success=False, error="other job failed")
    assert cron_command(args) == 1
    output = capsys.readouterr().out
    assert "other job failed" in output
    assert "previous watchdog findings" not in output
    jobs.mark_job_run(other["id"], success=True)
    assert cron_command(args) == 0


def test_doctor_bounds_failure_output_without_rewriting_history(watchdog, capsys):
    detail = "nested report " * 100 + "\nprevious day's full output"
    jobs.mark_job_run(watchdog["id"], success=False, error=detail)
    assert cron_command(Namespace(cron_command="doctor")) == 1
    output = capsys.readouterr().out
    line = next(line for line in output.splitlines() if "last run failed:" in line)
    assert len(line) < 200
    assert "previous day's full output" not in output
    assert jobs.get_job(watchdog["id"])["last_error"] == detail
