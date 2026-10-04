"""Script-only tool listings describe execution, without rewriting stored pins."""
import json

import pytest

from cron import jobs
from tools import cronjob_tools  # noqa: F401 -- register the real tool
from tools.registry import registry


@pytest.fixture
def stored_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr(jobs, "JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", tmp_path / "cron" / "output")
    # Liveness is unrelated to how persisted job metadata is presented.
    monkeypatch.setattr("hermes_cli.cron._builtin_gateway_liveness", lambda: True)
    return jobs.JOBS_FILE


def listing():
    return json.loads(registry.get_entry("cronjob_manage").handler({"action": "list"}))


@pytest.mark.parametrize("no_agent", [True, False])
def test_list_only_reports_operational_model_pins(stored_jobs, no_agent):
    record = {
        "id": "fixture-job", "name": "Fixture", "enabled": True,
        "script": "report.py", "no_agent": no_agent,
        "schedule": jobs.parse_schedule("every 1h"),
        "model": "stored-model", "provider": "stored-provider",
        "base_url": "https://provider.example/v1", "prompt": "Summarize output",
    }
    jobs.save_jobs([record])
    before = stored_jobs.read_bytes()
    result = listing()
    assert result["success"] is True
    displayed = result["jobs"][0]
    assert displayed["model"] == (None if no_agent else record["model"])
    assert displayed["provider"] == (None if no_agent else record["provider"])
    assert displayed["base_url"] == (None if no_agent else record["base_url"])
    assert displayed["pinned"] is (not no_agent)
    assert displayed.get("no_agent", False) is no_agent
    assert displayed["script"] == record["script"]
    assert stored_jobs.read_bytes() == before
    if no_agent:
        # Returning to agent mode recovers the original route, not nulls copied
        # into storage by a presentation-only operation.
        jobs.update_job(record["id"], {"no_agent": False})
        restored = listing()["jobs"][0]
        assert restored["model"] == record["model"]
        assert restored["provider"] == record["provider"]
        assert restored["base_url"] == record["base_url"]
        assert restored["pinned"] is True


def test_empty_list_and_legacy_unpinned_job(stored_jobs, monkeypatch):
    jobs.save_jobs([])
    assert listing()["jobs"] == []
    jobs.save_jobs([{"id": "legacy", "enabled": True}])
    displayed = listing()["jobs"][0]
    assert displayed["model"] is None
    assert displayed["provider"] is None
    assert displayed["base_url"] is None
    assert displayed["pinned"] is False

    def unavailable(**kwargs):
        raise OSError("fixture store unavailable")

    monkeypatch.setattr(cronjob_tools, "list_jobs", unavailable)
    failed = listing()
    assert failed["success"] is False
    assert "fixture store unavailable" in failed["error"]
