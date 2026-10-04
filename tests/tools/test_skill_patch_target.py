"""Explicit invalid patch targets must never select the default skill file."""
import json
from unittest.mock import patch

import pytest

from tools.skill_manager_tool import _patch_skill, apply_skill_pending, skill_manage


@pytest.fixture
def skill(tmp_path):
    root = tmp_path / "skills"
    directory = root / "target-probe"
    (directory / "references").mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        "---\nname: target-probe\ndescription: Test explicit patch targets.\n---\n\nStep one.\n",
        encoding="utf-8",
    )
    (directory / "references" / "guide.md").write_text("Step one.\n", encoding="utf-8")
    with patch("tools.skill_manager_tool.SKILLS_DIR", root), patch(
        "agent.skill_utils.get_all_skills_dirs", return_value=[root]
    ):
        yield directory


def invoke(entry, **kwargs):
    payload = dict(action="patch", name="target-probe", old_string="Step one.",
                   new_string="Step two.", **kwargs)
    if entry == "helper":
        payload.pop("action")
        return _patch_skill(**payload)
    if entry == "batch":
        return json.loads(skill_manage(action="", name="", operations=[payload]))
    if entry == "approved":
        return json.loads(apply_skill_pending(payload))
    return json.loads(skill_manage(**payload))


@pytest.mark.parametrize("entry", ["helper", "flat", "batch", "approved"])
@pytest.mark.parametrize("file_path", ["", "   "])
def test_invalid_target_preserves_all_files(skill, entry, file_path):
    before = {p.relative_to(skill): p.read_bytes() for p in skill.rglob("*") if p.is_file()}
    result = invoke(entry, file_path=file_path)
    assert result["success"] is False, result
    assert "file_path" in result["error"] or "File must be under" in result["error"]
    if file_path == "":
        assert "references/" in result["error"]
    assert {p.relative_to(skill): p.read_bytes() for p in skill.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("entry", ["flat", "batch", "approved"])
@pytest.mark.parametrize("target", ["omitted", None, "references/guide.md"])
def test_only_selected_target_changes(skill, entry, target):
    args = {} if target == "omitted" else {"file_path": target}
    result = invoke(entry, **args)
    assert result["success"] is True, result
    selected = "references/guide.md" if target == "references/guide.md" else "SKILL.md"
    for rel in ("SKILL.md", "references/guide.md"):
        text = (skill / rel).read_text(encoding="utf-8")
        assert ("Step two." in text) == (rel == selected)
        assert ("Step one." in text) == (rel != selected)
