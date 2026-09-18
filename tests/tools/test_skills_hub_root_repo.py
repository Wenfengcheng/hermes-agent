"""Root-layout skills.sh entries exercise both real source adapters."""

import httpx
import pytest

from tools.skills_hub_github import GitHubAuth
from tools.skills_hub_skillssh import SkillsShSource


@pytest.fixture
def source(monkeypatch):
    # Only the network boundary is replaced; resolution, metadata, cache and
    # bundle assembly are the production implementations under isolated HOME.
    files = {
        "SKILL.md": b"---\nname: root-skill\ndescription: A root skill.\n---\nRead scripts/run.py.\n",
        "scripts/run.py": b"print('fixture')\n",
    }
    requests = []

    def get(url, **kwargs):
        requests.append((url, kwargs.get("params")))
        request = httpx.Request("GET", url)
        if url == "https://skills.sh/api/search":
            return httpx.Response(200, json={"skills": [{"id": "owner/root-skill/root-skill"}]}, request=request)
        if url == "https://api.github.com/repos/owner/root-skill":
            return httpx.Response(200, json={"default_branch": "main"}, request=request)
        if url.endswith("/git/trees/main"):
            return httpx.Response(200, json={"sha": "fixture-revision", "tree": [
                {"path": p, "type": "blob", "mode": "100644"} for p in files
            ]}, request=request)
        prefix = "https://api.github.com/repos/owner/root-skill/contents/"
        if url.startswith(prefix):
            path = url[len(prefix):]
            if path in files:
                return httpx.Response(200, content=files[path], request=request)
            if not path:
                return httpx.Response(200, json=[
                    {"type": "file", "name": "SKILL.md"},
                    {"type": "dir", "name": "scripts"},
                ], request=request)
        return httpx.Response(404, request=request)

    monkeypatch.setattr("httpx.get", get)
    auth = GitHubAuth()
    monkeypatch.setattr(auth, "get_headers", lambda: {})
    return SkillsShSource(auth), files, requests


@pytest.mark.parametrize("operation", ["inspect", "fetch"])
def test_search_result_with_root_skill_resolves(source, operation):
    src, files, requests = source
    identifier = src.search("root skill")[0].identifier
    result = getattr(src, operation)(identifier)
    assert result is not None, "search advertised a root skill the resolver cannot load"
    assert result.identifier == identifier
    assert result.name == "root-skill"
    assert result.trust_level == "community"
    if operation == "fetch":
        assert result.files["scripts/run.py"] == files["scripts/run.py"]
        assert result.metadata["source_revision"] == "fixture-revision"
        pinned = [(url, params) for url, params in requests if params == {"ref": "fixture-revision"}]
        assert any(url.endswith("/contents/SKILL.md") for url, _ in pinned)
        assert any(url.endswith("/contents/scripts/run.py") for url, _ in pinned)


@pytest.mark.parametrize("operation", ["inspect", "fetch"])
def test_root_frontmatter_name_can_differ_from_repo_name(source, operation):
    src, files, _ = source
    files["SKILL.md"] = files["SKILL.md"].replace(b"name: root-skill", b"name: actual-skill")
    result = getattr(src, operation)("skills-sh/owner/root-skill/actual-skill")
    assert result is not None
    assert result.name == "actual-skill"


@pytest.mark.parametrize("operation", ["inspect", "fetch"])
def test_named_nested_skill_takes_precedence_over_root(source, operation):
    src, files, _ = source
    files["skills/root-skill/SKILL.md"] = b"---\nname: nested-winner\n---\nNested.\n"
    result = getattr(src, operation)("skills-sh/owner/root-skill/root-skill")
    assert result is not None
    if operation == "inspect":
        assert result.path == "skills/root-skill"
    else:
        assert "Nested." in result.files["SKILL.md"]
        assert "scripts/run.py" not in result.files


@pytest.mark.parametrize("operation", ["inspect", "fetch"])
def test_unrelated_slug_does_not_adopt_repo_root(source, operation):
    src, _, _ = source
    assert getattr(src, operation)("skills-sh/owner/root-skill/unrelated") is None
