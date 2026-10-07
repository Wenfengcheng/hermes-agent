"""Regression for #134540: resolve Windows CA references at the Git boundary."""

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from hermes_cli import update_cmd
from hermes_cli.env_loader import load_hermes_dotenv


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("reference", ["%USERPROFILE%", "~"])
def test_dotenv_ca_path_reaches_git_children(monkeypatch, tmp_path, reference):
    home = tmp_path / "profile"
    home.mkdir()
    bundle = tmp_path / "certs" / "corporate bundle.pem"
    bundle.parent.mkdir()
    bundle.write_text("local path fixture", encoding="utf-8")
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("GIT_SSL_CAINFO", raising=False)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    source = f"{reference}/certs/corporate bundle.pem"
    (home / ".env").write_text(f"SSL_CERT_FILE={source}\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(tmp_path / "repo")], check=True)
    monkeypatch.setattr(update_cmd, "_m", lambda: SimpleNamespace(PROJECT_ROOT=tmp_path / "repo"))

    load_hermes_dotenv(hermes_home=home, load_external_secrets=False)
    update_cmd._map_ssl_cert_file_for_git(["git"])

    mapped = update_cmd._no_prompt_git_kwargs()["env"]["GIT_SSL_CAINFO"]
    assert os.path.normpath(mapped) == str(bundle)
    assert os.path.isfile(mapped)
    child = subprocess.run(
        [sys.executable, "-c", "import os; print(os.environ['GIT_SSL_CAINFO'])"],
        env=update_cmd._no_prompt_git_kwargs()["env"], capture_output=True,
        text=True, check=True,
    )
    assert os.path.normpath(child.stdout.strip()) == str(bundle)
    # The dotenv source and unrelated consumers remain byte-exact: this is a Git boundary fix.
    assert os.environ["SSL_CERT_FILE"] == source
    assert (home / ".env").read_text(encoding="utf-8") == f"SSL_CERT_FILE={source}\n"


@pytest.mark.parametrize("pin", ["environment", "config"])
def test_explicit_git_trust_still_wins(monkeypatch, tmp_path, pin):
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("SSL_CERT_FILE", "%USERPROFILE%/certs/corporate.pem")
    monkeypatch.delenv("GIT_SSL_CAINFO", raising=False)
    monkeypatch.setattr(update_cmd, "_m", lambda: SimpleNamespace(PROJECT_ROOT=repo))
    if pin == "environment":
        monkeypatch.setenv("GIT_SSL_CAINFO", "explicit.pem")
    else:
        subprocess.run(["git", "-C", str(repo), "config", "http.sslCAInfo", "explicit.pem"], check=True)

    update_cmd._map_ssl_cert_file_for_git(["git"])

    assert os.environ.get("GIT_SSL_CAINFO") == ("explicit.pem" if pin == "environment" else None)


@pytest.mark.parametrize("case", ["empty", "unknown", "literal", "query_error"])
def test_mapping_preserves_existing_boundary_behavior(monkeypatch, tmp_path, case):
    source = "%SCOUT_UNDEFINED_CA%/missing.pem"
    monkeypatch.delenv("SCOUT_UNDEFINED_CA", raising=False)
    if case == "empty":
        source = ""
    elif case == "literal":
        source = str(tmp_path / "%SCOUT_DEFINED_CA%.pem")
        (tmp_path / "%SCOUT_DEFINED_CA%.pem").write_text("fixture", encoding="utf-8")
        monkeypatch.setenv("SCOUT_DEFINED_CA", "different")
    monkeypatch.setenv("SSL_CERT_FILE", source)
    monkeypatch.delenv("GIT_SSL_CAINFO", raising=False)

    def query(*args, **kwargs):
        if case == "empty":
            pytest.fail("empty CA must not query Git")
        if case == "query_error":
            raise OSError("Git unavailable")
        return subprocess.CompletedProcess(args, 1, "", "")

    monkeypatch.setattr(update_cmd, "_git_run", query)
    if case == "query_error":
        with pytest.raises(OSError, match="Git unavailable"):
            update_cmd._map_ssl_cert_file_for_git(["git"])
        assert "GIT_SSL_CAINFO" not in os.environ
    else:
        update_cmd._map_ssl_cert_file_for_git(["git"])
        assert os.environ.get("GIT_SSL_CAINFO") == (source or None)
