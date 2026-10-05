"""Companion assets never become selectable models or router launch sections."""

import configparser
import struct

import pytest

from hermes_cli.local_runtime import bootstrap, hf_browse, presets
from hermes_cli.local_runtime.estimator import HardwareBudget


COMPANIONS = [
    "mmproj-vision-BF16.gguf", "MMPROJ-audio-Q8_0.gguf",
    "vision-mmproj-F16.gguf", "projector-vision.gguf",
    "dspark-draft.gguf", "model-draft-Q4.gguf",
]


def _write_header(path):
    # Real minimal GGUF metadata consumed by the production reader/planner.
    def string(value):
        data = value.encode()
        return struct.pack("<Q", len(data)) + data

    metadata = {"general.architecture": "llama", "llama.block_count": 2,
                "llama.context_length": 4096, "llama.embedding_length": 64,
                "llama.attention.head_count": 2, "llama.attention.head_count_kv": 2}
    data = b"GGUF" + struct.pack("<IQQ", 3, 0, len(metadata))
    for key, value in metadata.items():
        data += string(key)
        data += (struct.pack("<I", 8) + string(value) if isinstance(value, str)
                 else struct.pack("<II", 4, value))
    path.write_bytes(data)


@pytest.mark.parametrize("require_complete", [True, False])
def test_staging_excludes_companions_without_changing_split_contract(tmp_path, require_complete):
    model = tmp_path / "chat.gguf"
    mtp_model = tmp_path / "chat-MTP.gguf"
    complete = tmp_path / "chat-split-00001-of-00002.gguf"
    incomplete = tmp_path / "pending-00001-of-00002.gguf"
    names = [model.name, mtp_model.name, complete.name, "chat-split-00002-of-00002.gguf", incomplete.name,
             *COMPANIONS, "mmproj-split-00001-of-00002.gguf", "mmproj-split-00002-of-00002.gguf"]
    for name in names:
        (tmp_path / name).touch()
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "hidden.gguf").touch()
    expected = {model, mtp_model, complete} | ({incomplete} if not require_complete else set())
    assert set(bootstrap.staged_in(tmp_path, require_complete=require_complete)) == expected
    assert all((tmp_path / name).exists() for name in names)


def test_picker_ids_and_real_generated_presets_agree(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap, "models_dir", lambda: tmp_path)
    for name in ["chat.gguf", *COMPANIONS]:
        _write_header(tmp_path / name)
    budget = HardwareBudget(usable_vram_bytes=8 << 30, total_device_bytes=8 << 30, ram_available_bytes=16 << 30, uma=False)
    output = tmp_path / "presets.ini"
    entries = presets.generate_presets(tmp_path, budget, output, live=budget)
    assert bootstrap.staged_model_ids() == ["chat"]
    assert [entry.model_id for entry in entries] == ["chat"]
    ini = configparser.ConfigParser()
    ini.read(output)
    assert ini.sections() == ["chat"]
    assert ini["chat"]["model"] == str(tmp_path / "chat.gguf")


def test_hf_browse_and_staged_discovery_use_same_companion_boundary(tmp_path, monkeypatch):
    names = ["chat.gguf", *COMPANIONS]
    monkeypatch.setattr(hf_browse, "_get_json", lambda url: [
        {"path": "nested/" + name, "size": 100} for name in names])
    for name in names:
        (tmp_path / name).touch()
    remote = [g.paths[0].rsplit("/", 1)[-1] for g in hf_browse.repo_files("fixture/models")]
    assert remote == ["chat.gguf"]
    assert [p.name for p in bootstrap.staged_in(tmp_path)] == remote


def test_empty_directory_has_no_models_or_presets(tmp_path):
    budget = HardwareBudget(usable_vram_bytes=8 << 30, total_device_bytes=8 << 30, ram_available_bytes=16 << 30, uma=False)
    assert bootstrap.staged_in(tmp_path) == []
    assert presets.plan_presets(tmp_path, budget, live=budget) == []
