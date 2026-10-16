"""Export and recovery may only publish bytes bound to their verified sources."""
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from moss_plugin.components import inspect_component
from moss_plugin.scripts import prepare_models as preparation


def record_sources(source, kind, revision="pinned"):
    files = [p for p in (source / kind).iterdir() if p.is_file() and not p.name.startswith("moss-export")]
    if kind == "text_encoder":
        files.extend((source / "tokenizer").iterdir())
    record = {"repo": "test", "revision": revision, "files": {
        p.relative_to(source).as_posix(): {"size": p.stat().st_size, "sha256": preparation.sha256(p)} for p in files}}
    (source / f"verified-{kind}.json").write_text(json.dumps(record))


def sources(tmp_path, kind="vae"):
    source = tmp_path / "source"
    directory = source / kind
    directory.mkdir(parents=True)
    config = ({"_class_name": "AutoencoderKLFlux2", "latent_channels": 32} if kind == "vae" else
              {"model_type": "qwen3_5", "text_config": {"hidden_size": 4096}})
    (directory / "config.json").write_text(json.dumps(config))
    weight = directory / ("diffusion_pytorch_model.safetensors" if kind == "vae" else "model.safetensors")
    save_file({"a": torch.ones(4)}, weight)
    if kind == "text_encoder":
        (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": weight.name}}))
        tokens = source / "tokenizer"
        tokens.mkdir()
        for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
            (tokens / name).write_text("original " + name)
    record_sources(source, kind)
    return source, weight


def test_changed_source_config_cannot_damage_a_ready_package(tmp_path):
    source, _ = sources(tmp_path)
    target = tmp_path / "vae.safetensors"
    preparation.export_component(source, "vae", target)
    assets = Path(str(target) + ".assets")
    before = (assets / "manifest.json").read_bytes()
    assert inspect_component(target, "vae")
    (source / "vae/config.json").write_text('{"_class_name":"AutoencoderKLFlux2","latent_channels":64}')
    with pytest.raises(ValueError, match="source"):
        preparation.export_component(source, "vae", target)
    assert (assets / "manifest.json").read_bytes() == before
    assert inspect_component(target, "vae")


def test_modified_source_weight_cannot_be_published_under_old_provenance(tmp_path):
    source, weight = sources(tmp_path)
    save_file({"a": torch.zeros(4)}, weight)
    target = tmp_path / "vae.safetensors"
    with pytest.raises(ValueError, match="source"):
        preparation.export_component(source, "vae", target)
    assert not target.exists()


def test_incompatible_revision_cannot_overwrite_a_valid_tokenizer(tmp_path):
    source, _ = sources(tmp_path, "text_encoder")
    target = tmp_path / "te.safetensors"
    preparation.export_component(source, "text_encoder", target)
    assets = Path(str(target) + ".assets")
    before = {p.relative_to(assets): p.read_bytes() for p in assets.rglob("*") if p.is_file()}
    (source / "tokenizer/tokenizer.json").write_text("different revision tokenizer")
    record_sources(source, "text_encoder", revision="different")
    with pytest.raises(FileExistsError):
        preparation.export_component(source, "text_encoder", target)
    assert {p.relative_to(assets): p.read_bytes() for p in assets.rglob("*") if p.is_file()} == before
    assert inspect_component(target, "text_encoder")


def test_source_change_after_verification_cannot_publish_a_model(tmp_path, monkeypatch):
    source, weight = sources(tmp_path)
    target = tmp_path / "vae.safetensors"
    original = preparation.merge_component
    def changed_source(*args, **kwargs):
        save_file({"a": torch.zeros(4)}, weight)
        return original(*args, **kwargs)
    monkeypatch.setattr(preparation, "merge_component", changed_source)
    with pytest.raises(ValueError, match="source changed"):
        preparation.export_component(source, "vae", target)
    assert not target.exists()
    assert not json.loads(Path(str(target) + ".assets/manifest.json").read_text())["ready"]


def test_pending_recovery_uses_the_actual_embedded_config_bytes(tmp_path):
    source, _ = sources(tmp_path)
    target = tmp_path / "vae.safetensors"
    preparation.export_component(source, "vae", target)
    metadata = preparation.component_metadata(target)
    metadata["moss_image.config"] = json.dumps(json.loads(metadata["moss_image.config"]), separators=(",", ":"))
    replacement = tmp_path / "replacement.safetensors"
    values = {key: value.clone() for key, value in load_file(target).items()}
    save_file(values, replacement, metadata=metadata)
    replacement.replace(target)
    manifest_path = Path(str(target) + ".assets/manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest.update(ready=False, pending_sha256=preparation.sha256(target), size=target.stat().st_size)
    manifest.pop("sha256")
    manifest.pop("config_sha256")
    manifest_path.write_text(json.dumps(manifest))
    preparation.export_component(source, "vae", target)
    assert inspect_component(target, "vae")


@pytest.mark.parametrize("pending", [False, True])
def test_te_recovery_keeps_the_weight_digest_separate_from_tokenizer_digests(tmp_path, pending):
    source, _ = sources(tmp_path, "text_encoder")
    target = tmp_path / "te.safetensors"
    preparation.export_component(source, "text_encoder", target)
    digest = preparation.sha256(target)
    manifest_path = Path(str(target) + ".assets/manifest.json")
    if pending:
        manifest = json.loads(manifest_path.read_text())
        manifest.update(ready=False, pending_sha256=manifest.pop("sha256"))
        manifest.pop("config_sha256")
        manifest_path.write_text(json.dumps(manifest))
    preparation.export_component(source, "text_encoder", target)
    assert json.loads(manifest_path.read_text())["sha256"] == digest
    assert inspect_component(target, "text_encoder")
