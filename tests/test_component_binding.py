"""A queued descriptor must never silently consume a changed component package."""
import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from moss_plugin import components, runtime


def package(tmp_path, kind="transformer"):
    config = ({"_class_name": "MOSSImageTransformer", "latent_channels": 128, "text_dim": 4096,
               "freq_dim": 256, "rope_base": 10000.0} if kind == "transformer" else
              {"model_type": "qwen3_5", "text_config": {"hidden_size": 4096}})
    config_json = json.dumps(config)
    path = tmp_path / "component.safetensors"
    metadata = {"moss_image.format": "1", "moss_image.component": kind, "moss_image.config": config_json}
    save_file({"a": torch.ones(4)}, path, metadata=metadata)
    assets = Path(str(path) + ".assets")
    assets.mkdir()
    manifest = {"ready": True, "component": kind, "size": path.stat().st_size,
                "sha256": components.file_sha256(path),
                "config_sha256": hashlib.sha256(config_json.encode()).hexdigest()}
    if kind == "text_encoder":
        (assets / "tokenizer").mkdir()
        names = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
        for name in names:
            (assets / "tokenizer" / name).write_text("original " + name)
        manifest["tokenizer_files"] = {name: components.file_sha256(assets / "tokenizer" / name) for name in names}
    (assets / "manifest.json").write_text(json.dumps(manifest))
    return path, assets, metadata


def replace_weights(path, metadata):
    replacement = path.with_name("replacement.safetensors")
    save_file({"a": torch.zeros(4)}, replacement, metadata=metadata)
    replacement.replace(path)


@pytest.mark.parametrize("change", ["weights", "manifest", "tokenizer"])
def test_changed_package_is_rejected_before_allocating_a_model(tmp_path, monkeypatch, change):
    kind = "text_encoder" if change == "tokenizer" else "transformer"
    path, assets, metadata = package(tmp_path, kind)
    component = components.inspect_component(path, kind, device="cpu")
    if change == "weights":
        replace_weights(path, metadata)
    elif change == "manifest":
        (assets / "manifest.json").write_text('{"ready":false}')
    else:
        (assets / "tokenizer/tokenizer.json").write_text("different tokenizer")
    monkeypatch.setattr(runtime, "materialize", lambda *_: pytest.fail("Changed package allocated a model"))
    monkeypatch.setattr(runtime, "clean_memory", lambda: None)
    with pytest.raises(ValueError, match="changed"):
        with runtime.loaded(component, kind):
            pytest.fail("Changed package entered a pipeline stage")


def test_package_change_during_materialization_never_reaches_the_stage(tmp_path, monkeypatch):
    path, _, metadata = package(tmp_path)
    component = components.inspect_component(path, "transformer", device="cpu")
    def materialize(*_):
        replace_weights(path, metadata)
        return object()
    monkeypatch.setattr(runtime, "materialize", materialize)
    monkeypatch.setattr(runtime, "clean_memory", lambda: None)
    with pytest.raises(ValueError, match="changed"):
        with runtime.loaded(component, "transformer"):
            pytest.fail("Changed package entered a pipeline stage")


def test_package_change_during_stage_cannot_return_success(tmp_path, monkeypatch):
    path, assets, _ = package(tmp_path)
    component = components.inspect_component(path, "transformer", device="cpu")
    sentinel = object()
    monkeypatch.setattr(runtime, "materialize", lambda *_: sentinel)
    monkeypatch.setattr(runtime, "clean_memory", lambda: None)
    with pytest.raises(ValueError, match="changed"):
        with runtime.loaded(component, "transformer") as model:
            assert model is sentinel
            (assets / "manifest.json").write_text('{"ready":false}')


def test_inspection_cannot_bind_a_package_changed_during_verification(tmp_path, monkeypatch):
    path, assets, _ = package(tmp_path)
    original = components.verified_weight_sha256
    def verify(*args):
        digest = original(*args)
        (assets / "manifest.json").write_text('{"ready":false}')
        return digest
    monkeypatch.setattr(components, "verified_weight_sha256", verify)
    with pytest.raises(ValueError, match="changed"):
        components.inspect_component(path, "transformer")


def file_symlink(alias, target):
    try:
        alias.symlink_to(target)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip("File symlinks need Windows developer mode or symlink privileges")
        raise


def test_hf_snapshot_weight_symlink_keeps_logical_sidecars(tmp_path):
    source = tmp_path / "blobs"
    source.mkdir()
    path, assets, _ = package(source)
    alias = tmp_path / "snapshot.safetensors"
    file_symlink(alias, path)
    logical_assets = Path(str(alias) + ".assets")
    assert logical_assets.resolve().is_relative_to(tmp_path.resolve())
    assets.rename(logical_assets)
    component = components.inspect_component(alias, "transformer")
    assert Path(component.path) == alias.absolute()
    assert component.assets == logical_assets


def test_weight_symlink_target_sidecar_changes_invalidate_fingerprint(tmp_path):
    source = tmp_path / "target"
    source.mkdir()
    path, assets, _ = package(source, "text_encoder")
    alias = tmp_path / "alias.safetensors"
    file_symlink(alias, path)
    component = components.inspect_component(alias, "text_encoder")
    before = components.fingerprint(alias)
    (assets / "tokenizer/tokenizer.json").write_text("changed")
    assert components.fingerprint(alias) != before
    with pytest.raises(ValueError, match="changed"):
        components.ensure_unchanged(component)


def test_invalid_logical_sidecar_cannot_fall_back_to_a_valid_target_package(tmp_path):
    source = tmp_path / "target"
    source.mkdir()
    path, _, _ = package(source)
    alias = tmp_path / "alias.safetensors"
    file_symlink(alias, path)
    logical_assets = Path(str(alias) + ".assets")
    logical_assets.mkdir()
    (logical_assets / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="Incomplete"):
        components.inspect_component(alias, "transformer")
