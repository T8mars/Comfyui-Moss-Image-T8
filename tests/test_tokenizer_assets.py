"""Only manifest-bound tokenizer inputs may reach the official encoder."""
from dataclasses import replace
from pathlib import Path
import json

import pytest

from moss_plugin import components, runtime
from moss_plugin.scripts import prepare_models as preparation
from moss_plugin.tests.test_component_binding import package
from moss_plugin.tests.test_export_integrity import sources


EXTRA_INPUTS = ("additional_chat_templates/default.jinja", "added_tokens.json", "special_tokens_map.json")


def extra_input(directory, name):
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("unverified input")
    return path


def snapshot(directory):
    return {p.relative_to(directory): p.read_bytes() for p in directory.rglob("*") if p.is_file()}


@pytest.mark.parametrize("extra", EXTRA_INPUTS)
def test_inspection_rejects_unlisted_inputs_before_hashing_weights(tmp_path, monkeypatch, extra):
    path, assets, _ = package(tmp_path, "text_encoder")
    extra_input(assets / "tokenizer", extra)
    monkeypatch.setattr(components, "verified_weight_sha256", lambda *_: pytest.fail("Invalid tokenizer hashed weights"))
    with pytest.raises(ValueError, match="tokenizer"):
        components.inspect_component(path, "text_encoder")


@pytest.mark.parametrize("bound", [True, False])
def test_extra_template_is_rejected_before_model_allocation(tmp_path, monkeypatch, bound):
    path, assets, _ = package(tmp_path, "text_encoder")
    descriptor = components.inspect_component(path, "text_encoder")
    if not bound:
        descriptor = replace(descriptor, package_fingerprint=None)
    extra_input(assets / "tokenizer", EXTRA_INPUTS[0])
    monkeypatch.setattr(runtime, "materialize", lambda *_: pytest.fail("Unverified tokenizer allocated a model"))
    monkeypatch.setattr(runtime, "clean_memory", lambda: None)
    with pytest.raises(ValueError, match="tokenizer"):
        with runtime.loaded(descriptor, "text_encoder"):
            pytest.fail("Unverified tokenizer entered encoding")


def test_extra_directory_invalidates_loader_fingerprint(tmp_path):
    path, assets, _ = package(tmp_path, "text_encoder")
    before = components.fingerprint(path)
    extra_input(assets / "tokenizer", EXTRA_INPUTS[0])
    assert components.fingerprint(path) != before


def test_required_tokenizer_files_can_be_hf_snapshot_symlinks(tmp_path):
    path, assets, _ = package(tmp_path, "text_encoder")
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    for name in components.TOKENIZER_FILES:
        logical = assets / "tokenizer" / name
        actual = blobs / name
        logical.replace(actual)
        try:
            logical.symlink_to(actual)
        except OSError as exc:
            if getattr(exc, "winerror", None) == 1314:
                pytest.skip("File symlinks need Windows developer mode or symlink privileges")
            raise
    descriptor = components.inspect_component(path, "text_encoder")
    components.ensure_unchanged(descriptor)


def test_extra_template_created_during_stage_cannot_return_success(tmp_path, monkeypatch):
    path, assets, _ = package(tmp_path, "text_encoder")
    descriptor = components.inspect_component(path, "text_encoder")
    monkeypatch.setattr(runtime, "materialize", lambda *_: object())
    monkeypatch.setattr(runtime, "clean_memory", lambda: None)
    with pytest.raises(ValueError, match="tokenizer"):
        with runtime.loaded(descriptor, "text_encoder"):
            extra_input(assets / "tokenizer", EXTRA_INPUTS[0])


@pytest.mark.parametrize("ready", [False, True])
@pytest.mark.parametrize("extra", EXTRA_INPUTS)
def test_export_preserves_and_rejects_unlisted_tokenizer_entries(tmp_path, ready, extra):
    source, _ = sources(tmp_path, "text_encoder")
    target = tmp_path / "te.safetensors"
    if ready:
        preparation.export_component(source, "text_encoder", target)
    assets = Path(str(target) + ".assets")
    extra_path = extra_input(assets / "tokenizer", extra)
    before = snapshot(assets)
    weights = target.read_bytes() if target.exists() else None
    with pytest.raises(ValueError, match="tokenizer"):
        preparation.export_component(source, "text_encoder", target)
    assert snapshot(assets) == before and extra_path.read_text() == "unverified input"
    assert (target.read_bytes() if target.exists() else None) == weights


@pytest.mark.parametrize("pending", [False, True])
def test_recovery_removes_only_the_three_owned_stale_partials(tmp_path, pending):
    source, _ = sources(tmp_path, "text_encoder")
    target = tmp_path / "te.safetensors"
    preparation.export_component(source, "text_encoder", target)
    assets = Path(str(target) + ".assets")
    manifest_path = assets / "manifest.json"
    if pending:
        manifest = json.loads(manifest_path.read_text())
        manifest.update(ready=False, pending_sha256=manifest.pop("sha256"))
        manifest_path.write_text(json.dumps(manifest))
    for name in preparation.TOKENIZER_FILES:
        (assets / "tokenizer" / (name + ".partial")).write_text("stale copy")
    preparation.export_component(source, "text_encoder", target)
    assert not list((assets / "tokenizer").glob("*.partial"))
    assert components.inspect_component(target, "text_encoder")


def test_partial_directory_is_preserved_and_rejected(tmp_path):
    source, _ = sources(tmp_path, "text_encoder")
    target = tmp_path / "te.safetensors"
    preparation.export_component(source, "text_encoder", target)
    assets = Path(str(target) + ".assets")
    entry = extra_input(assets / "tokenizer", "tokenizer.json.partial/user.txt")
    before = snapshot(assets)
    with pytest.raises(ValueError, match="tokenizer"):
        preparation.export_component(source, "text_encoder", target)
    assert snapshot(assets) == before and entry.exists()


def test_partial_hardlink_cannot_overwrite_another_file(tmp_path):
    source, _ = sources(tmp_path, "text_encoder")
    target = tmp_path / "te.safetensors"
    preparation.export_component(source, "text_encoder", target)
    tokens = Path(str(target) + ".assets/tokenizer")
    (tokens / "tokenizer.json").write_text("damaged")
    external = tmp_path / "external.txt"
    external.write_bytes(b"preserve this file")
    (tokens / "tokenizer.json.partial").hardlink_to(external)
    preparation.export_component(source, "text_encoder", target)
    assert external.read_bytes() == b"preserve this file"
    assert components.inspect_component(target, "text_encoder")


def test_partial_symlink_is_preserved_and_rejected(tmp_path):
    source, _ = sources(tmp_path, "text_encoder")
    target = tmp_path / "te.safetensors"
    preparation.export_component(source, "text_encoder", target)
    tokens = Path(str(target) + ".assets/tokenizer")
    external = tmp_path / "external.txt"
    external.write_bytes(b"preserve this file")
    link = tokens / "tokenizer.json.partial"
    try:
        link.symlink_to(external)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip("File symlinks need Windows developer mode or symlink privileges")
        raise
    with pytest.raises(ValueError, match="tokenizer"):
        preparation.export_component(source, "text_encoder", target)
    assert external.read_bytes() == b"preserve this file" and link.is_symlink()


def test_extra_input_during_merge_cannot_publish_a_ready_package(tmp_path, monkeypatch):
    source, _ = sources(tmp_path, "text_encoder")
    target = tmp_path / "te.safetensors"
    original = preparation.merge_component
    def changed_target(*args, **kwargs):
        result = original(*args, **kwargs)
        extra_input(Path(str(target) + ".assets/tokenizer"), EXTRA_INPUTS[0])
        return result
    monkeypatch.setattr(preparation, "merge_component", changed_target)
    with pytest.raises(ValueError, match="tokenizer"):
        preparation.export_component(source, "text_encoder", target)
    assert not json.loads(Path(str(target) + ".assets/manifest.json").read_text())["ready"]


def test_extra_input_during_recovery_does_not_rewrite_the_manifest(tmp_path, monkeypatch):
    source, _ = sources(tmp_path, "text_encoder")
    target = tmp_path / "te.safetensors"
    preparation.export_component(source, "text_encoder", target)
    manifest = Path(str(target) + ".assets/manifest.json")
    before = manifest.read_bytes()
    original = preparation.ensure_sources_unchanged
    def changed_target(*args):
        original(*args)
        extra_input(Path(str(target) + ".assets/tokenizer"), EXTRA_INPUTS[0])
    monkeypatch.setattr(preparation, "ensure_sources_unchanged", changed_target)
    with pytest.raises(ValueError, match="tokenizer"):
        preparation.export_component(source, "text_encoder", target)
    assert manifest.read_bytes() == before
