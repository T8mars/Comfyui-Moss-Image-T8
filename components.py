"""Immutable file descriptors. No model is retained in a ComfyUI node cache."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

TOKENIZER_FILES = frozenset({"tokenizer.json", "tokenizer_config.json", "chat_template.jinja"})


def check_tokenizer_directory(directory):
    directory = Path(directory)
    if (not directory.is_dir() or {entry.name for entry in directory.iterdir()} != TOKENIZER_FILES
            or any(not (directory / name).is_file() for name in TOKENIZER_FILES)):
        raise ValueError(f"Expected only the three verified official tokenizer files in {directory}. "
                         "Restore the matching tokenizer directory; extra inputs are not accepted.")


@dataclass(frozen=True)
class Component:
    path: str
    kind: str
    config_json: str
    scheduler_json: str = "{}"
    device: str = "auto"
    precision: str = "bf16"
    package_fingerprint: tuple | None = None

    @property
    def config(self):
        return json.loads(self.config_json)

    @property
    def assets(self):
        return Path(self.path + ".assets")


def interrupted():
    try:
        from comfy.model_management import throw_exception_if_processing_interrupted
    except ImportError:
        return
    throw_exception_if_processing_interrupted()


def file_sha256(path):
    interrupted()
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            interrupted()
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=32)
def verified_weight_sha256(path, size, modified_ns, changed_ns, device, inode):
    # File identity also catches atomic replacements that preserve size/mtime.
    return file_sha256(path)


def component_path(path):
    # HF snapshots keep sidecars beside a logical weight symlink, not its blob.
    logical = Path(path).absolute()
    if os.path.lexists(str(logical) + ".assets"):
        return logical
    return logical.resolve()


def inspect_component(path, kind, device="auto", precision="bf16"):
    interrupted()
    from safetensors import safe_open
    path = component_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"MOSS-Image component missing: {path}. Run scripts/prepare_models.py --models-dir <ComfyUI/models>.")
    signature = fingerprint(path)
    with safe_open(path, framework="pt", device="cpu") as stream:
        metadata = stream.metadata() or {}
    if metadata.get("moss_image.format") != "1" or metadata.get("moss_image.component") != kind:
        raise ValueError(f"Expected independently exported MOSS-Image {kind}: {path}. Run scripts/prepare_models.py first.")
    config = json.loads(metadata["moss_image.config"])
    if kind == "transformer":
        if config.get("_class_name") not in ("MOSSImageTransformer", "MOSImageTransformer"):
            raise ValueError("Unsupported MOSS-Image DiT class")
        expected = {"latent_channels": 128, "text_dim": 4096, "freq_dim": 256, "rope_base": 10000.0}
        if any(config.get(key) != value for key, value in expected.items()):
            raise ValueError("Incompatible MOSS-Image DiT configuration")
    elif kind == "text_encoder":
        if config.get("model_type") != "qwen3_5" or config.get("text_config", {}).get("hidden_size") != 4096:
            raise ValueError("MOSS-Image requires the official Qwen3.5-9B text encoder")
    elif kind == "vae":
        if config.get("_class_name") != "AutoencoderKLFlux2" or config.get("latent_channels") != 32:
            raise ValueError("MOSS-Image requires the official FLUX.2 VAE")
    else:
        raise ValueError(f"Unknown component: {kind}")
    assets = Path(str(path) + ".assets")
    manifest = json.loads((assets / "manifest.json").read_text(encoding="utf-8"))
    if (not manifest.get("ready") or manifest.get("component") != kind
            or manifest.get("size") != path.stat().st_size
            or manifest.get("config_sha256") != hashlib.sha256(metadata["moss_image.config"].encode()).hexdigest()):
        raise ValueError(f"Incomplete or mismatched component package: {path}")
    if kind == "text_encoder":
        names = manifest.get("tokenizer_files", {})
        if set(names) != TOKENIZER_FILES:
            raise ValueError("Incomplete MOSS-Image tokenizer manifest")
        check_tokenizer_directory(assets / "tokenizer")
    stat = path.stat()
    if verified_weight_sha256(str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns,
                              stat.st_dev, stat.st_ino) != manifest.get("sha256"):
        raise ValueError(f"MOSS-Image weight SHA256 mismatch: {path}. Re-run the preparation tool or restore the verified file.")
    if kind == "text_encoder":
        for name, digest in names.items():
            token_file = assets / "tokenizer" / name
            if not token_file.is_file() or file_sha256(token_file) != digest:
                raise ValueError(f"Missing or modified official tokenizer: {token_file}")
    if fingerprint(path) != signature:
        raise ValueError(f"MOSS-Image package changed during verification: {path}. Re-run the loader.")
    return Component(str(path), kind, metadata["moss_image.config"],
                     metadata.get("moss_image.scheduler", "{}"), device, precision, signature)


def ensure_unchanged(component):
    if component.kind == "text_encoder":
        check_tokenizer_directory(component.assets / "tokenizer")
    if component.package_fingerprint is not None and fingerprint(component.path) != component.package_fingerprint:
        raise ValueError(f"MOSS-Image package changed since verification: {component.path}. Re-run the workflow to reload it.")


def fingerprint(path):
    path = component_path(path)
    files = [path, Path(str(path) + ".assets") / "manifest.json"]
    tokenizer = Path(str(path) + ".assets") / "tokenizer"
    if tokenizer.is_dir():
        files.append(tokenizer)
        files.extend(sorted(tokenizer.iterdir()))
    result = []
    for file in files:
        if file.is_file() or file.is_dir():
            stat = file.stat()
            result.append((str(file), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_dev, stat.st_ino))
    return tuple(result)
