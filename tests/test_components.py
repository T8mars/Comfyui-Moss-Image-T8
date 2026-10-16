import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from moss_plugin.components import inspect_component
from moss_plugin.scripts.prepare_models import sha256


def test_loader_checks_complete_file_digest(tmp_path):
    config = json.dumps({"_class_name": "MOSSImageTransformer", "latent_channels": 128,
                         "text_dim": 4096, "freq_dim": 256, "rope_base": 10000.0})
    path = tmp_path / "dit.safetensors"
    save_file({"a": torch.ones(4)}, path, metadata={"moss_image.format": "1", "moss_image.component": "transformer",
                                                   "moss_image.config": config})
    assets = Path(str(path) + ".assets")
    assets.mkdir()
    manifest = {"ready": True, "component": "transformer", "size": path.stat().st_size,
                "sha256": sha256(path), "config_sha256": hashlib.sha256(config.encode()).hexdigest()}
    (assets / "manifest.json").write_text(json.dumps(manifest))
    assert inspect_component(path, "transformer").kind == "transformer"
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match="SHA256"):
        inspect_component(path, "transformer")
    with pytest.raises(ValueError, match="Expected"):
        inspect_component(path, "vae")


def test_vendor_is_unmodified():
    path = Path(__file__).resolve().parents[1] / "vendor/pipeline_moss_image.py"
    assert sha256(path) == "1dac1aca9feb1bfdfc727a401e4b67d61a505d3190706d1b7d2d784f88678b31"
