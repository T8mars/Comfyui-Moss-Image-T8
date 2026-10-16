"""Comfy Stop must be observed inside hashing and before model forward work."""
import hashlib
import json
import sys
from types import ModuleType

import pytest
import torch

from moss_plugin import components, runtime


class StopRequested(BaseException):
    pass


@pytest.fixture
def stop_flag(monkeypatch):
    flag = [False]
    management = ModuleType("comfy.model_management")
    def check():
        if flag[0]:
            flag[0] = False
            raise StopRequested("Comfy Stop")
    management.throw_exception_if_processing_interrupted = check
    package = ModuleType("comfy")
    package.model_management = management
    monkeypatch.setitem(sys.modules, "comfy", package)
    monkeypatch.setitem(sys.modules, "comfy.model_management", management)
    return flag


@pytest.mark.parametrize("cached", [False, True])
def test_stop_during_hash_aborts_and_cannot_cache_a_partial_digest(tmp_path, monkeypatch, stop_flag, cached):
    path = tmp_path / "weight"
    content = b"x" * (3 * 1024 * 1024 + 7)
    path.write_bytes(content)
    original = hashlib.sha256
    blocks = []
    class InterruptingDigest:
        def __init__(self):
            self.inner = original()
        def update(self, block):
            blocks.append(len(block))
            self.inner.update(block)
            stop_flag[0] = True
        def hexdigest(self):
            return self.inner.hexdigest()
    monkeypatch.setattr(components.hashlib, "sha256", InterruptingDigest)
    stat = path.stat()
    arguments = (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_dev, stat.st_ino)
    digest = (lambda: components.verified_weight_sha256(*arguments)) if cached else lambda: components.file_sha256(path)
    with pytest.raises(StopRequested):
        digest()
    assert blocks == [1024 * 1024]
    monkeypatch.setattr(components.hashlib, "sha256", original)
    assert digest() == original(content).hexdigest()


def test_stop_during_materialization_never_enters_vae_decode(monkeypatch, stop_flag):
    monkeypatch.setattr(runtime, "dependencies", lambda: (torch, None, None, None))
    def materialize(*_):
        stop_flag[0] = True
        return object()
    monkeypatch.setattr(runtime, "materialize", materialize)
    monkeypatch.setattr(runtime, "decode_tensor", lambda *_: pytest.fail("Cancelled model entered decode"))
    monkeypatch.setattr(runtime, "clean_memory", lambda: None)
    component = components.Component("unused", "vae", "{}", device="cpu")
    latent = {"format": runtime.LATENT_FORMAT, "samples": torch.zeros(1, 128, 1, 1)}
    with pytest.raises(StopRequested):
        runtime.decode(component, latent)


def test_stop_during_model_construction_does_not_start_loading_weights(monkeypatch, stop_flag):
    class Transformer:
        @staticmethod
        def from_config(config):
            stop_flag[0] = True
            return torch.nn.Linear(1, 1)
    monkeypatch.setattr(runtime, "dependencies", lambda: (torch, None, None, Transformer))
    import safetensors.torch
    monkeypatch.setattr(safetensors.torch, "load_file", lambda *a, **k: pytest.fail("Cancelled build read weights"))
    component = components.Component("unused", "transformer", json.dumps({}), device="cpu")
    with pytest.raises(StopRequested):
        runtime.materialize(component, torch.device("cpu"))
