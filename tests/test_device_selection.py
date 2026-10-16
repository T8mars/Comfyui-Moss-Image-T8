"""Respect ComfyUI's device policy and reclaim cached models before admission."""
import json
import sys
from types import ModuleType

import pytest
import torch

from moss_plugin.components import Component
from moss_plugin import runtime


def management(monkeypatch, device, release):
    module = ModuleType("comfy.model_management")
    module.get_torch_device = lambda: torch.device(device)
    module.free_memory = release
    module.get_free_memory = lambda device: torch.cuda.mem_get_info(device)[0]
    package = ModuleType("comfy")
    package.model_management = module
    monkeypatch.setitem(sys.modules, "comfy", package)
    monkeypatch.setitem(sys.modules, "comfy.model_management", module)
    return module


def descriptor(tmp_path, device="auto"):
    path = tmp_path / "component.safetensors"
    path.write_bytes(b"placeholder")
    return Component(str(path), "transformer", json.dumps({}), device=device)


def test_auto_obeys_comfy_forced_cpu(tmp_path, monkeypatch):
    management(monkeypatch, "cpu", lambda *_: pytest.fail("CPU must not reserve GPU memory"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_: pytest.fail("CPU must not query CUDA"))
    assert runtime.execution_device(descriptor(tmp_path)) == torch.device("cpu")


def test_auto_reclaims_comfy_models_on_selected_cuda_device(tmp_path, monkeypatch):
    free = [1024**3]
    calls = []
    def release(required, device):
        calls.append((required, device))
        free[0] = 8 * 1024**3
    management(monkeypatch, "cuda:1", release)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    def memory(device):
        assert device == torch.device("cuda:1")
        return free[0], 24 * 1024**3
    monkeypatch.setattr(torch.cuda, "mem_get_info", memory)
    assert runtime.execution_device(descriptor(tmp_path)) == torch.device("cuda:1")
    assert len(calls) == 1 and calls[0][0] > 2 * 1024**3


def test_explicit_cuda_reclaims_cached_models(tmp_path, monkeypatch):
    calls = []
    management(monkeypatch, "cpu", lambda required, device: calls.append((required, device)))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    assert runtime.execution_device(descriptor(tmp_path, "cuda")) == torch.device("cuda:0")
    assert len(calls) == 1 and calls[0][1] == torch.device("cuda:0")
