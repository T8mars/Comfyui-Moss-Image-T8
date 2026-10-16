"""Actual process locks protect successful packages and shared CLI namespaces."""
from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys
import time

import pytest

from moss_plugin.components import inspect_component
from moss_plugin.scripts import prepare_models as preparation
from moss_plugin.tests.test_export_integrity import sources


@contextmanager
def another_process_holds(lock_path, tmp_path):
    ready = tmp_path / "child-ready"
    code = """
import importlib.util, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location('preparation', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
with module.exclusive_path(sys.argv[2]):
    Path(sys.argv[3]).write_text('locked')
    sys.stdin.buffer.read(1)
"""
    child = subprocess.Popen([sys.executable, "-c", code, str(Path(preparation.__file__).resolve()),
                              str(lock_path), str(ready)], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists(), "Child did not acquire its preparation lock"
        yield child
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=5)


@pytest.mark.parametrize("crash", [False, True])
def test_process_lock_conflict_and_release_after_exit(tmp_path, crash):
    path = tmp_path / "prepare.lock"
    with another_process_holds(path, tmp_path) as child:
        with pytest.raises(RuntimeError, match="Another MOSS-Image preparation"):
            with preparation.exclusive_path(path):
                pytest.fail("Two processes acquired the same lock")
        if crash:
            child.kill()
            child.communicate(timeout=5)
        else:
            child.communicate(input=b"\n", timeout=5)
            assert child.returncode == 0
        with preparation.exclusive_path(path):
            pass


def test_busy_target_is_untouched_while_another_target_can_export(tmp_path):
    source, _ = sources(tmp_path)
    target = tmp_path / "vae.safetensors"
    preparation.export_component(source, "vae", target)
    assets = Path(str(target) + ".assets")
    manifest = (assets / "manifest.json").read_bytes()
    weight = target.read_bytes()
    with another_process_holds(str(target.resolve()) + ".prepare.lock", tmp_path):
        with pytest.raises(RuntimeError, match="Another MOSS-Image preparation"):
            preparation.export_component(source, "vae", target)
        assert target.read_bytes() == weight
        assert (assets / "manifest.json").read_bytes() == manifest
        assert inspect_component(target, "vae")
        other = tmp_path / "other.safetensors"
        preparation.export_component(source, "vae", other)
        assert inspect_component(other, "vae")


def test_cli_rejects_a_shared_busy_cache_before_starting_jobs(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    lock = cache / ".moss-image-prepare.lock"
    with preparation.exclusive_path(lock):
        monkeypatch.setattr(sys, "argv", ["prepare_models", "--models-dir", str(tmp_path / "models"),
                                          "--cache-dir", str(cache)])
        monkeypatch.setattr(preparation, "download_variant", lambda *a, **k: pytest.fail("Busy CLI started a download"))
        with pytest.raises(RuntimeError, match="Another MOSS-Image preparation"):
            preparation.main()
