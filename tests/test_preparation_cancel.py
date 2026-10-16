import json
from pathlib import Path
import subprocess
import sys
from concurrent.futures import CancelledError, ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest
import requests
import torch
from safetensors.torch import load_file, save_file

from moss_plugin.scripts import prepare_models


def test_ctrl_c_stops_executor_workers(tmp_path):
    # Isolate the real interrupt so it cannot interfere with pytest's own loop.
    source = Path(prepare_models.__file__).resolve()
    code = '''
import _thread, importlib.util, threading, time
from concurrent.futures import ThreadPoolExecutor
spec = importlib.util.spec_from_file_location("preparation", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
cancel = threading.Event()
with ThreadPoolExecutor(max_workers=1) as pool:
    job = pool.submit(cancel.wait, 10)
    timer = threading.Timer(0.05, _thread.interrupt_main)
    timer.start()
    started = time.monotonic()
    try:
        module.wait_for_jobs([job], cancel)
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("Expected KeyboardInterrupt")
    assert cancel.is_set()
    assert time.monotonic() - started < 2
    timer.join()
assert job.done()
'''.replace("SOURCE", repr(str(source)))
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=8)
    assert result.returncode == 0, result.stdout + result.stderr


def test_later_worker_failure_cancels_earlier_job_and_keeps_real_error():
    cancel = Event()
    started = Event()
    def slow():
        started.set()
        assert cancel.wait(3), "Other worker's error must stop this job"
        raise CancelledError()
    def fails():
        assert started.wait(2)
        raise TimeoutError("original worker timeout")
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(slow), pool.submit(fails)]
        with pytest.raises(TimeoutError, match="original worker timeout"):
            prepare_models.wait_for_jobs(futures, cancel)
    assert cancel.is_set()


def test_cooperative_cancel_does_not_mask_other_workers_error():
    cancel = Event()
    first_stopped = Event()
    def stopped():
        cancel.set()
        first_stopped.set()
        raise CancelledError()
    def actual_failure():
        assert first_stopped.wait(2)
        raise ValueError("original download failure")
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(stopped), pool.submit(actual_failure)]
        with pytest.raises(ValueError, match="original download failure"):
            prepare_models.wait_for_jobs(futures, cancel)


def test_download_error_cancels_other_workers_before_executor_shutdown(tmp_path, monkeypatch):
    from huggingface_hub import HfApi
    cancel = Event()
    started = Event()
    items = [SimpleNamespace(rfilename=f"vae/{name}", lfs=None, blob_id="pinned", size=1)
             for name in ("slow.safetensors", "failed.safetensors")]
    monkeypatch.setattr(HfApi, "model_info", lambda *args, **kwargs: SimpleNamespace(siblings=items))
    def verification(path, *args):
        if path.name == "slow.safetensors":
            started.set()
            assert cancel.wait(3), "Failure must set cancellation before waiting for this worker"
            raise CancelledError()
        assert started.wait(2)
        raise ValueError("original file verification error")
    monkeypatch.setattr(prepare_models, "verify_official_file", verification)
    with pytest.raises(ValueError, match="original file verification error"):
        prepare_models.download_variant("base", tmp_path, ["vae"], 2, cancel)
    assert cancel.is_set()
    assert not (tmp_path / "base/verified-vae.json").exists()


def test_cancel_during_publish_preserves_recoverable_export(tmp_path, monkeypatch):
    source = tmp_path / "source"
    vae = source / "vae"
    vae.mkdir(parents=True)
    values = {"a": torch.arange(4, dtype=torch.float32)}
    save_file(values, vae / "diffusion_pytorch_model.safetensors")
    (vae / "config.json").write_text(json.dumps({"_class_name": "AutoencoderKLFlux2"}))
    (source / "verified-vae.json").write_text(json.dumps({"repo": "test", "revision": "pinned", "files": {}}))
    target = tmp_path / "vae.safetensors"
    cancel = Event()
    original = prepare_models.atomic_json
    def cancel_after_pending(path, value):
        original(path, value)
        if value.get("pending_sha256"):
            cancel.set()
    monkeypatch.setattr(prepare_models, "atomic_json", cancel_after_pending)
    with pytest.raises(CancelledError):
        prepare_models.export_component(source, "vae", target, cancel)
    assert not target.exists()
    assert Path(str(target) + ".partial").is_file()
    manifest = Path(str(target) + ".assets/manifest.json")
    assert not json.loads(manifest.read_text())["ready"]
    monkeypatch.setattr(prepare_models, "atomic_json", original)
    cancel.clear()
    prepare_models.export_component(source, "vae", target, cancel)
    assert json.loads(manifest.read_text())["ready"]
    torch.testing.assert_close(load_file(target)["a"], values["a"], atol=0, rtol=0)


def test_cancel_exits_retry_backoff_and_keeps_download_prefix(tmp_path, monkeypatch):
    partial = tmp_path / "weight.partial"
    partial.write_bytes(b"verified-prefix")
    cancel = Event()
    attempts = []
    def unavailable(*args, **kwargs):
        attempts.append(1)
        cancel.set()
        raise requests.ConnectionError("interrupted connection")
    monkeypatch.setattr(requests.Session, "get", unavailable)
    with pytest.raises(CancelledError):
        prepare_models.range_download("http://127.0.0.1/weight", partial, 100, cancel)
    assert len(attempts) == 1
    assert partial.read_bytes() == b"verified-prefix"


def test_http_cancel_observed_before_full_transfer_or_idle_timeout(tmp_path):
    size = 1024 * 1024
    headers_sent, send_prefix, finish = Event(), Event(), Event()
    cancel = Event()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(206)
            self.send_header("Content-Range", f"bytes 0-{size - 1}/{size}")
            self.send_header("Content-Length", str(size))
            self.end_headers()
            headers_sent.set()
            assert send_prefix.wait(3)
            try:
                self.wfile.write(b"x" * (64 * 1024))
                self.wfile.flush()
                finish.wait(3)
                self.wfile.write(b"x" * (size - 64 * 1024))
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(prepare_models.range_download, f"http://127.0.0.1:{server.server_port}/weight",
                                 tmp_path / "download.partial", size, cancel)
            try:
                assert headers_sent.wait(3)
                cancel.set()
                send_prefix.set()
                with pytest.raises(CancelledError):
                    future.result(timeout=1.5)
            finally:
                send_prefix.set()
                finish.set()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
