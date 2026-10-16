import json
import hashlib
import struct
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import torch
from safetensors.torch import load_file, save_file

from moss_plugin.scripts.prepare_models import merge_component, tensor_header
from moss_plugin.scripts import prepare_models


def shards(tmp_path):
    values = {"a": torch.arange(12, dtype=torch.float32).reshape(3, 4),
              "b": torch.tensor([0.125, -1.5], dtype=torch.bfloat16),
              "c": torch.tensor([1, 5, -4], dtype=torch.int64)}
    save_file({"a": values["a"]}, tmp_path / "one.safetensors")
    save_file({"b": values["b"], "c": values["c"]}, tmp_path / "two.safetensors")
    (tmp_path / "index.json").write_text(json.dumps({"weight_map": {"a": "one.safetensors", "b": "two.safetensors", "c": "two.safetensors"}}))
    return values


def test_stream_merge_preserves_every_byte(tmp_path):
    values = shards(tmp_path)
    target = tmp_path / "out.safetensors"
    merge_component(tmp_path, "index.json", target, {"format": "pt"})
    actual = load_file(target)
    for key in values:
        assert actual[key].dtype == values[key].dtype
        assert torch.equal(actual[key], values[key])
        assert actual[key].view(torch.uint8).numpy().tobytes() == values[key].view(torch.uint8).numpy().tobytes()
    with pytest.raises(FileExistsError):
        merge_component(tmp_path, "index.json", target, {})


def test_index_and_path_tampering_rejected(tmp_path):
    shards(tmp_path)
    (tmp_path / "index.json").write_text(json.dumps({"weight_map": {"wrong": "one.safetensors"}}))
    with pytest.raises(ValueError, match="keys"):
        merge_component(tmp_path, "index.json", tmp_path / "out", {})
    (tmp_path / "index.json").write_text(json.dumps({"weight_map": {"a": "../one.safetensors"}}))
    with pytest.raises(ValueError, match="escapes"):
        merge_component(tmp_path, "index.json", tmp_path / "out", {})


@pytest.mark.parametrize("offsets", [[0, 1], [-1, 4], [0, 100]])
def test_corrupt_offsets_rejected(tmp_path, offsets):
    header = json.dumps({"a": {"dtype": "F32", "shape": [1], "data_offsets": offsets}}).encode()
    path = tmp_path / "bad.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"1234")
    with pytest.raises(ValueError):
        tensor_header(path)


def test_export_recovers_publish_window_and_repairs_tokenizer(tmp_path, monkeypatch):
    source = tmp_path / "source"
    te = source / "text_encoder"
    tokens = source / "tokenizer"
    te.mkdir(parents=True)
    tokens.mkdir()
    save_file({"a": torch.ones(2)}, te / "model.safetensors")
    (te / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "model.safetensors"}}))
    (te / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    token_names = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
    for name in token_names:
        (tokens / name).write_text("official " + name)
    (source / "verified.json").write_text(json.dumps({"repo": "test", "revision": "pinned", "files": {
        f"tokenizer/{name}": {"sha256": prepare_models.sha256(tokens / name)} for name in token_names}}))
    target = tmp_path / "out.safetensors"
    original = prepare_models.atomic_json
    def interrupted_publish(path, value):
        if value.get("ready"):
            raise RuntimeError("interrupted before ready manifest")
        original(path, value)
    monkeypatch.setattr(prepare_models, "atomic_json", interrupted_publish)
    with pytest.raises(RuntimeError, match="interrupted"):
        prepare_models.export_component(source, "text_encoder", target)
    assert target.exists()
    monkeypatch.setattr(prepare_models, "atomic_json", original)
    assets = tmp_path / "out.safetensors.assets"
    (assets / "tokenizer/tokenizer.json").write_text("damaged")
    prepare_models.export_component(source, "text_encoder", target)
    assert json.loads((assets / "manifest.json").read_text())["ready"]
    assert (assets / "tokenizer/tokenizer.json").read_text() == "official tokenizer.json"


def test_http_download_resumes_exact_range(tmp_path):
    content = bytes(range(256)) * 1024
    requested = []
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            value = self.headers["Range"]
            requested.append(value)
            start, end = map(int, value.removeprefix("bytes=").split("-"))
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(content)}")
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            self.wfile.write(content[start:end + 1])
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        partial = tmp_path / "weight.partial"
        partial.write_bytes(content[:12345])
        prepare_models.range_download(f"http://127.0.0.1:{server.server_port}/weight", partial, len(content))
        assert partial.read_bytes() == content
        assert requested == [f"bytes=12345-{len(content) - 1}"]
    finally:
        server.shutdown()
        server.server_close()


def test_non_lfs_cache_is_bound_to_official_git_blob(tmp_path):
    path = tmp_path / "config.json"
    content = b'{"version":1}'
    blob_id = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
    path.write_bytes(content)
    assert prepare_models.verify_official_file(path, len(content), git_blob_id=blob_id)
    path.write_bytes(b'{"version":2}')
    assert prepare_models.verify_official_file(path, len(content), git_blob_id=blob_id) is None
