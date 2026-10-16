import io
import urllib.error

import pytest
from PIL import Image

from moss_plugin.scripts import verify_comfy
from moss_plugin.scripts.verify_canvas_results import check_receipts


def history(cached=(), images=None, completed=True):
    return {"status": {"status_str": "success", "completed": completed,
                       "messages": [["execution_cached", {"nodes": list(cached)}]]},
            "outputs": {"7": {"images": images if images is not None else [{"filename": "image.png"}]}}}


@pytest.mark.parametrize("images", [[], [{"filename": "a.png"}, {"filename": "b.png"}]])
def test_missing_or_excess_images_do_not_pass_verification(images):
    with pytest.raises(RuntimeError, match="exactly one"):
        verify_comfy.validate_history(history(images=images))


def test_success_without_completed_flag_does_not_pass():
    with pytest.raises(RuntimeError):
        verify_comfy.validate_history(history(completed=False))


def test_cached_sampler_is_explicitly_distinguished_from_fresh_sampling():
    record = history(cached=("5",))
    with pytest.raises(RuntimeError, match="fresh sampler"):
        verify_comfy.validate_history(record)
    assert verify_comfy.validate_history(record, require_fresh=False)
    assert record["verification_sampler_fresh"] is False
    fresh = history(cached=("1", "2", "3", "4"))
    assert verify_comfy.validate_history(fresh)
    assert fresh["verification_sampler_fresh"] is True


def test_absent_cache_execution_evidence_is_not_assumed_fresh():
    record = history()
    record["status"]["messages"] = []
    with pytest.raises(RuntimeError, match="fresh sampler"):
        verify_comfy.validate_history(record)


def test_later_cache_event_with_sampler_is_not_missed():
    record = history()
    record["status"]["messages"].append(["execution_cached", {"nodes": [5]}])
    with pytest.raises(RuntimeError, match="fresh sampler"):
        verify_comfy.validate_history(record)


def test_missing_history_has_deadline_and_records_only_queue_ids(monkeypatch):
    now = [0.0]
    calls = []
    monkeypatch.setattr(verify_comfy.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(verify_comfy.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    def request(url, **kwargs):
        calls.append(url)
        if url.endswith("/queue"):
            return {"queue_running": [[1, "active-id", {"private_prompt": "unused"}]], "queue_pending": []}
        return {}
    monkeypatch.setattr(verify_comfy, "request", request)
    with pytest.raises(verify_comfy.VerificationTimeout) as error:
        verify_comfy.wait_for_history("http://localhost", "submitted-id", 1, poll_interval=0.3)
    assert now[0] == 1
    assert error.value.report["prompt_id"] == "submitted-id"
    assert error.value.report["queue_running"] == ["active-id"]
    assert len(calls) < 10


@pytest.mark.parametrize("error", [TimeoutError("slow history"), urllib.error.URLError(TimeoutError("slow connect"))])
def test_transport_timeout_preserves_prompt_and_queue_diagnostics(monkeypatch, error):
    def request(url, **kwargs):
        if url.endswith("/queue"):
            return {"queue_running": [[1, "submitted-id"]]}
        raise error
    monkeypatch.setattr(verify_comfy, "request", request)
    with pytest.raises(verify_comfy.VerificationTimeout) as caught:
        verify_comfy.wait_for_history("http://localhost", "submitted-id", 1)
    assert caught.value.report["queue_running"] == ["submitted-id"]
    assert caught.value.__cause__ is error


def test_connection_failure_is_not_misreported_as_timeout(monkeypatch):
    def request(*args, **kwargs):
        raise urllib.error.URLError(ConnectionRefusedError("offline"))
    monkeypatch.setattr(verify_comfy, "request", request)
    with pytest.raises(urllib.error.URLError):
        verify_comfy.wait_for_history("http://localhost", "submitted-id", 1)


def encoded_image(format="PNG", size=4, constant=False):
    image = Image.new("RGB", (size, size))
    if not constant:
        image.putpixel((0, 0), (255, 128, 64))
    stream = io.BytesIO()
    image.save(stream, format=format)
    stream.seek(0)
    return stream


@pytest.mark.parametrize("image,reason", [(dict(format="JPEG"), "PNG"),
                                         (dict(size=8), "size"), (dict(constant=True), "constant")])
def test_invalid_image_cannot_replace_existing_evidence(monkeypatch, tmp_path, image, reason):
    target = tmp_path / "image.png"
    target.write_bytes(b"previous evidence")
    monkeypatch.setattr(verify_comfy.urllib.request, "urlopen", lambda *args, **kwargs: encoded_image(**image))
    with pytest.raises(RuntimeError, match=reason):
        verify_comfy.download_image("http://localhost/view", target, 4)
    assert target.read_bytes() == b"previous evidence"
    assert not target.with_name("image.png.partial").exists()


def test_image_download_uses_timeout_and_publishes_valid_png(monkeypatch, tmp_path):
    def urlopen(url, timeout):
        assert timeout == 0.5
        return encoded_image()
    monkeypatch.setattr(verify_comfy.urllib.request, "urlopen", urlopen)
    target = tmp_path / "image.png"
    record = verify_comfy.download_image("http://localhost/view", target, 4, timeout=0.5)
    assert record["size"] == [4, 4]
    with Image.open(target) as image:
        assert image.format == "PNG"


def test_image_timeout_keeps_existing_png_and_removes_partial(monkeypatch, tmp_path):
    target = tmp_path / "image.png"
    target.write_bytes(b"previous evidence")
    class SlowResponse(io.BytesIO):
        def read(self, *args):
            raise TimeoutError("stalled image")
    monkeypatch.setattr(verify_comfy.urllib.request, "urlopen", lambda *args, **kwargs: SlowResponse())
    with pytest.raises(TimeoutError):
        verify_comfy.download_image("http://localhost/view", target, 4)
    assert target.read_bytes() == b"previous evidence"
    assert not target.with_name("image.png.partial").exists()


@pytest.mark.parametrize("mismatch", ["proof_id", "submission_id", "image", "node_errors"])
def test_offline_canvas_receipts_must_belong_to_the_same_execution(mismatch):
    record = history()
    record["prompt"] = [0, "actual-id"]
    images = record["outputs"]["7"]["images"]
    proof = {"prompt_id": "actual-id", "sampler_executed_fresh": True,
             "preview_verified": True, "images": images}
    submission = {"prompt_id": "actual-id", "node_errors": {}}
    assert check_receipts(proof, submission, record) == images
    if mismatch == "proof_id":
        proof["prompt_id"] = "stale-id"
    elif mismatch == "submission_id":
        submission["prompt_id"] = "other-id"
    elif mismatch == "image":
        proof["images"] = [{"filename": "different.png"}]
    else:
        submission["node_errors"] = {"5": "invalid"}
    with pytest.raises(AssertionError):
        check_receipts(proof, submission, record)


def test_offline_canvas_proof_cannot_override_cached_server_history():
    record = history(cached=("5",))
    record["prompt"] = [0, "actual-id"]
    proof = {"prompt_id": "actual-id", "sampler_executed_fresh": True,
             "preview_verified": True, "images": record["outputs"]["7"]["images"]}
    with pytest.raises(RuntimeError, match="fresh sampler"):
        check_receipts(proof, {"prompt_id": "actual-id"}, record)
