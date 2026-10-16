"""Submit a real workflow; save successful server history and PNG evidence."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
import urllib.error
import urllib.request


def request(url, data=None, timeout=120):
    payload = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(url, payload, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


class VerificationTimeout(TimeoutError):
    def __init__(self, report):
        self.report = report
        super().__init__(f"Timed out waiting for prompt {report['prompt_id']}; inspect its server queue/history before rerunning")


def timeout_report(url, prompt_id, timeout_seconds):
    report = {"prompt_id": prompt_id, "timeout_seconds": timeout_seconds}
    try:
        queue = request(url + "/queue", timeout=5)
        report.update({name: [item[1] for item in queue.get(name, [])]
                       for name in ("queue_running", "queue_pending")})
    except Exception as exc:
        report["queue_error"] = str(exc)
    return report


def wait_for_history(url, prompt_id, timeout_seconds, poll_interval=3):
    deadline = time.monotonic() + timeout_seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise VerificationTimeout(timeout_report(url, prompt_id, timeout_seconds))
        try:
            history = request(url + "/history/" + prompt_id, timeout=min(120, remaining))
        except (TimeoutError, urllib.error.URLError) as exc:
            if isinstance(exc, urllib.error.URLError) and not isinstance(exc.reason, TimeoutError):
                raise
            raise VerificationTimeout(timeout_report(url, prompt_id, timeout_seconds)) from exc
        if prompt_id in history:
            return history[prompt_id]
        time.sleep(min(poll_interval, max(0, deadline - time.monotonic())))


def validate_history(record, require_fresh=True):
    status = record.get("status", {})
    if status.get("status_str") != "success" or status.get("completed") is not True:
        raise RuntimeError(json.dumps(status, ensure_ascii=False))
    images = record.get("outputs", {}).get("7", {}).get("images")
    if not isinstance(images, list) or len(images) != 1:
        raise RuntimeError("Expected exactly one SaveImage output for this batch=1 verification")
    cached_events = [data for kind, data in status.get("messages", []) if kind == "execution_cached"]
    fresh = bool(cached_events) and all("5" not in map(str, data.get("nodes", [])) for data in cached_events)
    record["verification_sampler_fresh"] = fresh
    if require_fresh and not fresh:
        raise RuntimeError("A fresh sampler execution was not established; use --allow-cached to inspect cached results")
    return images


def download_image(url, path, size, timeout=120):
    from PIL import Image, ImageStat
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response, temporary.open("wb") as writer:
            shutil.copyfileobj(response, writer, length=64 * 1024)
        with Image.open(temporary) as image:
            image.load()
            if image.format != "PNG":
                raise RuntimeError(f"Expected a PNG image, got {image.format}")
            if image.size != (size, size):
                raise RuntimeError(f"Wrong generated image size: {image.size}")
            stddev = ImageStat.Stat(image.convert("RGB")).stddev
            if max(stddev) < 0.001:
                raise RuntimeError("Generated image is constant; inspect the model execution")
            record = {"filename": path.name, "size": list(image.size), "mode": image.mode,
                      "rgb_stddev": stddev, "sha256": hashlib.sha256(temporary.read_bytes()).hexdigest()}
        temporary.replace(path)
        return record
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8279")
    parser.add_argument("--variant", choices=["base", "sft"], default="sft")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--te-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--label", default="smoke")
    parser.add_argument("--timeout-seconds", type=float, default=1800,
                        help="Maximum time to wait for this prompt's history")
    parser.add_argument("--allow-cached", action="store_true",
                        help="Allow inspection of cached sampler output instead of requiring fresh sampling")
    args = parser.parse_args()
    args.url = args.url.rstrip("/")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be finite and positive")
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts" / f"{args.variant}-{args.label}"
    output.mkdir(parents=True, exist_ok=True)
    api = json.loads((root / f"example_workflows/moss-image-{args.variant}-api.json").read_text(encoding="utf-8"))
    api["2"]["inputs"]["device"] = args.te_device
    api["5"]["inputs"].update(width=args.size, height=args.size, steps=args.steps)
    api["7"]["inputs"]["filename_prefix"] = f"moss-{args.variant}-{args.label}"
    (output / "prompt.json").write_text(json.dumps(api, indent=2))
    start = time.monotonic()
    (output / "system.json").write_text(json.dumps(request(args.url + "/system_stats"), indent=2))
    response = request(args.url + "/prompt", {"prompt": api, "client_id": "moss-image-verification"})
    (output / "submission.json").write_text(json.dumps(response, indent=2))
    if response.get("node_errors"):
        raise RuntimeError(response["node_errors"])
    prompt_id = response["prompt_id"]
    print(f"Queued {prompt_id}: {args.variant} {args.size}px / {args.steps} steps", flush=True)
    try:
        record = wait_for_history(args.url, prompt_id, args.timeout_seconds)
    except VerificationTimeout as exc:
        (output / "timeout.json").write_text(json.dumps(exc.report, indent=2))
        raise
    record["verification_elapsed_seconds"] = round(time.monotonic() - start, 3)
    (output / "history.json").write_text(json.dumps(record, indent=2))
    images = validate_history(record, require_fresh=not args.allow_cached)
    image_records = []
    for item in images:
        from urllib.parse import urlencode
        url = args.url + "/view?" + urlencode(item)
        path = output / item["filename"]
        image_records.append(download_image(url, path, args.size))
    record["verification_images"] = image_records
    (output / "history.json").write_text(json.dumps(record, indent=2))
    print(f"Verified {output} in {time.monotonic()-start:.1f}s; sampler_fresh={record['verification_sampler_fresh']}", flush=True)


if __name__ == "__main__":
    main()
