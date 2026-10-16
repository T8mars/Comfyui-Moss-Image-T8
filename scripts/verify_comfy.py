"""Submit a real workflow; save successful server history and PNG evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def request(url, data=None):
    payload = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(url, payload, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8279")
    parser.add_argument("--variant", choices=["base", "sft"], default="sft")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--te-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--label", default="smoke")
    args = parser.parse_args()
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
    while True:
        history = request(args.url + "/history/" + prompt_id)
        if prompt_id in history:
            record = history[prompt_id]
            record["verification_elapsed_seconds"] = round(time.monotonic() - start, 3)
            (output / "history.json").write_text(json.dumps(record, indent=2))
            if record.get("status", {}).get("status_str") != "success":
                raise RuntimeError(json.dumps(record.get("status"), ensure_ascii=False))
            image_records = []
            for item in record["outputs"]["7"]["images"]:
                from urllib.parse import urlencode
                from PIL import Image, ImageStat
                url = args.url + "/view?" + urlencode(item)
                path = output / item["filename"]
                urllib.request.urlretrieve(url, path)
                with Image.open(path) as image:
                    image.load()
                    if image.size != (args.size, args.size):
                        raise RuntimeError(f"Wrong generated image size: {image.size}")
                    rgb = image.convert("RGB")
                    stddev = ImageStat.Stat(rgb).stddev
                    if max(stddev) < 0.001:
                        raise RuntimeError("Generated image is constant; inspect the model execution")
                    image_records.append({"filename": path.name, "size": list(image.size),
                                          "mode": image.mode, "rgb_stddev": stddev,
                                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            record["verification_images"] = image_records
            (output / "history.json").write_text(json.dumps(record, indent=2))
            print(f"Verified {output} in {time.monotonic()-start:.1f}s", flush=True)
            return
        time.sleep(3)


if __name__ == "__main__":
    main()
