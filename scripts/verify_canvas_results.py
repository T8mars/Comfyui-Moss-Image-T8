"""Verify browser Run receipts and PNG-embedded workflows; publish drag-in PNGs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageStat

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_prompt(found, expected):
    assert found.keys() == expected.keys()
    for key, node in expected.items():
        assert found[key]["class_type"] == node["class_type"]
        assert found[key]["inputs"] == node["inputs"], key


def check_gui(found, expected):
    actual = {str(node["id"]): node for node in found["nodes"]}
    assert len(actual) == len(expected["nodes"]) == 7
    for node in expected["nodes"]:
        other = actual[str(node["id"])]
        assert other["type"] == node["type"]
        assert other["widgets_values"] == node["widgets_values"], (node["id"], other["widgets_values"])
    assert found["links"] == expected["links"]


def main():
    report = []
    for variant in ("base", "sft"):
        folder = ROOT / "artifacts" / f"canvas-{variant}"
        proof = read(folder / "canvas-proof.json")
        request = read(folder / "browser-request.json")
        history = read(folder / "history.json")
        gui_path = ROOT / f"example_workflows/moss-image-{variant}.json"
        gui = read(gui_path)
        expected = read(ROOT / f"example_workflows/moss-image-{variant}-api.json")
        assert hashlib.sha256(gui_path.read_bytes()).hexdigest() == proof["workflow_sha256"]
        assert history["status"]["status_str"] == "success"
        assert proof["sampler_executed_fresh"] and proof["preview_verified"]
        assert not proof["page_errors"], proof["page_errors"]
        check_prompt(request["prompt"], expected)
        check_prompt(history["prompt"][2], expected)
        check_gui(request["extra_data"]["extra_pnginfo"]["workflow"], gui)
        assert expected["5"]["inputs"]["width"] == expected["5"]["inputs"]["height"] == 1024
        images = history["outputs"]["7"]["images"]
        assert len(images) == 1
        path = folder / images[0]["filename"]
        with Image.open(path) as image:
            image.load()
            assert image.size == (1024, 1024)
            assert image.mode == "RGB"
            assert max(ImageStat.Stat(image).stddev) > 0.001
            check_gui(json.loads(image.info["workflow"]), gui)
            check_prompt(json.loads(image.info["prompt"]), expected)
            pixels = np.array(image)
        record = {"variant": variant, "source": proof["source"], "prompt_id": proof["prompt_id"],
                  "workflow_sha256": proof["workflow_sha256"], "sampler_executed_fresh": True,
                  "canvas_preview_size": proof["after"]["preview_sizes"],
                  "fixed_seed_after_run": next(item["value"] for item in proof["after"]["sampler_widgets"] if item["name"] == "seed"),
                  "parameters": expected["5"]["inputs"], "png_size": [1024, 1024],
                  "png_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                  "png_embedded_prompt_and_gui_exact": True}
        if variant == "sft":
            reference = ROOT / "artifacts/sft-quality/moss-sft-quality_00001_.png"
            with Image.open(reference) as original:
                expected_pixels = np.array(original.convert("RGB"))
            difference = np.abs(pixels.astype(np.int16) - expected_pixels.astype(np.int16))
            record["against_previous_direct_api_run"] = {
                "pixels_identical": bool(np.array_equal(pixels, expected_pixels)),
                "max_absolute_rgb_difference": int(difference.max()),
                "mean_absolute_rgb_difference": float(difference.mean())}
        destination = ROOT / f"example_workflows/moss-image-{variant}.png"
        shutil.copyfile(path, destination)
        record["drag_in_png"] = str(destination.relative_to(ROOT)).replace("\\", "/")
        report.append(record)
        print(f"PASS: {variant} real canvas Run, 1024/50, fixed42, preview, exact PNG workflow metadata")
    destination = ROOT / "artifacts/canvas-validation.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
