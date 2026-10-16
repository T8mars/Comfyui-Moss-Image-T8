"""Audit shipped parameters against pinned upstream code and real weight bytes."""
from __future__ import annotations

import argparse
import ast
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import struct

from prepare_models import CHUNK, REVISIONS, sha256, tensor_header, verify_official_file

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_PIPELINE_SHA256 = "1dac1aca9feb1bfdfc727a401e4b67d61a505d3190706d1b7d2d784f88678b31"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def function(tree, class_name, function_name):
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    return next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == function_name)


def defaults(node):
    values = {argument.arg: ast.literal_eval(value) for argument, value in
              zip(node.args.kwonlyargs, node.args.kw_defaults) if value is not None}
    values.update({argument.arg: ast.literal_eval(value) for argument, value in
                   zip(node.args.args[-len(node.args.defaults):], node.args.defaults)})
    return values


def metadata(path):
    with Path(path).open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        assert 0 < length < 100_000_000
        return json.loads(stream.read(length))["__metadata__"]


def compare_raw_tensors(source, kind, target):
    source = source / kind
    if kind == "vae":
        source_header, _ = tensor_header(source / "diffusion_pytorch_model.safetensors")
        mapping = {key: "diffusion_pytorch_model.safetensors" for key in source_header}
    else:
        name = "model.safetensors.index.json" if kind == "text_encoder" else "diffusion_pytorch_model.safetensors.index.json"
        mapping = read(source / name)["weight_map"]
    actual, target_start = tensor_header(target)
    assert actual.keys() == mapping.keys(), f"Missing or extra tensors: {target.name}"
    source_headers = {name: tensor_header(source / name) for name in set(mapping.values())}
    for name, (header, _) in source_headers.items():
        assert header.keys() == {key for key, value in mapping.items() if value == name}
    count = 0
    with ExitStack() as stack:
        original = {name: stack.enter_context((source / name).open("rb")) for name in source_headers}
        exported = stack.enter_context(target.open("rb"))
        for key in sorted(mapping):
            header, source_start = source_headers[mapping[key]]
            expected = header[key]
            found = actual[key]
            assert found["dtype"] == expected["dtype"] and found["shape"] == expected["shape"], key
            left, right = expected["data_offsets"]
            assert right - left == found["data_offsets"][1] - found["data_offsets"][0]
            reader = original[mapping[key]]
            reader.seek(source_start + left)
            exported.seek(target_start + found["data_offsets"][0])
            remaining = right - left
            while remaining:
                length = min(CHUNK, remaining)
                before, after = reader.read(length), exported.read(length)
                assert len(before) == length and before == after, f"Tensor bytes differ: {key}"
                remaining -= length
                count += length
    return {"tensor_count": len(mapping), "source_shard_count": len(source_headers),
            "compared_tensor_bytes": count, "all_shapes_dtypes_and_tensor_bytes_equal": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-raw", action="store_true", help="Also compare every exported tensor byte with downloaded official shards")
    args = parser.parse_args()
    source_path = ROOT / "vendor/pipeline_moss_image.py"
    source = source_path.read_bytes()
    assert hashlib.sha256(source).hexdigest() == OFFICIAL_PIPELINE_SHA256
    tree = ast.parse(source)
    pipe_defaults = defaults(function(tree, "MOSSImagePipeline", "__call__"))
    scheduler_defaults = defaults(function(tree, "MOSSImageScheduler", "__init__"))
    official_blob = hashlib.sha1(f"blob {len(source)}\0".encode() + source).hexdigest()

    from huggingface_hub import HfApi
    remote = {}
    for variant, revision in REVISIONS.items():
        repo = f"OpenMOSS-Team/MOSS-Image-{'Base' if variant == 'base' else 'SFT'}"
        info = HfApi().model_info(repo, revision=revision, files_metadata=True)
        assert info.sha == revision
        files = {entry.rfilename: entry for entry in info.siblings}
        assert files["pipeline_moss_image.py"].blob_id == official_blob
        remote[variant] = {"repo": repo, "revision": revision, "files": files}
    shared = []
    for name, item in remote["base"]["files"].items():
        if name.startswith(("text_encoder/", "tokenizer/", "vae/")) and not name.endswith(".py"):
            other = remote["sft"]["files"][name]
            assert item.blob_id == other.blob_id
            if item.lfs:
                assert item.lfs.sha256 == other.lfs.sha256
            shared.append(name)

    runtime_tree = ast.parse((ROOT / "runtime.py").read_text(encoding="utf-8"))
    sample = next(node for node in runtime_tree.body if isinstance(node, ast.FunctionDef) and node.name == "sample")
    pipe_call = next(node for node in ast.walk(sample) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Name) and node.func.id == "pipe")
    forwarding = {item.arg: ast.unparse(item.value) for item in pipe_call.keywords}
    expected_forwarding = {"height": "height", "width": "width", "num_inference_steps": "steps",
                           "guidance_scale": "cfg", "num_images_per_prompt": "batch_size",
                           "uncond_skip_layers": "skip", "slg_layers": "slg", "slg_scale": "slg_scale",
                           "slg_start": "slg_start", "slg_stop": "slg_stop",
                           "generator": "torch.Generator('cpu').manual_seed(seed)", "output_type": "'latent'"}
    for key, value in expected_forwarding.items():
        assert forwarding[key] == value, (key, forwarding[key])

    workflow_audits = []
    for variant in ("base", "sft"):
        path = ROOT / f"example_workflows/moss-image-{variant}-api.json"
        api = read(path)
        sampler = api["5"]["inputs"]
        pairs = {"height": "height", "width": "width", "steps": "num_inference_steps",
                 "cfg": "guidance_scale", "batch_size": "num_images_per_prompt",
                 "slg_scale": "slg_scale", "slg_start": "slg_start", "slg_stop": "slg_stop"}
        for widget, official in pairs.items():
            assert sampler[widget] == pipe_defaults[official], (variant, widget)
        assert sampler["shift"] == scheduler_defaults["shift"]
        assert [int(value) for value in sampler["uncond_skip_layers"].split(",")] == list(pipe_defaults["uncond_skip_layers"])
        assert sampler["slg_layers"] == "" and pipe_defaults["slg_layers"] is None
        assert sampler["seed"] == 42
        assert api["4"]["inputs"]["negative"] == pipe_defaults["negative_prompt"]
        assert api["4"]["inputs"]["max_sequence_length"] == pipe_defaults["max_sequence_length"]
        assert api["6"]["inputs"]["tiled"] is False
        workflow_audits.append({"variant": variant, "all_values_match_pinned_official_defaults": True,
                                "gui_sha256": sha256(ROOT / f"example_workflows/moss-image-{variant}.json"),
                                "sampler": sampler})

    model_audits = []
    components = [("base", "transformer", "diffusion_models/moss-image-base-bf16.safetensors"),
                  ("sft", "transformer", "diffusion_models/moss-image-sft-bf16.safetensors"),
                  ("base", "text_encoder", "text_encoders/moss-image-qwen35-bf16.safetensors"),
                  ("base", "vae", "vae/moss-image-vae.safetensors")]
    for variant, kind, relative in components:
        target = ROOT / "models" / relative
        print(f"Audit full SHA256: {relative}", flush=True)
        manifest = read(str(target) + ".assets/manifest.json")
        actual_sha = sha256(target)
        assert manifest["ready"] and manifest["sha256"] == actual_sha and manifest["size"] == target.stat().st_size
        meta = metadata(target)
        assert meta["moss_image.component"] == kind
        official_config_name = f"{kind}/config.json"
        item = remote[variant]["files"][official_config_name]
        official_config_path = ROOT / "_downloads" / variant / official_config_name
        assert verify_official_file(official_config_path, item.size,
                                    item.lfs.sha256 if item.lfs else None, None if item.lfs else item.blob_id)
        config = json.loads(meta["moss_image.config"])
        assert config == read(official_config_path)
        if kind == "transformer":
            assert config["num_layers"] == 42 and 0 <= 22 < config["num_layers"]
            assert config["latent_channels"] == 128 and config["text_dim"] == 4096
            assert json.loads(meta["moss_image.scheduler"]) == read(ROOT / "_downloads" / variant / "scheduler/scheduler_config.json")
        if kind == "text_encoder":
            for name, expected in manifest["tokenizer_files"].items():
                assert sha256(Path(str(target) + ".assets/tokenizer") / name) == expected
                origin = remote[variant]["files"][f"tokenizer/{name}"]
                assert verify_official_file(Path(str(target) + ".assets/tokenizer") / name, origin.size,
                                            origin.lfs.sha256 if origin.lfs else None, None if origin.lfs else origin.blob_id)
        record = {"file": relative, "sha256": actual_sha, "bytes": target.stat().st_size,
                  "component": kind, "official_config_matches": True, "source_revision": REVISIONS[variant]}
        if args.verify_raw:
            print(f"Compare every original tensor byte: {relative}", flush=True)
            record["raw_tensor_audit"] = compare_raw_tensors(ROOT / "_downloads" / variant, kind, target)
        model_audits.append(record)
        print(f"PASS: {relative}", flush=True)
    report = {"pipeline_sha256": OFFICIAL_PIPELINE_SHA256, "both_pinned_remote_pipeline_blobs_match": True,
              "official_pipeline_defaults_from_ast": pipe_defaults, "official_scheduler_defaults_from_ast": scheduler_defaults,
              "runtime_forwarding_verified": expected_forwarding, "shared_official_files_verified": shared,
              "workflows": workflow_audits, "independent_models": model_audits}
    destination = ROOT / "artifacts/parameter-audit.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"PASS: pinned official parameters, independent full SHA256, {len(model_audits)} components; {destination}", flush=True)


if __name__ == "__main__":
    main()
