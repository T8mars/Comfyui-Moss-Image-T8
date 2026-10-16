"""Download pinned official weights and package EACH component as its own file.

No tensors are quantized, pruned, or combined across components. Export copies
the original tensor bytes; RAM usage stays bounded while packaging the 9B TE.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import time
from concurrent.futures import ThreadPoolExecutor, CancelledError, wait, FIRST_COMPLETED
from threading import Event

REVISIONS = {
    "base": "5f40f07a304541840b6019d2d34d30f966756787",
    "sft": "daf706d2e1c7a6369c17b6d1658671aee6db1ebe",
}
DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "U16": 2,
               "F16": 2, "BF16": 2, "I32": 4, "U32": 4, "F32": 4,
               "F64": 8, "I64": 8, "U64": 8, "F8_E4M3": 1, "F8_E5M2": 1}
CHUNK = 8 * 1024 * 1024


def check_cancelled(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledError("Model preparation cancelled; partial downloads are preserved")


def wait_for_jobs(futures, cancel_event):
    # Finite waits let Windows/Python 3.10 deliver Ctrl+C while workers run.
    try:
        pending = set(futures)
        while pending:
            wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
            completed = {future for future in pending if future.done()}
            pending -= completed
            errors = [future.exception() for future in completed if not future.cancelled()]
            for error in errors:
                if error is not None and not isinstance(error, CancelledError):
                    raise error
            if any(future.cancelled() for future in completed) or any(isinstance(e, CancelledError) for e in errors):
                cancel_event.set()
        check_cancelled(cancel_event)
    except BaseException:
        cancel_event.set()
        for future in futures:
            future.cancel()
        raise


def sha256(path, cancel_event=None):
    check_cancelled(cancel_event)
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            check_cancelled(cancel_event)
            result.update(block)
    return result.hexdigest()


def verify_official_file(path, size, lfs_sha256=None, git_blob_id=None, cancel_event=None):
    """Return local SHA256 only when official LFS or Git-blob content matches."""
    path = Path(path)
    check_cancelled(cancel_event)
    if not path.is_file() or path.stat().st_size != size:
        return None
    if not lfs_sha256 and not git_blob_id:
        raise ValueError(f"Official file metadata has no content hash: {path.name}")
    digest = hashlib.sha256()
    blob = hashlib.sha1(f"blob {size}\0".encode()) if not lfs_sha256 else None
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            check_cancelled(cancel_event)
            digest.update(block)
            if blob is not None:
                blob.update(block)
    actual = digest.hexdigest()
    if lfs_sha256:
        return actual if actual == lfs_sha256 else None
    return actual if blob.hexdigest() == git_blob_id else None


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def child_path(root, name):
    result = (Path(root) / name).resolve()
    if not result.is_relative_to(Path(root).resolve()) or result == Path(root).resolve():
        raise ValueError(f"Path escapes component directory: {name}")
    return result


def tensor_header(path):
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError(f"Truncated safetensors: {path}")
        length = struct.unpack("<Q", raw)[0]
        if length > 100_000_000 or length + 8 > size:
            raise ValueError(f"Invalid header length: {path}")
        def unique_pairs(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"Duplicate JSON key: {key}")
                result[key] = value
            return result
        header = json.loads(stream.read(length), object_pairs_hook=unique_pairs)
    if not isinstance(header, dict):
        raise ValueError("Safetensors header must be an object")
    header.pop("__metadata__", None)
    ranges = []
    for key, item in header.items():
        dtype, shape, offsets = item["dtype"], item["shape"], item["data_offsets"]
        if dtype not in DTYPE_BYTES or not isinstance(shape, list) or any(type(d) is not int or d < 0 for d in shape):
            raise ValueError(f"Invalid tensor dtype/shape: {key}")
        if len(offsets) != 2 or any(type(d) is not int for d in offsets):
            raise ValueError(f"Invalid tensor offsets: {key}")
        start, end = offsets
        if start < 0 or end < start or end > size - 8 - length:
            raise ValueError(f"Tensor outside file: {key}")
        if end - start != math.prod(shape) * DTYPE_BYTES[dtype]:
            raise ValueError(f"Tensor byte count mismatch: {key}")
        ranges.append((start, end))
    cursor = 0
    for start, end in sorted(ranges):
        if start != cursor:
            raise ValueError(f"Overlapping or non-contiguous tensor data: {path}")
        cursor = end
    if cursor != size - 8 - length:
        raise ValueError(f"Unindexed trailing bytes: {path}")
    return header, 8 + length


def merge_component(directory, index_name, target, metadata, before_publish=None, cancel_event=None):
    """Validate an HF index and stream its original tensor bytes into ONE file."""
    directory, target = Path(directory), Path(target)
    check_cancelled(cancel_event)
    mapping = read_json(directory / index_name)["weight_map"]
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Empty/invalid weight index")
    tensors = {}
    for name in sorted(set(mapping.values())):
        check_cancelled(cancel_event)
        path = child_path(directory, name)
        header, data_start = tensor_header(path)
        expected = {key for key, shard in mapping.items() if shard == name}
        if header.keys() != expected:
            raise ValueError(f"Shard keys do not match index: {name}")
        for key, info in header.items():
            if key in tensors:
                raise ValueError(f"Duplicate tensor: {key}")
            tensors[key] = (path, data_start, info)
    if tensors.keys() != mapping.keys():
        raise ValueError("Incomplete weight index")
    output = {"__metadata__": metadata}
    cursor = 0
    for key, (_, _, info) in sorted(tensors.items()):
        length = info["data_offsets"][1] - info["data_offsets"][0]
        output[key] = {"dtype": info["dtype"], "shape": info["shape"],
                       "data_offsets": [cursor, cursor + length]}
        cursor += length
    raw = json.dumps(output, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    raw += b" " * (-len(raw) % 8)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    temporary = target.with_name(target.name + ".partial")
    with temporary.open("wb") as writer:
        writer.write(struct.pack("<Q", len(raw)))
        writer.write(raw)
        for key, (path, start, info) in sorted(tensors.items()):
            left, right = info["data_offsets"]
            with path.open("rb") as reader:
                reader.seek(start + left)
                remaining = right - left
                while remaining:
                    check_cancelled(cancel_event)
                    block = reader.read(min(CHUNK, remaining))
                    if not block:
                        raise ValueError(f"Truncated tensor data: {key}")
                    writer.write(block)
                    remaining -= len(block)
        writer.flush()
        os.fsync(writer.fileno())
    verified, _ = tensor_header(temporary)
    if verified.keys() != mapping.keys():
        raise ValueError("Export verification failed")
    result = sha256(temporary, cancel_event)
    if before_publish is not None:
        before_publish(result, temporary.stat().st_size)
    check_cancelled(cancel_event)
    temporary.replace(target)
    return result


def download_variant(variant, cache, components, workers, cancel_event=None):
    from huggingface_hub import HfApi
    cancel_event = cancel_event if cancel_event is not None else Event()
    check_cancelled(cancel_event)
    repo = f"OpenMOSS-Team/MOSS-Image-{variant.upper() if variant == 'sft' else 'Base'}"
    revision = REVISIONS[variant]
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    files = [f for f in info.siblings if any(f.rfilename.startswith(c + "/") for c in components)
             and not f.rfilename.endswith(".py")]
    destination = cache / variant
    destination.mkdir(parents=True, exist_ok=True)
    def fetch(item):
        check_cancelled(cancel_event)
        expected = item.lfs.sha256 if item.lfs else None
        blob_id = item.blob_id if not item.lfs else None
        url = f"https://huggingface.co/{repo}/resolve/{revision}/{item.rfilename}"
        path = child_path(destination, item.rfilename)
        path.parent.mkdir(parents=True, exist_ok=True)
        digest = verify_official_file(path, item.size, expected, blob_id, cancel_event)
        if digest is None:
            partial = Path(str(path) + ".partial")
            # Adopt the largest incomplete Hub HTTP download, which is a contiguous
            # prefix. Xet files may be sparse/out-of-order, so final SHA is mandatory.
            old_cache = destination / ".cache/huggingface/download" / Path(item.rfilename).parent
            candidates = list(old_cache.glob(f"*.{expected}.*.incomplete")) if expected else []
            if not partial.exists() and candidates:
                candidate = max(candidates, key=lambda p: p.stat().st_size)
                if 0 < candidate.stat().st_size <= item.size:
                    with candidate.open("rb") as reader, partial.open("wb") as writer:
                        for block in iter(lambda: reader.read(CHUNK), b""):
                            check_cancelled(cancel_event)
                            writer.write(block)
            for checksum_attempt in range(2):
                range_download(url, partial, item.size, cancel_event)
                digest = verify_official_file(partial, item.size, expected, blob_id, cancel_event)
                if digest is not None:
                    check_cancelled(cancel_event)
                    partial.replace(path)
                    break
                if checksum_attempt:
                    raise ValueError(f"Official SHA256 mismatch after re-download: {item.rfilename}")
                # A corrupted download cache must be replaced, rather than reused.
                with partial.open("wb"):
                    pass
        print(f"Verified {variant}/{item.rfilename} ({path.stat().st_size:,} bytes)", flush=True)
        return item.rfilename, {"sha256": digest, "size": path.stat().st_size,
                                "official_lfs_sha256": expected, "official_git_blob_id": blob_id}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, item) for item in files]
        wait_for_jobs(futures, cancel_event)
        records = dict(future.result() for future in futures)
    record = {"repo": repo, "revision": revision, "files": records}
    # Component-specific manifests let shared components and the two DiTs prepare
    # independently without racing over one repository-wide verification file.
    for component in set(components) & {"transformer", "text_encoder", "vae"}:
        check_cancelled(cancel_event)
        atomic_json(destination / f"verified-{component}.json", record)
    return destination


def range_download(url, partial, size, cancel_event=None):
    """Resume bounded HTTP ranges with exact Content-Range and final SHA validation."""
    import requests
    partial = Path(partial)
    check_cancelled(cancel_event)
    partial.touch(exist_ok=True)
    if partial.stat().st_size > size:
        raise ValueError(f"Oversized download partial: {partial}")
    session = requests.Session()
    last_report = time.monotonic()
    try:
        while partial.stat().st_size < size:
            check_cancelled(cancel_event)
            start = partial.stat().st_size
            end = min(size - 1, start + 64 * 1024**2 - 1)
            for attempt in range(6):
                check_cancelled(cancel_event)
                try:
                    # A query bound to the range avoids intermediaries replaying a
                    # cached response for a different Range request.
                    ranged_url = url + f"?download=true&range={start}-{end}"
                    with session.get(ranged_url, headers={"Range": f"bytes={start}-{end}",
                                                          "Accept-Encoding": "identity"},
                                     stream=True, timeout=(20, 45)) as response:
                        response.raise_for_status()
                        expected_range = f"bytes {start}-{end}/{size}"
                        full_response = response.status_code == 200 and start == 0 and end == size - 1
                        if not full_response and (response.status_code != 206 or response.headers.get("Content-Range") != expected_range):
                            raise ValueError(f"Server did not honor exact download range: {partial.name}")
                        received = 0
                        with partial.open("ab") as writer:
                            for block in response.iter_content(CHUNK):
                                check_cancelled(cancel_event)
                                if received + len(block) > end - start + 1:
                                    raise ValueError("Server returned excess range data")
                                writer.write(block)
                                received += len(block)
                        if received != end - start + 1:
                            raise IOError("Incomplete HTTP range")
                    break
                except (requests.RequestException, OSError) as exc:
                    if attempt == 5:
                        raise
                    print(f"Resume {partial.name}: {type(exc).__name__}", flush=True)
                    delay = min(2**attempt, 10)
                    if cancel_event is None:
                        time.sleep(delay)
                    else:
                        cancel_event.wait(delay)
                        check_cancelled(cancel_event)
                    start = partial.stat().st_size
                    if start > end:
                        break
            if time.monotonic() - last_report >= 20:
                print(f"Downloading {partial.name}: {partial.stat().st_size / size:.1%}", flush=True)
                last_report = time.monotonic()
    finally:
        session.close()


def export_component(source, component, target, cancel_event=None):
    check_cancelled(cancel_event)
    assets = Path(str(target) + ".assets")
    assets.mkdir(parents=True, exist_ok=True)
    manifest_path = assets / "manifest.json"
    verification_path = source / f"verified-{component}.json"
    verified = read_json(verification_path if verification_path.exists() else source / "verified.json")
    config = read_json(source / component / "config.json")
    tokenizer_files = {}
    if component == "text_encoder":
        tokenizer = assets / "tokenizer"
        tokenizer.mkdir(exist_ok=True)
        for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
            original = source / "tokenizer" / name
            expected = verified["files"][f"tokenizer/{name}"]["sha256"]
            if sha256(original, cancel_event) != expected:
                raise ValueError(f"Modified tokenizer source: {original}")
            if not (tokenizer / name).exists() or sha256(tokenizer / name, cancel_event) != expected:
                check_cancelled(cancel_event)
                temporary = tokenizer / (name + ".partial")
                shutil.copyfile(original, temporary)
                temporary.replace(tokenizer / name)
            tokenizer_files[name] = expected
    if target.exists():
        if manifest_path.exists():
            previous = read_json(manifest_path)
            expected = previous.get("sha256") or previous.get("pending_sha256")
            if (previous.get("component") == component and previous.get("revision") == verified["revision"]
                    and expected and expected == sha256(target, cancel_event)):
                previous.update(ready=True, sha256=expected, size=target.stat().st_size,
                                config_sha256=hashlib.sha256(json.dumps(config).encode()).hexdigest())
                previous.pop("pending_sha256", None)
                if tokenizer_files:
                    previous["tokenizer_files"] = tokenizer_files
                check_cancelled(cancel_event)
                atomic_json(manifest_path, previous)
                print(f"Already verified: {target}", flush=True)
                return
        raise FileExistsError(f"Existing file has no matching ready manifest: {target}")
    manifest = {"schema_version": 1, "ready": False, "component": component,
                "repo": verified["repo"], "revision": verified["revision"],
                "source_files": {k: v for k, v in verified["files"].items()
                                 if k.startswith(component + "/") or (component == "text_encoder" and k.startswith("tokenizer/"))}}
    atomic_json(manifest_path, manifest)
    metadata = {"format": "pt", "moss_image.format": "1", "moss_image.component": component,
                "moss_image.config": json.dumps(config),
                "moss_image.source": json.dumps({"repo": verified["repo"], "revision": verified["revision"]})}
    if component == "transformer":
        metadata["moss_image.scheduler"] = json.dumps(read_json(source / "scheduler/scheduler_config.json"))
    if tokenizer_files:
        manifest["tokenizer_files"] = tokenizer_files
    def pending(digest, size):
        manifest.update(pending_sha256=digest, size=size)
        atomic_json(manifest_path, manifest)
    if component == "vae":
        # Rewrite only the header to attach the original config; tensor bytes stay unchanged.
        original = source / "vae/diffusion_pytorch_model.safetensors"
        header, _ = tensor_header(original)
        index = source / "vae/moss-export.index.json"
        atomic_json(index, {"weight_map": {key: original.name for key in header}})
        digest = merge_component(original.parent, index.name, target, metadata,
                                 before_publish=pending, cancel_event=cancel_event)
    else:
        index = "model.safetensors.index.json" if component == "text_encoder" else "diffusion_pytorch_model.safetensors.index.json"
        digest = merge_component(source / component, index, target, metadata,
                                 before_publish=pending, cancel_event=cancel_event)
    manifest.update(ready=True, sha256=digest, size=target.stat().st_size,
                    config_sha256=hashlib.sha256(metadata["moss_image.config"].encode()).hexdigest())
    manifest.pop("pending_sha256", None)
    check_cancelled(cancel_event)
    atomic_json(manifest_path, manifest)
    print(f"Ready: {target} SHA256={digest}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(__file__).resolve().parents[1] / "_downloads")
    parser.add_argument("--variant", choices=["both", "base", "sft"], default="both")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    cancel_event = Event()
    if not 1 <= args.workers <= 8:
        parser.error("--workers must be 1..8")
    # Shared TE/VAE source is pinned Base. Both official repositories have identical
    # LFS hashes for all four TE shards and the VAE (see docs/architecture.md).
    def common_components():
        common = download_variant("base", args.cache_dir, ["text_encoder", "tokenizer", "vae"], args.workers, cancel_event)
        export_component(common, "text_encoder", args.models_dir / "text_encoders/moss-image-qwen35-bf16.safetensors", cancel_event)
        export_component(common, "vae", args.models_dir / "vae/moss-image-vae.safetensors", cancel_event)
    def dit_component(variant):
        source = download_variant(variant, args.cache_dir, ["transformer", "scheduler"], args.workers, cancel_event)
        export_component(source, "transformer", args.models_dir / f"diffusion_models/moss-image-{variant}-bf16.safetensors", cancel_event)
    variants = ["base", "sft"] if args.variant == "both" else [args.variant]
    with ThreadPoolExecutor(max_workers=3) as jobs:
        futures = [jobs.submit(common_components)] + [jobs.submit(dit_component, variant) for variant in variants]
        wait_for_jobs(futures, cancel_event)


if __name__ == "__main__":
    main()
