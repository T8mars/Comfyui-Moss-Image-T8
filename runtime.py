"""Three memory-isolated stages using the pinned official MOSS pipeline."""
from __future__ import annotations

from contextlib import contextmanager
import gc
import json
import logging
import math
import threading

from .components import Component

LOGGER = logging.getLogger(__name__)
STAGE_LOCK = threading.RLock()
LATENT_FORMAT = "moss_image_bn_packed_128_v1"


def dependencies():
    # Runtime-only imports keep ComfyUI node discovery independent of Diffusers.
    try:
        import torch
        from .vendor.pipeline_moss_image import MOSSImagePipeline, MOSSImageScheduler, MOSSImageTransformer
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError("MOSS-Image needs PyTorch >=2.9, Diffusers >=0.38, Transformers >=5.3. "
                           f"Install requirements.txt with this ComfyUI's Python. Original error: {exc}") from exc
    return torch, MOSSImagePipeline, MOSSImageScheduler, MOSSImageTransformer


def interrupted():
    try:
        from comfy.model_management import throw_exception_if_processing_interrupted
    except ImportError:
        return
    throw_exception_if_processing_interrupted()


def clean_memory():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def execution_device(component):
    import torch
    if component.device == "cpu":
        return torch.device("cpu")
    if component.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda", torch.cuda.current_device())
    if component.device != "auto":
        raise ValueError(f"Invalid device: {component.device}")
    if torch.cuda.is_available():
        from pathlib import Path
        available, _ = torch.cuda.mem_get_info()
        # Reserve activation/workspace space in addition to full component weights.
        weight_bytes = Path(component.path).stat().st_size
        if component.precision == "fp32" and component.kind != "vae":
            weight_bytes *= 2
        if available > weight_bytes + 2 * 1024**3:
            return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def materialize(component, device):
    torch, _, _, transformer_class = dependencies()
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    config = component.config
    dtype = torch.float32 if component.kind == "vae" or component.precision == "fp32" else torch.bfloat16
    with init_empty_weights(include_buffers=False):
        if component.kind == "transformer":
            model = transformer_class.from_config(config)
        elif component.kind == "text_encoder":
            from transformers import AutoConfig, AutoModel
            values = dict(config)
            model_type = values.pop("model_type")
            model = AutoModel.from_config(AutoConfig.for_model(model_type, **values), dtype=dtype)
        elif component.kind == "vae":
            from diffusers import AutoencoderKLFlux2
            model = AutoencoderKLFlux2.from_config(config)
        else:
            raise ValueError(f"Unknown component: {component.kind}")
    state = load_file(component.path, device="cpu")
    model.load_state_dict(state, strict=True, assign=True)
    del state
    # Cast parameters without rounding Qwen's nonpersistent fp32 rotary buffers.
    for parameter in model.parameters():
        if parameter.is_floating_point() and parameter.dtype != dtype:
            parameter.data = parameter.data.to(dtype=dtype)
    if any(tensor.is_meta for tensor in (*model.parameters(), *model.buffers())):
        raise RuntimeError("Component contains unmaterialized meta tensors")
    model = model.eval().requires_grad_(False).to(device=device)
    LOGGER.info("MOSS-Image: %s on %s (%s)", component.kind, device, dtype)
    return model


@contextmanager
def loaded(component, kind):
    if not isinstance(component, Component) or component.kind != kind:
        raise ValueError(f"Expected a MOSS-Image {kind} component")
    with STAGE_LOCK:
        model = None
        try:
            interrupted()
            model = materialize(component, execution_device(component))
            yield model
        finally:
            # The caller must clear its pipeline reference before leaving this scope.
            model = None
            clean_memory()


def parse_layers(value, depth):
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        result = json.loads(text)
    else:
        result = [int(item.strip()) for item in text.split(",")]
    if not isinstance(result, list) or any(type(item) is not int or not 0 <= item < depth for item in result):
        raise ValueError(f"Layer indices must be integers between 0 and {depth - 1}")
    return result


def encode(component, positive, negative="", max_sequence_length=4096):
    torch, pipeline_class, scheduler_class, _ = dependencies()
    if not 1 <= max_sequence_length <= 4096:
        raise ValueError("max_sequence_length must be between 1 and 4096")
    from transformers import AutoTokenizer
    pipe = model = None
    try:
        with loaded(component, "text_encoder") as model:
            tokenizer = AutoTokenizer.from_pretrained(component.assets / "tokenizer", local_files_only=True)
            pipe = pipeline_class(transformer=None, scheduler=scheduler_class(), text_encoder=model, tokenizer=tokenizer)
            hooks = [block.register_forward_pre_hook(lambda *_: interrupted())
                     for block in model.language_model.layers]
            try:
                with torch.inference_mode():
                    pos, pos_mask = pipe.encode_prompt(positive, max_sequence_length)
                    neg, neg_mask = pipe.encode_prompt(negative, max_sequence_length)
                interrupted()
                result = {"positive": pos.cpu(), "positive_mask": pos_mask.cpu(),
                          "negative": neg.cpu(), "negative_mask": neg_mask.cpu()}
                del pos, pos_mask, neg, neg_mask
            finally:
                for hook in hooks:
                    hook.remove()
                pipe = model = None
        return result
    finally:
        pipe = model = None
        clean_memory()


def sample(component, conditioning, seed=42, steps=50, cfg=3.0, shift=5.0,
           width=1024, height=1024, batch_size=1, uncond_skip_layers="22",
           slg_layers="", slg_scale=0.0, slg_start=0.0, slg_stop=1.0, progress=None):
    torch, pipeline_class, scheduler_class, _ = dependencies()
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0 or width % 16 or height % 16:
        raise ValueError("Width and height must be positive multiples of 16")
    if not isinstance(conditioning, dict) or not {"positive", "positive_mask", "negative", "negative_mask"} <= conditioning.keys():
        raise ValueError("Connect MOSS-Image Official Text Encode conditioning")
    if not math.isfinite(shift) or shift <= 0 or steps < 1 or batch_size < 1:
        raise ValueError("Shift, steps and batch size must be positive")
    depth = component.config["num_layers"]
    skip = parse_layers(uncond_skip_layers, depth)
    slg = parse_layers(slg_layers, depth)
    pipe = model = None
    try:
        with loaded(component, "transformer") as model:
            scheduler = scheduler_class.from_config(json.loads(component.scheduler_json), shift=shift)
            pipe = pipeline_class(transformer=model, scheduler=scheduler)
            def progress_bar(items):
                for index, item in enumerate(items):
                    interrupted()
                    yield item
                    if progress:
                        progress(index + 1, len(items))
                interrupted()
            pipe.progress_bar = progress_bar
            try:
                with torch.inference_mode():
                    output = pipe(prompt_embeds=conditioning["positive"],
                                  prompt_attention_mask=conditioning["positive_mask"],
                                  negative_prompt_embeds=conditioning["negative"],
                                  negative_prompt_attention_mask=conditioning["negative_mask"],
                                  generator=torch.Generator("cpu").manual_seed(seed),
                                  num_inference_steps=steps, guidance_scale=cfg, width=width, height=height,
                                  num_images_per_prompt=batch_size, uncond_skip_layers=skip,
                                  slg_layers=slg, slg_scale=slg_scale, slg_start=slg_start,
                                  slg_stop=slg_stop, output_type="latent").images
                result = {"samples": output.cpu(), "format": LATENT_FORMAT, "width": width, "height": height}
                del output
            finally:
                pipe = model = None
        return result
    finally:
        pipe = model = None
        clean_memory()


def decode(component, latent, tiled=False):
    torch, _, _, _ = dependencies()
    if not isinstance(latent, dict) or latent.get("format") != LATENT_FORMAT:
        raise ValueError("Connect MOSS-Image Official Sampler's MOSS_IMAGE_LATENT output")
    samples = latent["samples"]
    if samples.ndim != 4 or samples.shape[1] != 128:
        raise ValueError("Expected packed MOSS-Image latents [B,128,H/16,W/16]")
    model = None
    try:
        with loaded(component, "vae") as model:
            try:
                if tiled:
                    model.enable_tiling()
                with torch.inference_mode():
                    result = decode_tensor(model, samples)
                interrupted()
            finally:
                model = None
        return result
    finally:
        model = None
        clean_memory()


def decode_tensor(vae, latents):
    """Exact upstream BN/unpack/decode expressions, returned as Comfy IMAGE."""
    z = latents.to(device=vae.device, dtype=vae.dtype)
    bn, eps = vae.bn, vae.config.batch_norm_eps
    z = z * (bn.running_var + eps).sqrt().to(z).view(1, -1, 1, 1) + bn.running_mean.to(z).view(1, -1, 1, 1)
    b, c, h, w = z.shape
    z = z.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * h, 2 * w)
    images = vae.decode(z, return_dict=False)[0].clamp(-1, 1)
    from diffusers.image_processor import VaeImageProcessor
    images = VaeImageProcessor(vae_scale_factor=16).postprocess(images, output_type="pt")
    return images.permute(0, 2, 3, 1).contiguous().float().cpu()
