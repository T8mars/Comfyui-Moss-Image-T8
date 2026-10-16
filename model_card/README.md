---
license: apache-2.0
language:
  - en
  - zh
pipeline_tag: text-to-image
base_model:
  - OpenMOSS-Team/MOSS-Image-Base
  - OpenMOSS-Team/MOSS-Image-SFT
tags:
  - comfyui
  - moss-image
  - safetensors
  - text-to-image
---

# MOSS-Image for ComfyUI

**English** | [简体中文](README_ZH.md)

Independent **DiT (UNET), text encoder and VAE** files for [ComfyUI MOSS-Image T8](https://github.com/T8mars/Comfyui-Moss-Image-T8). Supports MOSS-Image **Base and SFT** with the original OpenMOSS inference pipeline. This is a community repackaging: tensor names, shapes, dtypes and raw data are preserved; configuration is embedded in safetensors metadata.

## Files

| Component | File | Size |
|---|---|---:|
| Base DiT, BF16 | [diffusion_models/moss-image-base-bf16.safetensors](diffusion_models/moss-image-base-bf16.safetensors) | 10.16 GB |
| SFT DiT, BF16 | [diffusion_models/moss-image-sft-bf16.safetensors](diffusion_models/moss-image-sft-bf16.safetensors) | 10.16 GB |
| Shared full Qwen3.5 TE, BF16 | [text_encoders/moss-image-qwen35-bf16.safetensors](text_encoders/moss-image-qwen35-bf16.safetensors) | 16.79 GB |
| Shared FLUX.2 VAE, FP32 | [vae/moss-image-vae.safetensors](vae/moss-image-vae.safetensors) | 0.25 GB |

Download each selected weight **together with its matching `.assets` directory**. These contain integrity manifests and the TE tokenizer. Base and SFT use the same TE and VAE; both variants together total about 37.4 GB. Components are never merged into one checkpoint.

Keep `tokenizer/` limited to its three shipped files. Extra files and subdirectories are rejected by the loaders.

## Quick start

1. Install [ComfyUI MOSS-Image T8](https://github.com/T8mars/Comfyui-Moss-Image-T8) through ComfyUI Manager (publisher `t8star`, package `moss-image-t8`) or follow the manual installation instructions.
2. From the ComfyUI directory, download models into `models/` with the directory structure preserved:

```bash
hf download t8star/Moss-Image-Comfy --local-dir models --include "diffusion_models/*" --include "text_encoders/*" --include "vae/*"
```

3. Restart ComfyUI and drag a [SFT workflow PNG](example_workflows/moss-image-sft.png) or [Base workflow PNG](example_workflows/moss-image-base.png) onto the canvas. [JSON workflows](https://github.com/T8mars/Comfyui-Moss-Image-T8/tree/main/example_workflows) are also available.

For SFT only, replace `"diffusion_models/*"` with `"diffusion_models/moss-image-sft-bf16.safetensors*"`.

Defaults: **1024×1024, 50 steps, CFG 3, shift 5, seed 42, empty negative prompt, unconditional skip layer 22 (zero-based)**. TE/DiT use BF16 and VAE decoding uses FP32. Both variants passed actual ComfyUI canvas generation at these settings on a 24 GB RTX 5090 Laptop GPU. See [validation](https://github.com/T8mars/Comfyui-Moss-Image-T8/blob/main/docs/validation.md).

![SFT generated example, with embedded workflow](example_workflows/moss-image-sft.png)

## Provenance and integrity

- [OpenMOSS-Team/MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base/tree/5f40f07a304541840b6019d2d34d30f966756787), revision `5f40f07a304541840b6019d2d34d30f966756787`.
- [OpenMOSS-Team/MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT/tree/daf706d2e1c7a6369c17b6d1658671aee6db1ebe), revision `daf706d2e1c7a6369c17b6d1658671aee6db1ebe`.

All **1,936 tensors** were compared byte for byte with the source shards, including the complete TE vision weights. Configurations and tokenizer files were checked against the pinned upstream content IDs. Each `.assets/manifest.json` records the full weight SHA256 and provenance; [SHA256SUMS](SHA256SUMS) covers all distributed weights and assets.

Models and pipeline by [OpenMOSS](https://github.com/OpenMOSS/MOSS-Image); ComfyUI integration and repackaging by T8star. [Apache-2.0](LICENSE); see [NOTICE](NOTICE). This repository contains component files for the linked integration; use its dedicated loaders and official-pipeline sampler.

## Links

[Bilibili](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/) · [API](https://api.seedance.nz/sign-up?aff=5f4w) · [Free gallery](https://www.openzhenzhen.com) · [Online AI apps](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) · [ComfyUI bundle](https://pan.quark.cn/s/264edb7e36bd) · [Hugging Face](https://huggingface.co/t8star)
