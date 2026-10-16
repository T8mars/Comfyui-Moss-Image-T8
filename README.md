# ComfyUI MOSS-Image T8

**English** | [简体中文](README_ZH.md)

Run **MOSS-Image Base and SFT** in ComfyUI with the pinned, unmodified official inference pipeline. DiT (UNET), text encoder and VAE have separate files and loaders. Base and SFT share the full Qwen3.5 text encoder and FLUX.2 VAE.

[Models](https://huggingface.co/t8star/Moss-Image-Comfy) · [Comfy Registry](https://registry.comfy.org/nodes/moss-image-t8) · [Validation](docs/validation.md) · [Upstream](https://github.com/OpenMOSS/MOSS-Image)

## Install

In **ComfyUI Manager**, search for **ComfyUI MOSS-Image T8** (publisher `t8star`, package `moss-image-t8`), install and restart ComfyUI.

For manual installation, run from your ComfyUI directory using the Python that runs ComfyUI:

```bash
git clone https://github.com/T8mars/Comfyui-Moss-Image-T8 custom_nodes/Comfyui-Moss-Image-T8
python -m pip install -r custom_nodes/Comfyui-Moss-Image-T8/requirements.txt
```

Requires Python 3.10+, PyTorch 2.9+, Diffusers 0.38–0.41 and Transformers 5.3+. Restart after installing dependencies.

## Models

Download from [t8star/Moss-Image-Comfy](https://huggingface.co/t8star/Moss-Image-Comfy/tree/main) into `ComfyUI/models`, preserving the directory structure. This command downloads both variants and their shared components:

```bash
hf download t8star/Moss-Image-Comfy --local-dir models --include "diffusion_models/*" "text_encoders/*" "vae/*"
```

`hf` is installed by `huggingface_hub`; use its executable in your ComfyUI Python environment. To download SFT only, replace `"diffusion_models/*"` with `"diffusion_models/moss-image-sft-bf16.safetensors*"`.

```text
ComfyUI/models/
├── diffusion_models/
│   ├── moss-image-base-bf16.safetensors
│   ├── moss-image-base-bf16.safetensors.assets/manifest.json
│   ├── moss-image-sft-bf16.safetensors
│   └── moss-image-sft-bf16.safetensors.assets/manifest.json
├── text_encoders/
│   ├── moss-image-qwen35-bf16.safetensors
│   └── moss-image-qwen35-bf16.safetensors.assets/
│       ├── manifest.json
│       └── tokenizer/{tokenizer.json,tokenizer_config.json,chat_template.jinja}
└── vae/
    ├── moss-image-vae.safetensors
    └── moss-image-vae.safetensors.assets/manifest.json
```

**Keep each `.assets` directory beside its weight file.** Each DiT is 10.16 GB, the shared TE is 16.79 GB and VAE is 0.25 GB; both variants together need about 37.4 GB. Original tensor data and precision are preserved. To export directly from the pinned upstream repositories instead, use `scripts/prepare_models.py --models-dir <ComfyUI/models> --variant both` (about 75 GB including source cache).

## Workflows

Drag a JSON or PNG below onto the ComfyUI canvas. PNGs are actual generated images with the complete workflow embedded. The three model loaders are preselected for the filenames above.

| Model | Canvas workflow | Image with embedded workflow |
|---|---|---|
| SFT | [JSON](example_workflows/moss-image-sft.json) | [PNG](example_workflows/moss-image-sft.png) |
| Base | [JSON](example_workflows/moss-image-base.json) | [PNG](example_workflows/moss-image-base.png) |

Six dedicated nodes cover DiT, TE and VAE loading, text encoding, official sampling and VAE decoding. Decoded output is standard ComfyUI `IMAGE`; packed latents use `MOSS_IMAGE_LATENT` and connect to this package's decoder. API workflows are included as `*-api.json`.

Defaults: **1024×1024 · 50 steps · CFG 3 · shift 5 · seed 42 · empty negative prompt**. The unconditional branch skips layer `22` (zero-based); extra SLG and VAE tiling are off. TE/DiT use BF16; VAE decoding uses FP32. Width and height must be positive multiples of 16.

Models are released between encoding, sampling and decoding. `auto` chooses CUDA or CPU using available memory; the full TE can fall back to CPU and run more slowly. Tested on Windows with a 24 GB RTX 5090 Laptop GPU; this is a tested configuration, not a minimum VRAM claim.

![MOSS-Image SFT, 1024×1024, 50 steps](docs/examples/sft-quality.png)

Both workflows passed real canvas Run-button generation at 1024×1024 / 50 steps and all eight JSON/PNG import checks. All 1,936 tensors match the official source shards byte for byte; 15 tests cover pipeline parity, sampling and export integrity. See [validation](docs/validation.md) and [source / architecture notes](docs/architecture.md).

## Credits and license

Models and inference pipeline by [OpenMOSS](https://github.com/OpenMOSS/MOSS-Image): [Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base) and [SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT). This community integration and model repackaging are maintained by T8star. [Apache-2.0](LICENSE); upstream revisions and attribution are recorded in [NOTICE](NOTICE).

## Links

[Bilibili](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/) · [API](https://api.seedance.nz/sign-up?aff=5f4w) · [Free gallery](https://www.openzhenzhen.com) · [Online AI apps](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) · [ComfyUI bundle](https://pan.quark.cn/s/264edb7e36bd) · [Hugging Face](https://huggingface.co/t8star)
