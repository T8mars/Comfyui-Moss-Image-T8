# MOSS-Image for ComfyUI

[English](README.md) | **简体中文**

为 [ComfyUI MOSS-Image T8](https://github.com/T8mars/Comfyui-Moss-Image-T8) 提供独立的 **DiT（UNET）、文本编码器和 VAE** 文件，支持 MOSS-Image **Base / SFT** 官方推理管道。本仓库由社区重新封装，保留原始 tensor 名称、形状、精度与数据，配置写入 safetensors metadata。

## 文件

| 组件 | 文件 | 大小 |
|---|---|---:|
| Base DiT，BF16 | [diffusion_models/moss-image-base-bf16.safetensors](diffusion_models/moss-image-base-bf16.safetensors) | 10.16 GB |
| SFT DiT，BF16 | [diffusion_models/moss-image-sft-bf16.safetensors](diffusion_models/moss-image-sft-bf16.safetensors) | 10.16 GB |
| 共用完整 Qwen3.5 TE，BF16 | [text_encoders/moss-image-qwen35-bf16.safetensors](text_encoders/moss-image-qwen35-bf16.safetensors) | 16.79 GB |
| 共用 FLUX.2 VAE，FP32 | [vae/moss-image-vae.safetensors](vae/moss-image-vae.safetensors) | 0.25 GB |

**下载权重时，必须同时下载对应的 `.assets` 目录**，其中包含完整性清单及 TE tokenizer。Base / SFT 共用 TE、VAE，两版合计约 37.4 GB。DiT、TE、VAE 始终分开存放。

`tokenizer/` 目录只保留交付的三个文件；额外文件和子目录会被加载器拒绝。

## 快速开始

1. 在 ComfyUI Manager 安装 [ComfyUI MOSS-Image T8](https://github.com/T8mars/Comfyui-Moss-Image-T8)（发布者 `t8star`，包名 `moss-image-t8`），或按 GitHub 说明手动安装。
2. 在 ComfyUI 根目录下载，保留目录结构：

```bash
hf download t8star/Moss-Image-Comfy --local-dir models --include "diffusion_models/*" --include "text_encoders/*" --include "vae/*"
```

3. 重启 ComfyUI，将 [SFT 工作流 PNG](example_workflows/moss-image-sft.png) 或 [Base 工作流 PNG](example_workflows/moss-image-base.png) 拖入画布。也可使用 [JSON 工作流](https://github.com/T8mars/Comfyui-Moss-Image-T8/tree/main/example_workflows)。

只需要 SFT 时，将 `"diffusion_models/*"` 换成 `"diffusion_models/moss-image-sft-bf16.safetensors*"`。

默认参数：**1024×1024、50 步、CFG 3、shift 5、seed 42、空负面提示词、负分支跳过第 22 层（从 0 开始）**。TE / DiT 使用 BF16，VAE 解码使用 FP32。两版均在 RTX 5090 Laptop 24 GB 显卡上通过真实画布运行，详见[实测记录](https://github.com/T8mars/Comfyui-Moss-Image-T8/blob/main/docs/validation.md)。

![SFT 实际生成图，内嵌完整工作流](example_workflows/moss-image-sft.png)

## 来源与校验

- [OpenMOSS-Team/MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base/tree/5f40f07a304541840b6019d2d34d30f966756787)：`5f40f07a304541840b6019d2d34d30f966756787`。
- [OpenMOSS-Team/MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT/tree/daf706d2e1c7a6369c17b6d1658671aee6db1ebe)：`daf706d2e1c7a6369c17b6d1658671aee6db1ebe`。

全部 **1,936 个 tensor**（包含完整 TE 视觉权重）已与原始分片逐字节比对一致；配置及 tokenizer 对照固定官方内容 ID 校验。每个 `.assets/manifest.json` 记录权重完整 SHA256 及来源，[SHA256SUMS](SHA256SUMS) 覆盖发布的权重和配套文件。

模型与管道由 [OpenMOSS](https://github.com/OpenMOSS/MOSS-Image) 提供，T8star 维护 ComfyUI 集成与文件封装。采用 [Apache-2.0](LICENSE)，归属见 [NOTICE](NOTICE)。请使用上述插件的独立加载节点及官方管道采样器运行。

## 相关链接

[B站](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/) · [API](https://api.seedance.nz/sign-up?aff=5f4w) · [免费画廊](https://www.openzhenzhen.com) · [在线 AI 应用](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) · [ComfyUI 整合包](https://pan.quark.cn/s/264edb7e36bd) · [Hugging Face](https://huggingface.co/t8star)
