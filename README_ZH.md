# ComfyUI MOSS-Image T8

[English](README.md) | **简体中文**

在 ComfyUI 中运行 **MOSS-Image Base / SFT**，调用固定版本、未经修改的官方推理管道。DiT（UNET）、文本编码器和 VAE 均为独立文件、独立加载节点；Base 与 SFT 共用完整 Qwen3.5 文本编码器及 FLUX.2 VAE。

[模型下载](https://huggingface.co/t8star/Moss-Image-Comfy) · [Comfy Registry](https://registry.comfy.org/nodes/moss-image-t8) · [实测记录](docs/validation.md) · [上游项目](https://github.com/OpenMOSS/MOSS-Image)

## 安装

在 **ComfyUI Manager** 搜索 **ComfyUI MOSS-Image T8**（发布者 `t8star`，包名 `moss-image-t8`），安装后重启 ComfyUI。

手动安装：在 ComfyUI 根目录执行，并使用 ComfyUI 实际运行的 Python：

```bash
git clone https://github.com/T8mars/Comfyui-Moss-Image-T8 custom_nodes/Comfyui-Moss-Image-T8
python -m pip install -r custom_nodes/Comfyui-Moss-Image-T8/requirements.txt
```

需要 Python 3.10+、PyTorch 2.9+、Diffusers 0.38–0.41、Transformers 5.3+。依赖安装完成后重启。

## 模型

从 [t8star/Moss-Image-Comfy](https://huggingface.co/t8star/Moss-Image-Comfy/tree/main) 下载到 `ComfyUI/models`，保留目录结构。以下命令下载两个版本和共用组件：

```bash
hf download t8star/Moss-Image-Comfy --local-dir models --include "diffusion_models/*" --include "text_encoders/*" --include "vae/*"
```

`hf` 随 `huggingface_hub` 安装，请调用 ComfyUI Python 环境中的可执行文件。只需要 SFT 时，将 `"diffusion_models/*"` 换成 `"diffusion_models/moss-image-sft-bf16.safetensors*"`。

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

**每个 `.assets` 目录必须与对应权重放在一起。** Base / SFT 的 DiT 各 10.16 GB，共用 TE 16.79 GB、VAE 0.25 GB，两版合计约 37.4 GB。原始 tensor 数据与精度均保留。也可在源码仓库目录运行 `python scripts/prepare_models.py --models-dir <ComfyUI/models> --variant both`，从固定官方版本重新导出，含原始缓存需约 75 GB。

`tokenizer/` 目录只保留交付的三个文件；额外文件和子目录会被加载器拒绝。

## 工作流

将下方 JSON 或 PNG 拖入 ComfyUI 画布即可加载完整连线。PNG 为实际生成图片，内嵌完整工作流；三个加载器已预选上面的模型文件名。

| 模型 | 画布工作流 | 内嵌工作流的图片 |
|---|---|---|
| SFT | [JSON](example_workflows/moss-image-sft.json) | [PNG](example_workflows/moss-image-sft.png) |
| Base | [JSON](example_workflows/moss-image-base.json) | [PNG](example_workflows/moss-image-base.png) |

共六个独立节点：DiT、TE、VAE 加载器，以及提示词编码、官方采样、VAE 解码。采样输出 `MOSS_IMAGE_LATENT`，连接本插件解码器；解码后输出标准 ComfyUI `IMAGE`，可接保存、预览等图像节点。另附 `*-api.json` 工作流。

默认参数：**1024×1024 · 50 步 · CFG 3 · shift 5 · seed 42 · 空负面提示词**。负分支跳过第 `22` 层（从 0 开始），额外 SLG 与 VAE tiling 关闭。TE / DiT 使用 BF16，VAE 解码使用 FP32。宽高需为 16 的正整数倍。

编码、采样、解码分阶段加载与释放模型；`auto` 遵循 ComfyUI 设备设置，先协调释放缓存模型，再按可用内存选择 CUDA / CPU。完整 TE 回退 CPU 时速度会降低。已在 Windows、RTX 5090 Laptop 24 GB 显卡完成实测，该配置不代表最低显存要求。

![MOSS-Image SFT，1024×1024，50 步](docs/examples/sft-quality.png)

Base / SFT 均通过真实画布“运行”按钮完成 1024×1024、50 步出图，JSON / PNG 导入共 8 项通过；1,936 个 tensor 与官方原始分片逐字节一致。测试覆盖管道一致性、采样、文件完整性、设备选择、取消、验证工具及 Windows 启动器。详见[实测记录](docs/validation.md)、[最新审查](docs/review-20-rounds-5.md)与[架构及来源](docs/architecture.md)。

## 致谢与许可

模型及官方推理管道来自 [OpenMOSS](https://github.com/OpenMOSS/MOSS-Image)：[Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base)、[SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT)。本社区集成与模型文件封装由 T8star 维护，采用 [Apache-2.0](LICENSE)；固定来源与归属见 [NOTICE](NOTICE)。

## 相关链接

[B站](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/) · [API](https://api.seedance.nz/sign-up?aff=5f4w) · [免费画廊](https://www.openzhenzhen.com) · [在线 AI 应用](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) · [ComfyUI 整合包](https://pan.quark.cn/s/264edb7e36bd) · [Hugging Face](https://huggingface.co/t8star)
