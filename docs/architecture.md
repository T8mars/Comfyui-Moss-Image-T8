# 10 轮子 Agent 讨论记录

用户要求先讨论 10 轮再实现；主 Agent 与 architecture_review 逐轮问答，全部完成后开始开发。

1. **官方架构**：MOSSImageTransformer（DiT）+ Qwen3_5Model + Qwen tokenizer + AutoencoderKLFlux2 + MOSSImageScheduler。Base/SFT 同架构，不把 DiT 错当传统 UNet。
2. **官方管道边界**：直接复用固定版本官方代码；原生 Comfy KSampler 不包含官方 training-grid snapping，不能声称同 seed 完全等价。保留完整 TE，含视觉权重。
3. **分阶段生命周期**：独立文件描述符；TE 编码后释放，DiT 采样后释放，最后加载 VAE。使用自定义 MOSS_IMAGE_LATENT 端口，避免误接普通 VAE Decode。
4. **低峰值加载**：init_empty_weights(include_buffers=False) + strict assign；避免随机模型与权重双份内存；保护 Qwen 非持久 FP32 RoPE buffer。
5. **独立权重导出**：只合并同组件分片，原字节流式复制；索引、路径、offset、dtype、shape、官方 SHA 全部校验。TE、VAE 可共享必须有哈希证据。
6. **验证**：真实微型 Qwen/DiT/VAE 与官方整管道对照；覆盖 CFG、negative、skip22、SLG、BN/unpack；随后真实模型与 Comfy 队列出图。
7. **文件和交互**：标准模型目录，配置嵌入 safetensors metadata；每个权重使用 `<filename>.assets/manifest.json`，TE 另含 tokenizer；默认官方参数。
8. **执行可靠性**：进程内 RLock，CPU 中间结果，异常和取消时 finally 清理；逐步进度与 TE block 中断检查；模型不常驻节点缓存。
9. **失败行为**：组件/配置不符拒绝；缺文件给准备命令；不静默降精度；节点仅本地加载；下载缓存损坏可重试，完整文件不覆盖未知内容。
10. **最终验收**：六节点各自独立源码、四个独立权重、Base/SFT 两种工作流、固定官方来源与许可、可追溯出图记录。只报告实际完成的测试。

## 固定来源

- [Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base)：`5f40f07a304541840b6019d2d34d30f966756787`
- [SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT)：`daf706d2e1c7a6369c17b6d1658671aee6db1ebe`
- [官方 ComfyUI 插件](https://github.com/OpenMOSS/MOSS-Image/tree/1f808b7c4cfd9b72e7d81bbda9b067bc4d306114/comfyui)：作为架构参考和原生生态兼容入口，不重写其模型实现。
- vendor/pipeline_moss_image.py SHA256：`1dac1aca9feb1bfdfc727a401e4b67d61a505d3190706d1b7d2d784f88678b31`，文件原文未修改。
  已核对 Base 与 SFT 上述固定 revision 的 pipeline 文件逐字节一致。

## 共享组件依据

两仓库相同 revision 下，四个 TE 分片 LFS SHA256 全部一致：

```
abfaa3d3a8612508c85ee8f6da9c734b862f10298b903765c4ad384177c32b0b
9112637fa6bf4d07bafd6ae268852b18d3d0c1242f4ee901c5579006e5bd4894
956f1ef51d0cfe610b0d22987c0b7821c36e3ce228abceccef523518bcf079dc
9e21d20e8adde88c3e0b65d4a0d251f143d98ee9d52db36b4348361091e4e208
```

VAE LFS SHA256 两仓相同：`c3bdf94dc67ec69b30ea9fe1fa8340085679cae64d019f7c7e4a9921f1a7f17a`。
TE/VAE config、tokenizer 文件也一致。下载原始权重约 37.4 GB，导出后四个独立文件约 37.4 GB；保留下载缓存时总计约 74.8 GB。

VAE 导出仅重写 header 以嵌入配置，所有 tensor 的 dtype、shape 和数据字节保持不变；整个文件 SHA 因 header 变化而不同。TE 和 DiT 同理，不量化、不裁剪、不合并不同组件。
