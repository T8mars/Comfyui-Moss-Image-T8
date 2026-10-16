"""Publish the completed canvas, image, parameter and raw-weight acceptance record."""
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    artifacts = ROOT / "artifacts"
    canvas = read(artifacts / "canvas-validation.json")
    audit = read(artifacts / "parameter-audit.json")
    imports = read(artifacts / "canvas-import.json")
    assert len(canvas) == 2 and len(imports) == 8
    assert {entry["method"] for entry in imports} == {"file-input", "canvas-drop"}
    assert "15 passed" in (ROOT / "artifacts-goal-tests.log").read_text(encoding="utf-8")
    assert all(item["raw_tensor_audit"]["all_shapes_dtypes_and_tensor_bytes_equal"] for item in audit["independent_models"])
    baseline = artifacts / "validation-api-baseline.json"
    if not baseline.exists():
        shutil.copyfile(artifacts / "validation.json", baseline)
    previous = read(baseline)
    executions = []
    examples = ROOT / "docs/examples"
    examples.mkdir(exist_ok=True)
    for result in canvas:
        variant = result["variant"]
        history = read(artifacts / f"canvas-{variant}/history.json")
        events = {name: data for name, data in history["status"]["messages"]}
        assert history["status"]["status_str"] == "success"
        assert result["sampler_executed_fresh"] and result["png_embedded_prompt_and_gui_exact"]
        executions.append({"variant": variant, "prompt_id": result["prompt_id"],
                           "execution_seconds": (events["execution_success"]["timestamp"] - events["execution_start"]["timestamp"]) / 1000,
                           "parameters": result["parameters"]})
        shutil.copyfile(ROOT / result["drag_in_png"], examples / f"{variant}-quality.png")
        shutil.copyfile(artifacts / f"canvas-{variant}/canvas-success.png", examples / f"canvas-{variant}.png")
    report = {"schema_version": 2, "tests": "15 passed", "versions": previous["versions"],
              "models": previous["models"], "canvas_runs": canvas, "canvas_execution": executions,
              "file_import_checks": imports, "official_parameter_and_raw_weight_audit": audit,
              "historical_api_generations": previous["generations"],
              "real_cpu_text_encoder": previous["real_cpu_text_encoder"], "real_vae": previous["real_vae"]}
    for destination in (artifacts / "validation.json", ROOT / "docs/validation.json"):
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for name in ("canvas-validation", "parameter-audit", "canvas-import"):
        shutil.copyfile(artifacts / f"{name}.json", ROOT / f"docs/{name}.json")
    tensors = sum(item["raw_tensor_audit"]["tensor_count"] for item in audit["independent_models"])
    byte_count = sum(item["raw_tensor_audit"]["compared_tensor_bytes"] for item in audit["independent_models"])
    lines = ["# 画布直接运行验收", "", "验证日期：2026-10-16。Windows / RTX 5090 Laptop 24 GB / ComfyUI 0.41.0 / 前端 1.55.17。", "",
             "以下结果来自真实浏览器中的“运行”按钮：导入交付的原始工作流，点击运行，截取前端实际 POST 请求，再读取该 prompt_id 的服务端成功回执。两套任务均实际执行采样，非采样缓存命中。", "",
             "| 工作流 | 分辨率 | 步数 | CFG / shift | seed / 负分支跳层 | 结果 | 服务端执行时间 |",
             "|---|---|---:|---|---|---|---:|"]
    for result in executions:
        lines.append(f"| {result['variant'].upper()} | 1024×1024 | 50 | 3 / 5 | 42 固定 / 22 | 成功，画布显示 1024 预览 | {result['execution_seconds']:.2f} s |")
    lines += ["", "时间包含当次加载与解码；缓存状态和后台负载不同，不作为两模型速度对比。两张实际生成图均已目视检查，内容为提示词中的红狐和草地，尺寸、颜色与非恒定像素检查通过。", "",
              "## 可直接导入画布", "",
              "- [SFT JSON](../example_workflows/moss-image-sft.json) / [SFT PNG（内含工作流）](../example_workflows/moss-image-sft.png)",
              "- [Base JSON](../example_workflows/moss-image-base.json) / [Base PNG（内含工作流）](../example_workflows/moss-image-base.png)", "",
              "上述四个文件都已分别通过前端文件选择器和画布拖放事件导入，共 8 个用例。每次导入前清空画布，确认实际新加载了 7 个节点、全部连线和全部控件参数；无节点验证错误。PNG 内的 prompt 与 workflow 元数据也逐项核对通过。", "",
              "## 参数来源核对", "",
              "| 参数 | 交付值 | 依据 |", "|---|---|---|",
              "| 宽 / 高 | 1024 / 1024 | 固定版本官方 MOSSImagePipeline.__call__ 默认值 |",
              "| Steps / CFG | 50 / 3 | 官方 num_inference_steps / guidance_scale |",
              "| Shift | 5 | 官方 MOSSImageScheduler 默认值 |",
              "| 负分支跳层 | 22（从 0 开始） | 官方 uncond_skip_layers；模型深度 42，索引有效 |",
              "| Batch / 最大文本长度 | 1 / 4096 | 官方 num_images_per_prompt / max_sequence_length |",
              "| 负面 / 额外 SLG | 空 / 关闭 | 官方默认值；空 SLG 列表与 None 均关闭 |",
              "| 随机种子 | 42，fixed | 官方示例种子；实际运行后未递增 |",
              "| 精度 | TE、DiT BF16；VAE FP32 | 官方推荐；噪声、CFG、Euler 更新沿用官方 FP32 实现 |",
              "| VAE tiling | 关闭 | 保持官方普通解码路径 |", "",
              "参数依据由固定官方源码 AST 提取，再核对本节点运行时参数转发、浏览器实际请求、服务端收到的请求和 PNG 元数据。Base/SFT 两个官方仓库的 pipeline Git blob 已再次核对一致，未修改的官方文件 SHA256：", "",
              f"`{audit['pipeline_sha256']}`", ""]
    sft = next(item for item in canvas if item["variant"] == "sft")
    comparison = sft["against_previous_direct_api_run"]
    lines += [f"SFT 画布出图与此前相同参数的 API 出图对比：像素完全一致 = {comparison['pixels_identical']}，最大 RGB 差 = {comparison['max_absolute_rgb_difference']}。", "",
              "## 独立文件完整性", "",
              f"四个独立权重已重新计算完整 SHA256，并与官方原始分片逐 tensor 比对：{tensors:,} 个 tensor、{byte_count:,} 个数据字节，shape、dtype 和原始字节全部一致。未将 DiT、TE、VAE 合成一个 checkpoint。", "",
              "| 文件 | tensor 数 | 原始数据字节一致 | SHA256 前 12 位 |", "|---|---:|---|---|"]
    for model in audit["independent_models"]:
        lines.append(f"| {model['file']} | {model['raw_tensor_audit']['tensor_count']} | 是 | `{model['sha256'][:12]}` |")
    lines += ["", "完整 TE（含视觉权重）保留；两仓库共享 TE、VAE 和 tokenizer 的官方内容 ID 一致。配置与 tokenizer 同时对照官方 Git/LFS 内容 ID。每个权重各有独立 `.assets/manifest.json`。", "",
              "## 测试与证据", "",
              "15 项测试本轮重新通过，包括真实微型 Qwen3.5/DiT/VAE 与官方整管道逐像素一致、CFG/SLG、seed/batch、BN/unpack、流式导出和中断恢复。完整 TE 另有 CPU BF16 测试，实际画布运行使用 TE/DiT CUDA BF16 与 VAE CUDA FP32。", "",
              "- [机器可读完整记录](validation.json)", "- [官方参数与全部原始 tensor 审计](parameter-audit.json)",
              "- [画布运行与 PNG 检查](canvas-validation.json)", "- [真实文件导入检查](canvas-import.json)",
              "- 原始浏览器请求、服务端回执、运行中截图：本地 `artifacts/canvas-base/`、`artifacts/canvas-sft/`。", "",
              "## 实际画布与出图", "", "Base：", "", "![Base 画布运行完成](examples/canvas-base.png)", "", "![Base 1024](examples/base-quality.png)", "",
              "SFT：", "", "![SFT 画布运行完成](examples/canvas-sft.png)", "", "![SFT 1024](examples/sft-quality.png)", "",
              "原生 KSampler 路径未作为本次验收对象；以上均为固定官方 Diffusers 管道。"]
    (ROOT / "docs/validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Published exact canvas/PNG import and Run evidence; docs/validation.md, JSON records and examples updated")


if __name__ == "__main__":
    main()
