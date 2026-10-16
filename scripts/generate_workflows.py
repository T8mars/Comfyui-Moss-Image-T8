"""Generate matching ComfyUI GUI and API workflows for both independent DiTs."""
import json
from pathlib import Path


def workflows(variant):
    api = {
        "1": {"class_type": "MOSSOfficialDiffusionLoader", "inputs": {
            "model_name": f"moss-image-{variant}-bf16.safetensors", "device": "auto", "precision": "bf16"}},
        "2": {"class_type": "MOSSOfficialTextEncoderLoader", "inputs": {
            "model_name": "moss-image-qwen35-bf16.safetensors", "device": "auto"}},
        "3": {"class_type": "MOSSOfficialVAELoader", "inputs": {
            "model_name": "moss-image-vae.safetensors", "device": "auto"}},
        "4": {"class_type": "MOSSOfficialTextEncode", "inputs": {
            "text_encoder": ["2", 0], "positive": "A small red fox sitting in a green meadow, soft morning light, detailed fur, photography.",
            "negative": "", "max_sequence_length": 4096}},
        "5": {"class_type": "MOSSOfficialSampler", "inputs": {
            "dit": ["1", 0], "conditioning": ["4", 0], "seed": 42, "steps": 50, "cfg": 3.0, "shift": 5.0,
            "width": 1024, "height": 1024, "batch_size": 1, "uncond_skip_layers": "22", "slg_layers": "",
            "slg_scale": 0.0, "slg_start": 0.0, "slg_stop": 1.0}},
        "6": {"class_type": "MOSSOfficialVAEDecode", "inputs": {"vae": ["3", 0], "latent": ["5", 0], "tiled": False}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": f"MOSS-Image-{variant.upper()}"}},
    }
    output_types = {"1": "MOSS_IMAGE_DIT", "2": "MOSS_IMAGE_TE", "3": "MOSS_IMAGE_VAE",
                    "4": "MOSS_IMAGE_CONDITIONING", "5": "MOSS_IMAGE_LATENT", "6": "IMAGE"}
    output_names = {"1": "dit", "2": "text_encoder", "3": "vae", "4": "conditioning", "5": "latent", "6": "images"}
    titles = {"1": f"{variant.upper()} · 独立 UNET / DiT", "2": "共享 · Qwen3.5 TE",
              "3": "共享 · FP32 VAE", "4": "正负提示词 · 官方编码",
              "5": f"{variant.upper()} · 官方采样", "6": "官方 VAE 解码", "7": f"{variant.upper()} · 保存图像"}
    positions = {"1": [30, 30], "2": [30, 250], "3": [30, 480], "4": [440, 260],
                 "5": [930, 30], "6": [1360, 180], "7": [1740, 180]}
    sizes = {"1": [360, 130], "2": [360, 110], "3": [360, 110], "4": [440, 320],
             "5": [360, 460], "6": [320, 110], "7": [420, 440]}
    nodes, links = [], []
    for key, value in api.items():
        inputs, widgets = [], []
        for name, item in value["inputs"].items():
            if isinstance(item, list):
                source, slot = item
                link_id = len(links) + 1
                input_slot = len(inputs)
                links.append([link_id, int(source), slot, int(key), input_slot, output_types[source]])
                inputs.append({"name": name, "type": output_types[source], "link": link_id})
            else:
                widgets.append(item)
                if name == "seed":
                    widgets.append("fixed")
        outputs = []
        if key in output_types:
            outputs = [{"name": output_names[key], "type": output_types[key], "links": [], "slot_index": 0}]
        nodes.append({"id": int(key), "type": value["class_type"], "pos": positions[key], "size": sizes[key],
                      "flags": {}, "order": int(key) - 1, "mode": 0, "inputs": inputs, "outputs": outputs,
                      "title": titles[key],
                      "properties": {"Node name for S&R": value["class_type"]}, "widgets_values": widgets})
    for link in links:
        nodes[link[1] - 1]["outputs"][link[2]]["links"].append(link[0])
    gui = {"last_node_id": 7, "last_link_id": len(links), "nodes": nodes, "links": links,
           "groups": [], "config": {}, "extra": {"ds": {"scale": 0.72, "offset": [150, 150]}}, "version": 0.4}
    return api, gui


def main():
    folder = Path(__file__).resolve().parents[1] / "example_workflows"
    folder.mkdir(exist_ok=True)
    for variant in ("base", "sft"):
        api, gui = workflows(variant)
        for suffix, document in (("-api", api), ("", gui)):
            (folder / f"moss-image-{variant}{suffix}.json").write_text(
                json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
