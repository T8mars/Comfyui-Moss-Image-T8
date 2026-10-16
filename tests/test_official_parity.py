"""Real tiny versions of the official modules; no mocked model forwards."""
import json
from pathlib import Path

import pytest
import torch
from diffusers import AutoencoderKLFlux2
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoModel, PreTrainedTokenizerFast, Qwen3_5Config

from moss_plugin.components import Component
from moss_plugin import runtime
from moss_plugin.vendor.pipeline_moss_image import MOSSImagePipeline, MOSSImageScheduler, MOSSImageTransformer


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    torch.set_num_threads(2)
    torch.manual_seed(123)
    root = tmp_path_factory.mktemp("tiny-models")
    dit = MOSSImageTransformer(dim=32, ffn_dim=64, num_heads=4, num_layers=24,
                               latent_channels=128, text_dim=32).eval()
    vae = AutoencoderKLFlux2(block_out_channels=(8, 8, 8, 8), decoder_block_out_channels=(8, 8, 8, 8),
                            layers_per_block=1, norm_num_groups=4, mid_block_add_attention=False).eval()
    vae.bn.running_mean.copy_(torch.linspace(-0.3, 0.4, 128))
    vae.bn.running_var.copy_(torch.linspace(0.7, 1.3, 128))
    config = Qwen3_5Config(text_config={
        "vocab_size": 32, "hidden_size": 32, "intermediate_size": 64,
        "num_hidden_layers": 4, "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 16,
        "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 8,
        "linear_value_head_dim": 8, "layer_types": ["linear_attention"] * 3 + ["full_attention"],
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5,
                            "mrope_section": [1, 1, 2]},
    }, vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 2,
                      "out_hidden_size": 32, "num_position_embeddings": 16},
        image_token_id=29, video_token_id=30, vision_start_token_id=28, vision_end_token_id=31)
    te = AutoModel.from_config(config).eval()
    backend = Tokenizer(models.WordLevel({"<unk>": 0, "<pad>": 1, "user": 2, "assistant": 3,
                                         "fox": 4, "blue": 5, "red": 6, "think": 7}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>")
    tokenizer.chat_template = "{% for message in messages %}user {{ message['content'] }} {% endfor %}assistant think"
    descriptors = {}
    for name, model in (("transformer", dit), ("text_encoder", te), ("vae", vae)):
        path = root / f"{name}.safetensors"
        save_file(model.state_dict(), path)
        model_config = dict(model.config) if name != "text_encoder" else model.config.to_dict()
        descriptors[name] = Component(str(path), name, json.dumps(model_config), "{}", "cpu", "fp32")
    tokenizer.save_pretrained(Path(descriptors["text_encoder"].path + ".assets") / "tokenizer")
    return dit, te, vae, tokenizer, descriptors


@pytest.mark.parametrize("cfg,skip,slg,scale", [(3.0, "22", "", 0.0), (1.0, "", "", 0.0),
                                              (3.0, "22", "23", 1.5), (1.0, "", "23", 2.0)])
def test_split_matches_official_pipeline(tiny, cfg, skip, slg, scale):
    dit, te, vae, tokenizer, descriptors = tiny
    pipe = MOSSImagePipeline(dit, MOSSImageScheduler(), te, tokenizer, vae)
    pipe.set_progress_bar_config(disable=True)
    with torch.inference_mode():
        expected = pipe("red fox", negative_prompt="blue", height=32, width=32, num_inference_steps=3,
                        guidance_scale=cfg, generator=torch.Generator("cpu").manual_seed(42),
                        uncond_skip_layers=runtime.parse_layers(skip, 24),
                        slg_layers=runtime.parse_layers(slg, 24), slg_scale=scale, output_type="pt").images
    conditioning = runtime.encode(descriptors["text_encoder"], "red fox", "blue", 4096)
    latent = runtime.sample(descriptors["transformer"], conditioning, steps=3, cfg=cfg, width=32, height=32,
                            uncond_skip_layers=skip, slg_layers=slg, slg_scale=scale)
    actual = runtime.decode(descriptors["vae"], latent)
    assert actual.shape == (1, 32, 32, 3)
    assert torch.isfinite(actual).all() and actual.min() >= 0 and actual.max() <= 1
    torch.testing.assert_close(actual, expected.permute(0, 2, 3, 1), atol=0, rtol=0)


def test_seed_batch_and_bad_inputs(tiny):
    _, _, _, _, desc = tiny
    conditioning = runtime.encode(desc["text_encoder"], "red fox", "", 16)
    a = runtime.sample(desc["transformer"], conditioning, steps=2, width=32, height=32, batch_size=2)
    b = runtime.sample(desc["transformer"], conditioning, steps=2, width=32, height=32, batch_size=2)
    torch.testing.assert_close(a["samples"], b["samples"], rtol=0, atol=0)
    assert a["samples"].shape == (2, 128, 2, 2)
    with pytest.raises(ValueError, match="multiples"):
        runtime.sample(desc["transformer"], conditioning, width=33)
    with pytest.raises(ValueError, match="Layer indices"):
        runtime.parse_layers("24", 24)
    with pytest.raises(ValueError, match="LATENT"):
        runtime.decode(desc["vae"], {"samples": a["samples"]})
