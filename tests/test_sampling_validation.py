"""Invalid GUI/API settings must fail before allocating a multi-GB DiT."""
import json

import pytest

from moss_plugin import runtime
from moss_plugin.components import Component


@pytest.mark.parametrize("settings", [
    {"cfg": float("nan")}, {"cfg": -1.0},
    {"slg_scale": float("nan")}, {"slg_scale": -0.1},
    {"slg_start": 0.9, "slg_stop": 0.1},
    {"steps": 1.5}, {"batch_size": 1.5},
])
def test_invalid_settings_fail_before_loading_dit(monkeypatch, settings):
    monkeypatch.setattr(runtime, "materialize", lambda *_: pytest.fail("Invalid settings loaded the model"))
    component = Component("unused.safetensors", "transformer", json.dumps({"num_layers": 42}))
    conditioning = dict.fromkeys(("positive", "positive_mask", "negative", "negative_mask"))
    with pytest.raises(ValueError):
        runtime.sample(component, conditioning, **settings)
