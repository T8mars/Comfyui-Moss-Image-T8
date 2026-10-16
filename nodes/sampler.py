class MOSSOfficialSampler:
    CATEGORY = "MOSS-Image/Official"
    FUNCTION = "sample"
    RETURN_TYPES = ("MOSS_IMAGE_LATENT",)
    RETURN_NAMES = ("latent",)
    DESCRIPTION = "Pinned official MOSSImagePipeline, nearest-training-grid Euler schedule, FP32 noise/CFG/updates. Empty layer lists disable skipping."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "dit": ("MOSS_IMAGE_DIT",), "conditioning": ("MOSS_IMAGE_CONDITIONING",),
            "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
            "steps": ("INT", {"default": 50, "min": 1, "max": 1000}),
            "cfg": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 30.0, "step": 0.1}),
            "shift": ("FLOAT", {"default": 5.0, "min": 0.01, "max": 100.0, "step": 0.1}),
            "width": ("INT", {"default": 1024, "min": 16, "max": 4096, "step": 16}),
            "height": ("INT", {"default": 1024, "min": 16, "max": 4096, "step": 16}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 16}),
            "uncond_skip_layers": ("STRING", {"default": "22"}),
            "slg_layers": ("STRING", {"default": ""}),
            "slg_scale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0}),
            "slg_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
            "slg_stop": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
        }}

    def sample(self, dit, conditioning, **kwargs):
        from comfy.utils import ProgressBar
        from ..runtime import sample
        bar = ProgressBar(kwargs.get("steps", 50))
        return (sample(dit, conditioning, progress=lambda done, total: bar.update_absolute(done, total), **kwargs),)
