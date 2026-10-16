"""ComfyUI entry point. Model libraries are imported only during execution."""
from .nodes.diffusion_loader import MOSSOfficialDiffusionLoader
from .nodes.text_encoder_loader import MOSSOfficialTextEncoderLoader
from .nodes.vae_loader import MOSSOfficialVAELoader
from .nodes.encode import MOSSOfficialTextEncode
from .nodes.sampler import MOSSOfficialSampler
from .nodes.decode import MOSSOfficialVAEDecode

NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (
    MOSSOfficialDiffusionLoader, MOSSOfficialTextEncoderLoader, MOSSOfficialVAELoader,
    MOSSOfficialTextEncode, MOSSOfficialSampler, MOSSOfficialVAEDecode,
)}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MOSSOfficialDiffusionLoader": "MOSS-Image Official UNET / DiT Loader",
    "MOSSOfficialTextEncoderLoader": "MOSS-Image Official TE Loader",
    "MOSSOfficialVAELoader": "MOSS-Image Official VAE Loader",
    "MOSSOfficialTextEncode": "MOSS-Image Official Text Encode",
    "MOSSOfficialSampler": "MOSS-Image Official Sampler",
    "MOSSOfficialVAEDecode": "MOSS-Image Official VAE Decode",
}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
