class MOSSOfficialVAEDecode:
    CATEGORY = "MOSS-Image/Official"
    FUNCTION = "decode"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    DESCRIPTION = "Official packed-latent BN inverse, 2x2 unpack, FP32 VAE. Tiling is optional and may change pixels."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"vae": ("MOSS_IMAGE_VAE",), "latent": ("MOSS_IMAGE_LATENT",),
                             "tiled": ("BOOLEAN", {"default": False})}}

    def decode(self, vae, latent, tiled=False):
        from ..runtime import decode
        return (decode(vae, latent, tiled),)
