from .common import Loader


class MOSSOfficialVAELoader(Loader):
    folder = "vae"
    kind = "vae"
    RETURN_TYPES = ("MOSS_IMAGE_VAE",)
    RETURN_NAMES = ("vae",)
    DESCRIPTION = "Load the independent official FLUX.2 VAE. Decoding uses FP32."
