from .common import Loader


class MOSSOfficialDiffusionLoader(Loader):
    folder = "diffusion_models"
    kind = "transformer"
    RETURN_TYPES = ("MOSS_IMAGE_DIT",)
    RETURN_NAMES = ("dit",)
    DESCRIPTION = "Load the independent Base or SFT DiT file. Used by the official MOSS sampler."
