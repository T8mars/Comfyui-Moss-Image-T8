from .common import Loader


class MOSSOfficialTextEncoderLoader(Loader):
    folder = "text_encoders"
    kind = "text_encoder"
    RETURN_TYPES = ("MOSS_IMAGE_TE",)
    RETURN_NAMES = ("text_encoder",)
    DESCRIPTION = "Load the full independent Qwen3.5-9B TE and its local tokenizer assets."
