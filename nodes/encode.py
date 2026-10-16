class MOSSOfficialTextEncode:
    CATEGORY = "MOSS-Image/Official"
    FUNCTION = "encode"
    RETURN_TYPES = ("MOSS_IMAGE_CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    DESCRIPTION = "Official Qwen chat template and last hidden state, without prompt-weight rewriting. TE is released after encoding."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "text_encoder": ("MOSS_IMAGE_TE",),
            "positive": ("STRING", {"multiline": True, "default": "A small red fox sitting in a green meadow, soft morning light."}),
            "negative": ("STRING", {"multiline": True, "default": ""}),
            "max_sequence_length": ("INT", {"default": 4096, "min": 1, "max": 4096}),
        }}

    def encode(self, text_encoder, positive, negative="", max_sequence_length=4096):
        from ..runtime import encode
        return (encode(text_encoder, positive, negative, max_sequence_length),)
