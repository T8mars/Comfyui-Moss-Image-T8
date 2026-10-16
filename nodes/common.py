from ..components import fingerprint, inspect_component


class Loader:
    CATEGORY = "MOSS-Image/Official"
    FUNCTION = "load"
    folder = ""
    kind = ""

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        result = {"model_name": (folder_paths.get_filename_list(cls.folder),),
                  "device": (["auto", "cuda", "cpu"], {"default": "auto"})}
        if cls.kind == "transformer":
            result["precision"] = (["bf16", "fp32"], {"default": "bf16"})
        return {"required": result}

    @classmethod
    def IS_CHANGED(cls, model_name, **kwargs):
        import folder_paths
        return fingerprint(folder_paths.get_full_path_or_raise(cls.folder, model_name))

    def load(self, model_name, device="auto", precision="bf16"):
        import folder_paths
        path = folder_paths.get_full_path_or_raise(self.folder, model_name)
        return (inspect_component(path, self.kind, device, precision),)
