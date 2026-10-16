"""Build a portable custom-node ZIP without environments, caches or weights."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def main():
    root = Path(__file__).resolve().parents[1]
    destination = root / "dist/ComfyUI-MOSS-Image.zip"
    destination.parent.mkdir(exist_ok=True)
    files = [root / name for name in ("__init__.py", "components.py", "runtime.py", "README.md", "README_ZH.md", "LICENSE",
                                      "NOTICE", "requirements.txt", "pyproject.toml", "launch_local.ps1")]
    for name in ("nodes", "scripts", "vendor", "example_workflows", "docs", "tests"):
        files.extend(file for file in (root / name).rglob("*") if file.is_file()
                     and "__pycache__" not in file.parts and file.suffix != ".pyc")
    with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
        for file in sorted(files):
            archive.write(file, "ComfyUI-MOSS-Image/" + file.relative_to(root).as_posix())
    with ZipFile(destination) as archive:
        assert archive.testzip() is None
        assert not any(name.endswith(".safetensors") for name in archive.namelist())
    print(f"Packaged {len(files)} files: {destination} ({destination.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
