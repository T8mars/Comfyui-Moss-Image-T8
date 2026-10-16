"""Exercise the actual PowerShell entry point without starting ComfyUI."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows PowerShell launcher")


@pytest.mark.parametrize("exit_code", [0, 23])
def test_launcher_preserves_python_exit_code_and_spaced_unicode_paths(tmp_path, exit_code):
    project = tmp_path / "节点 工程"
    comfy = tmp_path / "Comfy 源码"
    project.mkdir()
    comfy.mkdir()
    launcher = project / "launch_local.ps1"
    shutil.copyfile(Path(__file__).resolve().parents[1] / "launch_local.ps1", launcher)
    marker = comfy / "arguments.json"
    (comfy / "main.py").write_text(
        "import json, sys\nfrom pathlib import Path\n"
        "Path(__file__).with_name('arguments.json').write_text(json.dumps(sys.argv[1:]))\n"
        f"sys.exit({exit_code})\n", encoding="utf-8")
    result = subprocess.run(["powershell.exe", "-NoProfile", "-File", str(launcher),
                             "-ComfyRoot", str(comfy), "-Python", sys.executable, "-Port", "8293"],
                            capture_output=True, timeout=30)
    assert result.returncode == exit_code, result.stderr.decode(errors="replace")
    arguments = json.loads(marker.read_text())
    assert arguments[arguments.index("--base-directory") + 1] == str(project / ".comfy")
    assert arguments[arguments.index("--models-directory") + 1] == str(project / "models")
    assert arguments[arguments.index("--port") + 1] == "8293"
