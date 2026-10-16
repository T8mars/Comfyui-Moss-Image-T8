import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("moss_plugin", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
plugin = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plugin
spec.loader.exec_module(plugin)
# Pytest's package collector imports a hyphenated custom-node root as __init__.
# Reuse the explicitly loaded package rather than a parentless relative import.
sys.modules.setdefault("__init__", plugin)
