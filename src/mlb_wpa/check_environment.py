"""Verify installed libraries without network, using project-local caches."""

import json
import os
import platform
from importlib import import_module, metadata

from .sample import ROOT


def main():
    # pybaseball creates a directory even when importing with caching off.
    # Respect an explicit user override; otherwise keep generated files local.
    cache = ROOT / ".cache"
    os.environ.setdefault("PYBASEBALL_CACHE", str(cache / "pybaseball"))
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    versions = {}
    modules = ["numpy", "pandas", "matplotlib", "seaborn", "scipy", "pybaseball",
               "requests", "bs4", "pytest", "nbformat", "ipykernel"]
    for name in modules:
        import_module(name)
        versions[name] = metadata.version("beautifulsoup4" if name == "bs4" else name)
    report = {"python": platform.python_version(), "platform": platform.platform(),
              "installed_imports": versions, "status": "all imports passed",
              "cache_configuration": {k: os.environ[k] for k in ("PYBASEBALL_CACHE", "MPLCONFIGDIR")}}
    (ROOT / "reports/environment.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
