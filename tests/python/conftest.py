"""pytest fixtures for moeinsum tests."""

import os
import sys
import sysconfig
from pathlib import Path

# Allow running tests directly from a source checkout without `pip
# install -e .` first - useful for iterating on the parser without
# rebuilding the Mojo extension. (The extension is required for
# tests that exercise `_native`, of course.)
_REPO_ROOT = Path(__file__).parent.parent.parent
_PY_ROOT = _REPO_ROOT / "python"
if str(_PY_ROOT) not in sys.path:
  sys.path.insert(0, str(_PY_ROOT))

# The mohaus editable hook rebuilds `_native` on import, and that build
# needs to link a libpython. uv-managed Pythons aren't on a system
# loader path, so point Mojo at the canonical dylib derived from
# sysconfig.
if "MOJO_PYTHON_LIBRARY" not in os.environ:
  _libdir = sysconfig.get_config_var("LIBDIR")
  _libname = sysconfig.get_config_var("LDLIBRARY")
  if _libdir and _libname:
    _candidate = Path(_libdir) / _libname
    if _candidate.is_file():
      os.environ["MOJO_PYTHON_LIBRARY"] = str(_candidate)

# `_native.so` is built against the repo-local uv environment. Keep that
# runtime first in subprocesses too; stale in-tree Modular dylibs can share
# names while missing symbols from the packaged compiler runtime.
_MODULAR_LIB = _REPO_ROOT / ".venv" / "lib" / "python3.11" / "site-packages" / "modular" / "lib"
if _MODULAR_LIB.is_dir():
  _existing = os.environ.get("DYLD_LIBRARY_PATH", "")
  os.environ["DYLD_LIBRARY_PATH"] = f"{_MODULAR_LIB}:{_existing}" if _existing else str(_MODULAR_LIB)
