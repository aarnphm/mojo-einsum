"""pytest fixtures for moeinsum tests."""

import os
import sys
import sysconfig
from pathlib import Path

# Allow running tests directly from a source checkout without `pip
# install -e .` first — useful for iterating on the parser without
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

# `_native.so` is built with rpaths into the in-source bazel layout
# (`~/workspace/modular/bazel-bin/{KGEN,MLRT}/`), but the runtime dylibs
# actually live in the `_solib_darwin_arm64/_U{KGEN,MLRT,Support}/`
# variant dirs alongside them. Prepend those to DYLD_LIBRARY_PATH so
# `dlopen()` finds `libKGENCompilerRTShared.dylib` &
# `libAsyncRTMojoBindings.dylib` without forcing a manual install_name
# rewrite.
_BAZEL_BIN = Path("/Users/aarnphm/workspace/modular/bazel-bin")
_SOLIB_ROOTS = [
  _BAZEL_BIN / "_solib_darwin_arm64" / "_UKGEN",
  _BAZEL_BIN / "_solib_darwin_arm64" / "_UMLRT",
  _BAZEL_BIN / "_solib_darwin_arm64" / "_USupport",
]
if any(p.is_dir() for p in _SOLIB_ROOTS):
  _existing = os.environ.get("DYLD_LIBRARY_PATH", "")
  _new = ":".join(str(p) for p in _SOLIB_ROOTS if p.is_dir())
  os.environ["DYLD_LIBRARY_PATH"] = f"{_new}:{_existing}" if _existing else _new
