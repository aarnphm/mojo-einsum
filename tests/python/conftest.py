"""pytest fixtures for mojo-einsum tests."""

import sys
from pathlib import Path

# Allow running tests directly from a source checkout without `pip
# install -e .` first — useful for iterating on the parser without
# rebuilding the Mojo extension. (The extension is required for
# tests that exercise `_native`, of course.)
_REPO_ROOT = Path(__file__).parent.parent.parent
_PY_ROOT = _REPO_ROOT / "python"
if str(_PY_ROOT) not in sys.path:
    sys.path.insert(0, str(_PY_ROOT))
