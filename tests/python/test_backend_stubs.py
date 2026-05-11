"""Backend stub tests — exercise the architectural seam.

`MaxGraphBackend` is a stretch deliverable (P14). The shim exists so
callers can detect availability and get clear error messages, not so
they can actually run a MAX graph. These tests verify the detection /
error surface — actual execution tests land with the P14 implementation.
"""

from __future__ import annotations

import importlib

import pytest
from moeinsum._max_graph import MaxGraphBackend, is_available, require_max_graph


def test_is_available_matches_import_spec() -> None:
  try:
    spec_present = importlib.util.find_spec("max.graph") is not None
  except ModuleNotFoundError:
    spec_present = False
  assert is_available() is spec_present


def test_max_graph_backend_init_raises_not_implemented() -> None:
  with pytest.raises(NotImplementedError, match="P14"):
    MaxGraphBackend()


def test_require_max_graph_error_when_missing() -> None:
  if is_available():
    pytest.skip("max.graph installed; cannot test the missing-dep path")
  with pytest.raises(ImportError, match="MaxGraphBackend requires"):
    require_max_graph()
