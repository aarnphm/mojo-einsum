"""DLPack / framework-interop tests.

Verifies the Python wrapper consumes any array-like through
DLPack-or-fallback. Torch / JAX / MLX are tested only when installed;
the always-on path is numpy + Python lists.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure the python/ root is on sys.path (mirrors conftest behavior).
_REPO_ROOT = Path(__file__).parent.parent.parent
_PY_ROOT = _REPO_ROOT / "python"
if str(_PY_ROOT) not in sys.path:
  sys.path.insert(0, str(_PY_ROOT))

import moeinsum
from moeinsum._interop import source_kind, to_numpy


def test_source_kind_numpy() -> None:
  assert source_kind(np.eye(3)) == "numpy"


def test_source_kind_other_for_list() -> None:
  # Plain Python lists aren't a framework array.
  assert source_kind([[1.0, 2.0], [3.0, 4.0]]) == "other"


def test_to_numpy_numpy_passthrough() -> None:
  a = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
  b = to_numpy(a)
  assert b.dtype == np.float64
  assert b.flags["C_CONTIGUOUS"]
  np.testing.assert_array_equal(b, a.astype(np.float64))


def test_to_numpy_from_python_list() -> None:
  out = to_numpy([[1, 2], [3, 4]])
  assert out.dtype == np.float64
  assert out.shape == (2, 2)


def test_einsum_accepts_python_lists() -> None:
  # Plain Python nested lists must work via np.asarray fallback.
  a = [[1.0, 2.0], [3.0, 4.0]]
  b = [[5.0, 6.0], [7.0, 8.0]]
  out = moeinsum.einsum("ij,jk->ik", a, b)
  np.testing.assert_allclose(out, np.array(a) @ np.array(b))


def _try_import(name: str):
  try:
    return importlib.import_module(name)
  except ImportError:
    return None


_torch = _try_import("torch")
_jax = _try_import("jax")
_mlx = _try_import("mlx") if _try_import("mlx.core") else None


@pytest.mark.skipif(_torch is None, reason="torch not installed")
def test_einsum_torch_dlpack() -> None:
  assert _torch is not None
  a = _torch.eye(3, dtype=_torch.float64)
  b = _torch.eye(3, dtype=_torch.float64) * 2.0
  out = moeinsum.einsum("ij,jk->ik", a, b)
  np.testing.assert_allclose(out, np.eye(3) * 2.0)


@pytest.mark.skipif(_jax is None, reason="jax not installed")
def test_einsum_jax_dlpack() -> None:
  assert _jax is not None
  a = _jax.numpy.eye(3, dtype="float64")
  b = _jax.numpy.eye(3, dtype="float64") * 3.0
  out = moeinsum.einsum("ij,jk->ik", a, b)
  np.testing.assert_allclose(out, np.eye(3) * 3.0)


@pytest.mark.skipif(_mlx is None, reason="mlx not installed")
def test_einsum_mlx_dlpack() -> None:
  import mlx.core as mx

  a = mx.eye(3, dtype=mx.float32)
  b = mx.eye(3, dtype=mx.float32) * 4.0
  out = moeinsum.einsum("ij,jk->ik", a, b)
  np.testing.assert_allclose(out, np.eye(3) * 4.0, atol=1e-6)
