"""DLPack / framework-interop tests.

Verifies the Python wrapper consumes any array-like through
DLPack-or-fallback, preserves dtype on the way in, and routes the
output back to the source framework. Torch / JAX / MLX are tested
only when installed; the always-on path is numpy + Python lists.
"""

from __future__ import annotations

import importlib
from types import ModuleType

import moeinsum
import numpy as np
import pytest
from moeinsum._interop import from_numpy, source_kind, to_numpy


def test_source_kind_numpy() -> None:
  assert source_kind(np.eye(3)) == "numpy"


def test_source_kind_other_for_list() -> None:
  # Plain Python lists aren't a framework array.
  assert source_kind([[1.0, 2.0], [3.0, 4.0]]) == "other"


def test_to_numpy_preserves_dtype() -> None:
  """Default behaviour: pass-through dtype, no auto-cast to fp64."""
  a = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
  b = to_numpy(a)
  assert b.dtype == np.float32
  assert b.flags["C_CONTIGUOUS"]
  np.testing.assert_array_equal(b, a)


def test_to_numpy_explicit_dtype() -> None:
  """`dtype=` still casts when the caller asks for it."""
  a = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
  b = to_numpy(a, dtype=np.float64)
  assert b.dtype == np.float64
  np.testing.assert_array_equal(b, a.astype(np.float64))


def test_to_numpy_from_python_list() -> None:
  out = to_numpy([[1, 2], [3, 4]])
  # numpy infers int64 from a list of ints — we preserve that.
  assert out.dtype == np.int64
  assert out.shape == (2, 2)


def test_from_numpy_numpy_passthrough() -> None:
  a = np.eye(3)
  b = from_numpy(a, "numpy")
  assert b is a


def test_from_numpy_other_passthrough() -> None:
  a = np.eye(3)
  b = from_numpy(a, "other")
  assert b is a


def test_einsum_accepts_python_lists() -> None:
  # Plain Python nested lists must work via np.asarray fallback.
  a = [[1.0, 2.0], [3.0, 4.0]]
  b = [[5.0, 6.0], [7.0, 8.0]]
  out = moeinsum.einsum("ij,jk->ik", a, b)
  np.testing.assert_allclose(out, np.array(a) @ np.array(b))


def test_einsum_preserves_fp32_dtype() -> None:
  a = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
  b = np.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
  out = moeinsum.einsum("ij,jk->ik", a, b)
  assert isinstance(out, np.ndarray)
  assert out.dtype == np.float32
  np.testing.assert_allclose(out, a @ b, atol=1e-5)


def test_einsum_preserves_int_dtype() -> None:
  a = np.arange(6, dtype=np.int64).reshape(2, 3)
  b = np.arange(12, dtype=np.int64).reshape(3, 4)
  out = moeinsum.einsum("ij,jk->ik", a, b)
  assert isinstance(out, np.ndarray)
  assert out.dtype == np.int64
  np.testing.assert_array_equal(out, a @ b)


def test_einsum_explicit_dtype_override() -> None:
  a = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
  b = np.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
  out = moeinsum.einsum("ij,jk->ik", a, b, dtype=np.float64)
  assert isinstance(out, np.ndarray)
  assert out.dtype == np.float64
  np.testing.assert_allclose(out, (a @ b).astype(np.float64))


def _try_import(name: str) -> ModuleType | None:
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
  # torch in → torch out.
  assert isinstance(out, _torch.Tensor)
  np.testing.assert_allclose(np.asarray(out), np.eye(3) * 2.0)


@pytest.mark.skipif(_jax is None, reason="jax not installed")
def test_einsum_jax_dlpack() -> None:
  """jax in → jax out via DLPack.

  Note: jax demotes `dtype="float64"` to fp32 unless JAX_ENABLE_X64=1.
  That's expected jax behavior, not a bug in this layer — the test
  builds operands with whatever dtype jax actually allocates.
  Pre-fix, `source_kind` returned "other" for jax arrays (their type
  module is `jaxlib._jax`, not `jax`), so the round-trip collapsed to
  numpy. Fixed by aliasing `jaxlib` → `jax` in `_MODULE_ALIASES`.
  """
  assert _jax is not None
  a = _jax.numpy.eye(3)
  b = _jax.numpy.eye(3) * 3.0
  out = moeinsum.einsum("ij,jk->ik", a, b)
  # jax in → jax out. type(out).__module__ is "jaxlib._jax" — startswith
  # "jax" is satisfied either way.
  assert type(out).__module__.startswith("jax")
  np.testing.assert_allclose(np.asarray(out), np.eye(3) * 3.0)


@pytest.mark.skipif(_mlx is None, reason="mlx not installed")
def test_einsum_mlx_dlpack() -> None:
  import mlx.core as mx

  a = mx.eye(3, dtype=mx.float32)
  b = mx.eye(3, dtype=mx.float32) * 4.0
  out = moeinsum.einsum("ij,jk->ik", a, b)
  # mlx in → mlx out.
  assert type(out).__module__.startswith("mlx")
  np.testing.assert_allclose(np.asarray(out), np.eye(3) * 4.0, atol=1e-6)


@pytest.mark.skipif(_torch is None, reason="torch not installed")
def test_einsum_return_type_override() -> None:
  """`return_type="numpy"` forces a numpy return even from a torch input."""
  assert _torch is not None
  a = _torch.eye(3, dtype=_torch.float64)
  b = _torch.eye(3, dtype=_torch.float64) * 5.0
  out = moeinsum.einsum("ij,jk->ik", a, b, return_type="numpy")
  assert isinstance(out, np.ndarray)
  np.testing.assert_allclose(out, np.eye(3) * 5.0)
