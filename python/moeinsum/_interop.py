"""Framework-agnostic array → NumPy adapter.

The Mojo FFI consumes flat Python lists + shape lists (until the
TileTensor FFI lands in P5). Above that, we let callers pass any
NumPy-style array — `numpy.ndarray`, `torch.Tensor`, `jax.Array`,
`mlx.array`, etc. This module normalizes them to a single internal
`(ndarray, original)` tuple so the caller can choose what to return.

Conversion order (cheapest first):
  1. `numpy.ndarray` — pass through.
  2. `array.__dlpack__()` — DLPack capsule (numpy 1.23+, torch 2.0+,
     jax 0.4+, mlx 0.2+). We convert via `numpy.from_dlpack`.
  3. `np.asarray(array)` — fallback for anything with the array
     protocol (`__array_interface__`, `__array__`, etc.).

We always materialize as a contiguous fp64 buffer before handing off
to the reference backend — that's a v0.1 simplification. P8 polish:
keep the original dtype, route through TileTensor for zero-copy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import DTypeLike


_ZERO_COPY_DLPACK_SOURCES = ("torch", "jax", "mlx", "cupy", "tensorflow")


def _looks_like_dlpack_source(obj: Any) -> bool:
  """True if `obj` exposes DLPack but isn't already a numpy array."""
  if isinstance(obj, np.ndarray):
    return False
  return hasattr(obj, "__dlpack__") and hasattr(obj, "__dlpack_device__")


def to_numpy(obj: Any, *, dtype: DTypeLike = np.float64) -> np.ndarray:
  """Convert an arbitrary array-like to a contiguous NumPy array.

  Prefer DLPack zero-copy when the source exposes the protocol;
  fall back to `np.asarray` otherwise. The result is always
  C-contiguous and cast to `dtype` (default fp64).
  """
  if _looks_like_dlpack_source(obj):
    try:
      candidate = np.from_dlpack(obj)
    except (TypeError, RuntimeError, BufferError):
      # GPU-resident tensors fail np.from_dlpack — fall through to
      # the generic adapter, which will trip the source's own
      # CPU-fallback (`.cpu()` for torch, `np.asarray` for jax).
      candidate = np.asarray(obj)
  else:
    candidate = np.asarray(obj)

  return np.ascontiguousarray(candidate, dtype=dtype)


def source_kind(obj: Any) -> str:
  """Best-effort identification of the input array's framework.

  Returns one of "numpy", "torch", "jax", "mlx", "cupy",
  "tensorflow", or "other". Used by `einsum` to choose what to
  return to the caller (a `numpy.ndarray` for numpy inputs, the
  original framework's array type otherwise — P8 polish).
  """
  module = type(obj).__module__.split(".", 1)[0]
  if module == "numpy":
    return "numpy"
  if module in _ZERO_COPY_DLPACK_SOURCES:
    return module
  return "other"
