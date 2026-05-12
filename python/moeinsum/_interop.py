"""Framework-agnostic array adapter.

Two responsibilities - converting into a NumPy view without forcing
fp64, and converting out of one back to the caller's framework. Both
paths prefer DLPack zero-copy, then fall back to `np.asarray` / the
destination framework's array constructor.

Conversion in (`to_numpy`):
  1. `numpy.ndarray` - passthrough.
  2. `array.__dlpack__()` - DLPack capsule. numpy 1.23+, torch 2.0+,
     jax 0.4+, mlx 0.2+. Routed through `np.from_dlpack`.
  3. `np.asarray(array)` - fallback for `__array_interface__` /
     `__array__` / nested-list objects.

Conversion out (`from_numpy`):
  1. `"numpy"` - passthrough.
  2. `"torch"` - `torch.from_dlpack(arr)` where possible, else
     `torch.from_numpy(arr)`.
  3. `"jax"` - `jax.numpy.asarray(arr)`; lands on default device.
  4. `"mlx"` - `mx.array(arr)`; mlx has no DLPack-from-numpy in 0.x,
     so this is a copy.
  5. `"cupy"` / `"tensorflow"` - framework constructors.

Dtype handling: by default, both functions preserve the source's dtype.
Pass `dtype=` to cast explicitly. The fp64 conversion belongs at the
reference-backend FFI boundary, not here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeGuard

if TYPE_CHECKING:
  from numpy.typing import DTypeLike
  from typing_extensions import CapsuleType

import numpy as np

_ZERO_COPY_DLPACK_SOURCES = ("torch", "jax", "mlx", "cupy", "tensorflow")

# Some frameworks ship array types from a sibling C-extension module whose
# name differs from the user-facing pip name. `type(jax_array).__module__`
# can be `"jaxlib._jax"` (or `"jaxlib.xla_extension"` on older versions),
# so a naive module split returns `"jaxlib"` and kills the round-trip back
# to a jax array. Alias the C-extension package to the public framework.
_MODULE_ALIASES = {
  "jaxlib": "jax",
}


class _DLPackSource(Protocol):
  def __dlpack__(self, *, stream: None = None) -> CapsuleType: ...

  def __dlpack_device__(self) -> tuple[int, int]: ...


def _looks_like_dlpack_source(obj: object) -> TypeGuard[_DLPackSource]:
  if isinstance(obj, np.ndarray):
    return False
  return hasattr(obj, "__dlpack__") and hasattr(obj, "__dlpack_device__")


def to_numpy(
  obj: object,
  *,
  dtype: DTypeLike | None = None,
) -> np.ndarray:
  """Convert an arbitrary array-like to a contiguous NumPy array.

  Prefer DLPack zero-copy when the source exposes the protocol; fall
  back to `np.asarray` otherwise. The result is always C-contiguous.
  `dtype` defaults to None, preserving the source dtype.
  """
  if _looks_like_dlpack_source(obj):
    try:
      candidate = np.from_dlpack(obj)
    except (TypeError, RuntimeError, BufferError):
      # GPU-resident tensors can fail `np.from_dlpack` when NumPy cannot
      # consume the device. Fall through to the framework's generic
      # adapter, which may perform its own CPU fallback.
      candidate = np.asarray(obj)
  else:
    candidate = np.asarray(obj)

  if dtype is None:
    return np.ascontiguousarray(candidate)
  return np.ascontiguousarray(candidate, dtype=dtype)


def source_kind(obj: object) -> str:
  """Best-effort identification of the input array's framework.

  Returns one of `"numpy"`, `"torch"`, `"jax"`, `"mlx"`, `"cupy"`,
  `"tensorflow"`, or `"other"`. Used by `einsum` to choose the return
  type - same kind in as out, matching numpy.einsum / torch.einsum /
  jnp.einsum conventions.
  """
  module = type(obj).__module__.split(".", 1)[0]
  module = _MODULE_ALIASES.get(module, module)
  if module == "numpy":
    return "numpy"
  if module in _ZERO_COPY_DLPACK_SOURCES:
    return module
  return "other"


def from_numpy(arr: np.ndarray, kind: str) -> object:
  """Round-trip a NumPy result back to the source framework.

  No-op for `"numpy"` / `"other"`. Optional frameworks import lazily so
  they remain genuinely optional. A missing package should surface as
  ImportError when the caller explicitly chose that return path.
  """
  if kind in ("numpy", "other"):
    return arr
  if kind == "torch":
    import torch  # noqa: PLC0415 - lazy by design

    try:
      # DLPack-from-numpy works in torch 2.0+; older versions fall back
      # to from_numpy.
      return torch.from_dlpack(arr)
    except (TypeError, RuntimeError):
      return torch.from_numpy(arr)
  if kind == "jax":
    import jax.numpy as jnp  # noqa: PLC0415

    return jnp.asarray(arr)
  if kind == "mlx":
    import mlx.core as mx  # noqa: PLC0415

    return mx.array(arr)
  if kind == "cupy":
    import cupy  # noqa: PLC0415

    return cupy.asarray(arr)
  if kind == "tensorflow":
    import tensorflow as tf  # noqa: PLC0415

    return tf.convert_to_tensor(arr)
  # Unknown kind - be conservative.
  return arr
