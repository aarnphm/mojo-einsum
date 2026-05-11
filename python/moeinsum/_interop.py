"""Framework-agnostic array adapter.

Two responsibilities - converting *into* a NumPy view (without forcing
fp64) and converting *out of* one back to the caller's framework.
Both paths prefer DLPack zero-copy, then fall back to `np.asarray` /
the destination framework's array constructor.

Conversion in (`to_numpy`):
  1. `numpy.ndarray` - passthrough.
  2. `array.__dlpack__()` - DLPack capsule. numpy 1.23+, torch 2.0+,
     jax 0.4+, mlx 0.2+. Routed through `np.from_dlpack`.
  3. `np.asarray(array)` - fallback for `__array_interface__` /
     `__array__` / nested-list objects.

Conversion out (`from_numpy`):
  1. `"numpy"` - passthrough.
  2. `"torch"` - `torch.from_numpy(arr).to(reference_device)`.
  3. `"jax"` - `jax.numpy.asarray(arr)`; lands on default device.
  4. `"mlx"` - `mx.array(arr)`; mlx has no DLPack-from-numpy in 0.x,
     so this is a copy.
  5. `"cupy"` / `"tensorflow"` - `np.asarray` round-trip via DLPack
     when supported, copy when not.

Dtype handling: by default, both functions preserve the source's
dtype. Pass `dtype=` to cast explicitly. The fp64-everywhere v0.1
simplification is gone - callers see the dtype they sent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeGuard

if TYPE_CHECKING:
  from typing_extensions import CapsuleType
  from numpy.typing import DTypeLike

import numpy as np

_ZERO_COPY_DLPACK_SOURCES = ("torch", "jax", "mlx", "cupy", "tensorflow")

# Some frameworks ship their array types from a sibling C-extension module
# whose name differs from the user-facing pip name. `type(jax_array).__module__`
# is `"jaxlib._jax"` (or `"jaxlib.xla_extension"` on older versions), so a naive
# `module.split('.', 1)[0]` returns `"jaxlib"` and the dispatch falls through to
# `"other"`, killing the round-trip back to a jax array. Alias the C-extension
# package to the user-facing one here.
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

  `dtype` defaults to None - preserve the source's dtype. Pass an
  explicit dtype to cast (the v0.1 callsite did `dtype=np.float64`
  to force fp64 for the reference backend's FFI; that conversion now
  lives at the FFI boundary, not here).
  """
  if _looks_like_dlpack_source(obj):
    try:
      candidate = np.from_dlpack(obj)
    except (TypeError, RuntimeError, BufferError):
      # GPU-resident tensors fail np.from_dlpack - fall through to
      # the generic adapter, which will trip the source's own
      # CPU-fallback (`.cpu()` for torch, `np.asarray` for jax).
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

  No-op for `"numpy"` / `"other"` (the caller never asked for a
  framework array). For `"torch"` / `"jax"` / `"mlx"` /
  `"cupy"` / `"tensorflow"` we import lazily so the dependency
  is genuinely optional.

  Raises ImportError if the named framework isn't installed - the
  caller chose this return path explicitly via the first operand,
  so a missing-package surface is the right failure mode.
  """
  if kind in ("numpy", "other"):
    return arr
  if kind == "torch":
    import torch  # noqa: PLC0415 - lazy by design

    # DLPack-from-numpy works in torch 2.0+; older falls through to
    # the from_numpy path (a copy).
    try:
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
