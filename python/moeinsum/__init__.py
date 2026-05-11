"""moeinsum public Python API.

For v0.1:
  - `einsum(eq, *operands, backend, optimize, accum_dtype)` against
    numpy ndarrays. The reference backend ships now; max /
    native / max_graph land in later phases.
  - `einsum_path(eq, *shapes, optimize)` returns the contraction pair
    sequence the planner chose.
  - `parse_equation(eq)` is a debugging surface that returns the IR.
  - Per-signature LRU cache short-circuits parse + path planning on
    hot calls (see `_cache.py`).

DLPack zero-copy + JAX/PyTorch/MLX interop arrives in P8.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
from numpy.typing import DTypeLike

from ._cache import PLAN_CACHE
from ._cost import path_cost
from ._interop import to_numpy as _to_numpy
from ._native import (
  einsum_compute_path as _einsum_compute_path_native,
)
from ._native import (
  einsum_path as _einsum_path_native,
)
from ._native import (
  einsum_reference as _einsum_reference_native,
)
from ._native import (
  parse_equation as _parse_equation_native,
)

__all__ = [
  "einsum",
  "einsum_path",
  "parse_equation",
  "path_cost",
  "PLAN_CACHE",
]


_BACKENDS = ("reference",)  # max lands in P5.
_OPTIMIZE = (
  "naive",
  "greedy",
  "optimal",
  "auto",
  "random-greedy",
  "branch-all",
  "branch-2",
  "branch-1",
)


def _is_explicit_path(optimize: object) -> bool:
  """True iff `optimize` looks like a caller-supplied path.

  Accept any non-string sequence of step-shaped sequences. We delay the
  per-step validation to `_validate_explicit_path` so the error message
  carries the offending index.
  """
  if isinstance(optimize, str):
    return False
  if not isinstance(optimize, Sequence):
    return False
  # Empty sequence with n_operands ≤ 1 is a valid no-op path; the validator
  # decides. Reject obvious non-paths (lists of strings, ints, etc).
  return all(isinstance(s, Sequence) and not isinstance(s, str) for s in optimize)


def _is_known_optimize(name: str) -> bool:
  """True iff `name` is a known optimizer string.

  Accepts the literal entries from `_OPTIMIZE` plus `random-greedy-N` for
  any N ≥ 1 — the Mojo dispatcher parses the suffix.
  """
  if name in _OPTIMIZE:
    return True
  prefix = "random-greedy-"
  if name.startswith(prefix):
    suffix = name[len(prefix) :]
    if suffix.isdigit():
      return int(suffix) >= 1
  return False


def _validate_explicit_path(
  raw_path: Sequence[Sequence[int]],
  n_operands: int,
) -> list[tuple[int, ...]]:
  """Sanity-check a caller-supplied path, return it as a list of tuples.

  Working-set semantics: each pairwise step removes two operands and
  appends one result; each unary step leaves the working set size
  unchanged. After all steps the working set must contain exactly one
  tensor (the output).

  Raises ValueError with the offending step index on:
    - non-int step entries
    - out-of-range indices
    - step arity ≠ 1 or 2
    - lhs == rhs in a pairwise step
    - final working set ≠ 1 tensor
  """
  path: list[tuple[int, ...]] = []
  working_size = n_operands
  for step_idx, step in enumerate(raw_path):
    step_tuple = tuple(int(s) for s in step)
    if len(step_tuple) == 1:
      (idx,) = step_tuple
      if not (0 <= idx < working_size):
        raise ValueError(
          f"explicit path step {step_idx}: index {idx} out of range "
          f"[0, {working_size})"
        )
      path.append(step_tuple)
    elif len(step_tuple) == 2:
      li, ri = step_tuple
      if not (0 <= li < working_size) or not (0 <= ri < working_size):
        raise ValueError(
          f"explicit path step {step_idx}: indices ({li}, {ri}) out of range "
          f"[0, {working_size})"
        )
      if li == ri:
        raise ValueError(
          f"explicit path step {step_idx}: lhs and rhs both reference {li}"
        )
      working_size -= 1  # two removed, one appended
      path.append(step_tuple)
    else:
      raise ValueError(
        f"explicit path step {step_idx}: arity {len(step_tuple)} not in (1, 2)"
      )

  if working_size != 1:
    raise ValueError(
      f"explicit path leaves {working_size} tensors in working set; expected 1"
    )
  return path


def parse_equation(eq: str) -> dict[str, object]:
  """Parse `eq` and return the structured IR.

  Returns a dict with:
    `inputs`: list of per-operand label-int sequences
    `output`: output label-int sequence
    `n_labels`: distinct label count
    `has_explicit_output`: True iff equation contained `->`
    `label_chars`: label-int -> single-char str (for debug)
  """
  return _parse_equation_native(eq)


def einsum(
  eq: str,
  *operands: np.ndarray,
  backend: str = "reference",
  optimize: str | Sequence[Sequence[int]] = "auto",
  accum_dtype: DTypeLike | None = None,
  dtype: DTypeLike | None = None,
) -> np.ndarray:
  """Compute an einsum.

  Args:
      eq:          NumPy-style einsum equation (e.g. ``"ij,jk->ik"``).
      operands:    Tensor operands. NumPy ndarrays for v0.1.
      backend:     ``"reference"`` (v0.1). ``"max"``,
                   ``"native"``, ``"max_graph"`` land in later phases.
      optimize:    Path optimizer name. ``"auto"`` (default),
                   ``"greedy"``, ``"optimal"``, ``"random-greedy"``,
                   ``"branch-all"`` / ``"branch-2"`` / ``"branch-1"``,
                   or ``"naive"``. Alternatively a caller-supplied
                   explicit path ``[(i, j), ...]``; numpy.einsum / opt_einsum
                   accept the same shape.
      accum_dtype: Internal accumulator precision. None = automatic
                   (fp32 for fp16/bf16 inputs, else match input). Set
                   explicitly to override.
      dtype:       Output dtype; defaults to ``np.result_type(*operands)``.

  Returns:
      A NumPy ndarray with the equation's output shape.
  """
  if backend not in _BACKENDS:
    raise ValueError(f"unknown backend {backend!r}; available: {_BACKENDS}")
  if _is_explicit_path(optimize):
    # Validate eagerly so callers get a clear error. The reference backend
    # ignores path order (it's a global-index loop), so we don't plumb the
    # explicit path through; the validation is the only side-effect for v0.1.
    _validate_explicit_path(
      cast("Sequence[Sequence[int]]", optimize), len(operands)
    )
  elif not _is_known_optimize(cast("str", optimize)):
    raise ValueError(f"unknown optimize {optimize!r}; available: {_OPTIMIZE}")

  if not operands:
    raise ValueError("einsum requires at least one operand")

  # Normalize every operand to a contiguous fp64 numpy array via the
  # DLPack-first adapter — accepts numpy / torch / jax / mlx / cupy /
  # tensorflow / anything implementing `__array__`. The first operand's
  # framework decides the return type (P8 polish; v0.1 always returns
  # numpy regardless).
  arrays = [_to_numpy(o) for o in operands]
  flats = [a.ravel().tolist() for a in arrays]
  shapes = [list(a.shape) for a in arrays]

  flat_out, out_shape = _einsum_reference_native(eq, flats, shapes)
  out = np.array(flat_out, dtype=np.float64).reshape(tuple(out_shape))

  if dtype is None:
    dtype = np.result_type(*arrays) if arrays else np.float64
  if out.dtype != dtype:
    out = out.astype(dtype)
  return out


def einsum_path(
  eq: str,
  *operand_shapes: tuple[int, ...],
  optimize: str | Sequence[Sequence[int]] = "auto",
) -> list[tuple[int, ...]]:
  """Return the contraction pair sequence chosen by the planner.

  Pass ``optimize=[(i, j), ...]`` to validate and echo back a caller-supplied
  explicit path. Otherwise dispatches to the named algorithm.

  Caches by (equation, shape-tuple, optimize) — repeated calls with
  the same arguments are a hash lookup. See `_cache.PLAN_CACHE`.
  """
  shapes_tuple = tuple(tuple(s) for s in operand_shapes)

  if _is_explicit_path(optimize):
    explicit_path = _validate_explicit_path(
      cast("Sequence[Sequence[int]]", optimize), len(shapes_tuple)
    )
    # Caller-supplied paths bypass the LRU — the path is already
    # materialized, caching adds nothing.
    return explicit_path

  if not _is_known_optimize(cast("str", optimize)):
    raise ValueError(f"unknown optimize {optimize!r}; available: {_OPTIMIZE}")

  key = ("__einsum_path__", eq, shapes_tuple, optimize)
  cached = PLAN_CACHE.get(key)
  if cached is not None:
    # Hand back a fresh list — the path tuples are immutable, but the
    # outer container is not, so we don't want a caller's `path.append(...)`
    # to pollute the next cache hit.
    return list(cast("list[tuple[int, ...]]", cached))

  shapes_lists = [list(s) for s in operand_shapes]
  if optimize == "naive" or len(shapes_tuple) <= 1:
    # build_naive_plan emits unary singletons for 1-operand einsums.
    result = _einsum_path_native(eq, shapes_lists)
  else:
    # compute_path's greedy / optimal / auto algorithms return only
    # pairwise steps; 1-operand cases trivially produce an empty path.
    result = _einsum_compute_path_native(eq, shapes_lists, optimize)
  # Store an immutable snapshot — defends against the same caller-
  # mutation surface on the cold path.
  PLAN_CACHE.put(key, tuple(result))
  return list(result)
