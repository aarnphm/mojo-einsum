"""mojo-einsum public Python API.

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

from typing import Any

import numpy as np

from ._cache import PLAN_CACHE, make_key
from ._native import (
    einsum_path as _einsum_path_native,
    einsum_reference as _einsum_reference_native,
    parse_equation as _parse_equation_native,
)

__all__ = ["einsum", "einsum_path", "parse_equation", "PLAN_CACHE"]


_BACKENDS = ("reference",)  # max lands in P5.
_OPTIMIZE = ("naive", "greedy", "optimal", "auto")


def parse_equation(eq: str) -> dict[str, Any]:
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
    optimize: str = "auto",
    accum_dtype: Any = None,
    dtype: Any = None,
) -> np.ndarray:
    """Compute an einsum.

    Args:
        eq:          NumPy-style einsum equation (e.g. ``"ij,jk->ik"``).
        operands:    Tensor operands. NumPy ndarrays for v0.1.
        backend:     ``"reference"`` (v0.1). ``"max"``,
                     ``"native"``, ``"max_graph"`` land in later phases.
        optimize:    Path optimizer name. ``"auto"`` (default),
                     ``"greedy"``, ``"optimal"``, or ``"naive"``.
                     opt_einsum's ``random-greedy`` and ``branch`` family
                     are P4 polish.
        accum_dtype: Internal accumulator precision. None = automatic
                     (fp32 for fp16/bf16 inputs, else match input). Set
                     explicitly to override.
        dtype:       Output dtype; defaults to ``np.result_type(*operands)``.

    Returns:
        A NumPy ndarray with the equation's output shape.
    """
    if backend not in _BACKENDS:
        raise ValueError(
            f"unknown backend {backend!r}; available: {_BACKENDS}"
        )
    if optimize not in _OPTIMIZE:
        raise ValueError(
            f"unknown optimize {optimize!r}; available: {_OPTIMIZE}"
        )

    if not operands:
        raise ValueError("einsum requires at least one operand")

    arrays = [np.ascontiguousarray(np.asarray(o, dtype=np.float64)) for o in operands]
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
    eq: str, *operand_shapes: tuple[int, ...], optimize: str = "auto"
) -> list[tuple[int, ...]]:
    """Return the contraction pair sequence chosen by the planner.

    Caches by (equation, shape-tuple, optimize) — repeated calls with
    the same arguments are a hash lookup. See `_cache.PLAN_CACHE`.
    """
    if optimize not in _OPTIMIZE:
        raise ValueError(
            f"unknown optimize {optimize!r}; available: {_OPTIMIZE}"
        )

    shapes_tuple = tuple(tuple(s) for s in operand_shapes)
    key = ("__einsum_path__", eq, shapes_tuple, optimize)
    cached = PLAN_CACHE.get(key)
    if cached is not None:
        return cached

    shapes_lists = [list(s) for s in operand_shapes]
    result = _einsum_path_native(eq, shapes_lists)
    PLAN_CACHE.put(key, result)
    return result
