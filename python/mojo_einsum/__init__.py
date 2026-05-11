"""mojo-einsum public Python API.

For v0.1 (P1):
  - `einsum(eq, *operands, backend='reference', optimize='naive')`
    works against numpy ndarrays via copy-in / copy-out. Mojo-side runs
    the reference backend (naive nested loop).
  - `einsum_path(eq, *shapes, optimize='naive')` returns the contraction
    pair sequence the plan-builder chose.
  - `parse_equation(eq)` is a debugging surface that returns the IR.

DLPack zero-copy + JAX/PyTorch/MLX interop arrives in P8.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._native import (
    einsum_path as _einsum_path_native,
    einsum_reference as _einsum_reference_native,
    parse_equation as _parse_equation_native,
)

__all__ = ["einsum", "einsum_path", "parse_equation"]


_BACKENDS = ("reference",)  # max_kernels lands in P5.


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
    optimize: str = "naive",
    dtype: Any = None,
) -> np.ndarray:
    """Compute an einsum.

    Args:
        eq:       NumPy-style einsum equation (e.g. ``"ij,jk->ik"``).
        operands: Tensor operands. NumPy ndarrays for now. The reference
                  backend casts everything to float64 internally;
                  arbitrary dtypes are P9.
        backend:  Currently only ``"reference"``. ``"max_kernels"``,
                  ``"native"``, ``"max_graph"`` land in later phases.
        optimize: Path optimizer. Currently only ``"naive"``
                  (left-to-right). ``"greedy"``, ``"optimal"``, etc.
                  land in P4.
        dtype:    Output dtype; defaults to ``float64`` (the reference
                  backend's internal type) or ``np.result_type(*operands)``.

    Returns:
        A NumPy ndarray with the equation's output shape.
    """
    if backend not in _BACKENDS:
        raise ValueError(
            f"unknown backend {backend!r}; available: {_BACKENDS}"
        )
    if optimize != "naive":
        raise NotImplementedError(
            f"optimize={optimize!r} not yet supported; only 'naive' in P1."
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
    eq: str, *operand_shapes: tuple[int, ...], optimize: str = "naive"
) -> list[tuple[int, ...]]:
    """Return the contraction pair sequence chosen by the planner.

    Operands are described by their shapes (no data needed). Each
    returned tuple is either ``(lhs_idx, rhs_idx)`` for a pairwise step
    or ``(operand_idx,)`` for a unary step.
    """
    if optimize != "naive":
        raise NotImplementedError(
            f"optimize={optimize!r} not yet supported"
        )
    shapes_lists = [list(s) for s in operand_shapes]
    return _einsum_path_native(eq, shapes_lists)
