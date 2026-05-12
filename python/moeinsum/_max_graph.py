"""`MaxGraphBackend` - lifts a contraction plan to a MAX graph.

The reference backend executes each plan step eagerly with a global-index
reduction loop. `MaxGraphBackend` instead builds a graph-shaped description of
the whole plan, so MAX can eventually own fusion across the full contraction.

The plan-to-graph translation is owned by the Mojo `_native` extension:
parsing, working-set semantics, unary detection, and B/K/M/N classification all
come from the same IR as the reference backend. The actual `max.graph` import is
lazy: when the package is missing, `MaxGraphBackend()` raises a precise
`ImportError`, but `plan_to_graph_spec(...)` still returns the abstract plan
description so callers can inspect what would be emitted.

Per contraction step this mirrors JAX's lowering shape:

  - classify axes into B/K/M/N buckets,
  - permute lhs to `(*B, *M, *K)` and rhs to `(*B, *K, *N)`,
  - issue matmul to produce `(*B, *M, *N)`,
  - permute back to the equation's stated output order.

Unary steps (trace, diagonal, axis-sum, transpose) are emitted in the spec. The
current executable MAX path supports the BMM-lowerable subset; diagonal
extraction remains a reference-backend case.
"""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass, field
from typing import cast

import numpy as np


def is_available() -> bool:
  """Return True when the `max.graph` Python package is importable.

  `importlib.util.find_spec("max.graph")` raises `ModuleNotFoundError` when the
  parent `max` package is missing, so this normalizes that Python import quirk
  into False.
  """
  try:
    return importlib.util.find_spec("max.graph") is not None
  except ModuleNotFoundError:
    return False


def is_loadable() -> bool:
  """Return True when `max._core` imports successfully in this process.

  Stricter than `is_available`: a local install can exist on disk but fail at
  dlopen time. The concrete dev failure this guards is rpath collision between
  this repo's editable `_native.so` and the PyPI `max` wheel. Test markers should
  gate on this helper so they skip cleanly instead of crashing import.
  """
  if not is_available():
    return False
  try:
    importlib.import_module("max._core")
    return True
  except ImportError:
    return False


def require_max_graph() -> object:
  """Import and return `max.graph` or raise a clear `ImportError`."""
  if not is_available():
    raise ImportError(
      "MaxGraphBackend requires the `max` Python package "
      "(install with `uv pip install -e '.[max]'` and re-import). "
      "See docs/comparisons.md for context."
    )
  return importlib.import_module("max.graph")


# ---------------------------------------------------------------------
# Plan -> abstract op description
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class DimClassification:
  """The B/K/M/N split for a single pairwise contraction step.

  Each tuple holds label characters in the source operand order. `batch` is
  intentionally lhs-order; rhs can place the same batch labels elsewhere, and
  the downstream layout step is responsible for that permutation.
  """

  lhs_labels: str
  rhs_labels: str
  out_labels: str
  batch: tuple[str, ...]
  contract: tuple[str, ...]
  free_lhs: tuple[str, ...]
  free_rhs: tuple[str, ...]


@dataclass
class GraphSpec:
  """An abstract description of what `plan_to_graph(...)` would emit.

  Each entry is a `(op_kind, payload)` pair. `op_kind` is a string like
  `"matmul"`, `"transpose"`, `"reduce_sum"`, or `"diagonal"`. The payload schema
  depends on the op kind. This keeps the spec testable and serializable without
  depending on `max.graph`.
  """

  ops: list[tuple[str, dict[str, object]]] = field(default_factory=list)
  result_index: int = -1  # index into the working-set when the spec runs


def classify_pair(
  lhs_labels: str,
  rhs_labels: str,
  out_labels: str,
) -> DimClassification:
  """Compute the B/K/M/N split for a pairwise step.

  Mirrors JAX's `_einsum` algorithm. The order of each bucket follows
  the lhs's label order for B/K/M and rhs's order for N - matches what
  numpy.einsum produces and keeps the resulting matmul shape stable.
  """
  lhs_set = set(lhs_labels)
  rhs_set = set(rhs_labels)
  out_set = set(out_labels)

  batch: list[str] = []
  for c in lhs_labels:
    if c in rhs_set and c in out_set and c not in batch:
      batch.append(c)

  contract: list[str] = []
  for c in lhs_labels:
    if c in rhs_set and c not in out_set and c not in contract:
      contract.append(c)

  free_lhs: list[str] = []
  for c in lhs_labels:
    if c not in rhs_set and c in out_set and c not in free_lhs:
      free_lhs.append(c)

  free_rhs: list[str] = []
  for c in rhs_labels:
    if c not in lhs_set and c in out_set and c not in free_rhs:
      free_rhs.append(c)

  return DimClassification(
    lhs_labels=lhs_labels,
    rhs_labels=rhs_labels,
    out_labels=out_labels,
    batch=tuple(batch),
    contract=tuple(contract),
    free_lhs=tuple(free_lhs),
    free_rhs=tuple(free_rhs),
  )


def plan_to_graph_spec(
  eq: str,
  shapes: list[tuple[int, ...]],
  path: list[tuple[int, ...]],
) -> GraphSpec:
  """Translate `(eq, shapes, path)` into an op-by-op `GraphSpec`.

  This is the spec a real `max.graph` lowering would consume. It is independent
  of `max.graph` itself, so tests and debug tools can validate lowering shape
  without the runtime package installed.

  Current limitations:
    - no ellipsis,
    - repeated labels within one operand emit a `"diagonal"` unary step,
    - unary path steps are passed through, and reduce-out labels become
      `"reduce_sum"` ops.
  """
  from . import _native  # noqa: PLC0415

  # Mojo raises `Error` which the std.python bindings surface as a bare
  # `Exception` (not RuntimeError). Catch broadly and re-raise the one
  # user-actionable subset (ellipsis) as ValueError so callers can pin a narrow
  # except clause.
  try:
    raw = _native.max_graph_spec(eq, shapes, path)
  except Exception as exc:
    if "does not support ellipsis" in str(exc):
      raise ValueError("plan_to_graph_spec does not support ellipsis") from exc
    raise

  ops: list[tuple[str, dict[str, object]]] = []
  raw_ops = cast("list[tuple[str, dict[str, object]]]", raw["ops"])
  for kind, payload in raw_ops:
    normalized = dict(payload)
    for key in ("batch", "contract", "free_lhs", "free_rhs"):
      if key in normalized:
        normalized[key] = tuple(cast("list[str]", normalized[key]))
    ops.append((kind, normalized))

  return GraphSpec(ops=ops, result_index=int(cast("int", raw["result_index"])))


# ---------------------------------------------------------------------
# MaxGraphBackend - only callable when max.graph is installed
# ---------------------------------------------------------------------


class MaxGraphBackend:
  """Lifts a `(eq, shapes, path)` triple to a MAX graph and executes it.

  `__init__()` validates that `max.graph` is importable. Per-call execution is
  `execute(eq, shapes, path, operands)`, the same signature future native
  backends should share.

  v0.1 status: executable for the same BMM-lowerable subset as `backend="max"`.
  The plan-to-graph translation (`plan_to_graph_spec`) is Mojo-owned and tested;
  graph object construction still uses MAX's Python API.
  """

  def __init__(self) -> None:
    self._max_graph = require_max_graph()

  def graph_spec_for(
    self,
    eq: str,
    shapes: list[tuple[int, ...]],
    path: list[tuple[int, ...]],
  ) -> GraphSpec:
    """Return the abstract `GraphSpec` for `(eq, shapes, path)`.

    Useful for inspection / cache keying without paying the
    `max.graph.Graph.compile()` cost.
    """
    return plan_to_graph_spec(eq, shapes, path)

  def execute(
    self,
    eq: str,
    shapes: list[tuple[int, ...]],
    path: list[tuple[int, ...]],
    operands: list[object],
  ) -> object:  # pragma: no cover - exercised only when max.graph is installed
    """Run the plan via the executable MAX Graph bridge."""
    _ = shapes
    from ._max_backend import execute_max  # noqa: PLC0415

    return execute_max(eq, [np.asarray(op) for op in operands], path, "max")
