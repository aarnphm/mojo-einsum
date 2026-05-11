"""`MaxGraphBackend` - lifts a contraction plan to a MAX graph.

P14 stretch deliverable. The default `reference` backend executes each
plan step eagerly with a global-index reduction loop; `MaxGraphBackend`
instead assembles the entire plan as a `max.graph.Graph`, hands it to
MAX's compiler, and lets MAX fuse - including any elementwise ops the
caller composes around the einsum - into one kernel.

The plan-to-graph translation is owned by the Mojo `_native` extension:
parsing, working-set semantics, unary detection, and B/K/M/N
classification all come from the same IR as the reference backend.
The actual `max.graph` import is lazy: when the package isn't
installed, `MaxGraphBackend()` raises a precise `ImportError`, but
`plan_to_graph_spec(...)` still returns the abstract plan description
so callers can inspect what *would* be emitted. Tests that need the
runtime call `require_max_graph()` and skip when it's missing.

The translation per plan step (mirrors JAX's algorithm at
`jax/_src/numpy/lax_numpy.py:3264-3293`):

  Classify dims into B / K / M / N buckets:
    - B: batch - appears in lhs, rhs, and output
    - K: contract - appears in lhs and rhs, not in output
    - M: free-left - in lhs and output only
    - N: free-right - in rhs and output only

  Permute lhs to `(*B, *M, *K)` and rhs to `(*B, *K, *N)`.
  Issue `matmul` -> result has shape `(*B, *M, *N)`.
  Permute back to the equation's stated output order.

Unary steps (trace, diagonal, axis-sum, transpose) are emitted in the
spec. The current executable MAX path supports axis sums and
transposes; diagonal extraction is still a reference-backend case.
"""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass, field
from typing import cast

import numpy as np


def is_available() -> bool:
  """Return True when the `max.graph` Python package is importable.

  `importlib.util.find_spec("max.graph")` raises ModuleNotFoundError
  when the parent `max` package is missing (Python >=3.4 quirk), so we
  swallow that and return False.
  """
  try:
    return importlib.util.find_spec("max.graph") is not None
  except ModuleNotFoundError:
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

  Each list holds *label characters* (single-char strings) in the order
  they appear in the source operand. ``batch`` is intentionally the
  intersection-in-order on lhs; the per-axis index in rhs may differ
  and is recorded in `rhs_batch_perm` so reshape-only paths can still
  go through.
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

  Each entry is a `(op_kind, payload)` pair. `op_kind` is a string  - 
  `"matmul"`, `"transpose"`, `"reduce_sum"`, `"diagonal"`. The payload
  is a dict whose schema depends on the op_kind. This keeps the spec
  hashable / testable / serializable without depending on max.graph.
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

  This is the spec a real `max.graph` lowering would consume - each
  entry is the kind + parameters of a single graph op. Independent of
  `max.graph` itself; tests can validate the structure without the
  package installed.

  Limitations matching v0.1 scope:
    - no ellipsis (raises)
    - repeated labels within one operand (trace/diag) are emitted as
      a `"diagonal"` unary step
    - unary steps in `path` are passed through; reduce-out labels
      become `"reduce_sum"` ops
  """
  from . import _native  # noqa: PLC0415

  # Mojo raises `Error` which the std.python bindings surface as a bare
  # `Exception` (not RuntimeError). Catch broadly and re-raise the one
  # user-actionable subset (ellipsis) as ValueError so callers can pin a
  # narrow except clause.
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

  `__init__()` validates that `max.graph` is importable. Per-call
  execution is `execute(eq, shapes, path, operands)` - the same
  signature any future native backend will land on.

  v0.1 status: executable for the same BMM-lowerable subset as
  `backend="max"`. The plan-to-graph translation
  (`plan_to_graph_spec`) is Mojo-owned and tested; the actual graph
  object construction still uses MAX's Python API.
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
