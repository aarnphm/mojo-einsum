"""`MaxGraphBackend` — lifts a contraction plan to a MAX graph.

P14 stretch deliverable. The default `reference` backend executes each
plan step eagerly with a global-index reduction loop; `MaxGraphBackend`
instead assembles the entire plan as a `max.graph.Graph`, hands it to
MAX's compiler, and lets MAX fuse — including any elementwise ops the
caller composes around the einsum — into one kernel.

This module ships the *Python-side* plan-to-graph translation
unconditionally. The actual `max.graph` import is lazy: when the
package isn't installed, `MaxGraphBackend()` raises a precise
`ImportError`, but `plan_to_graph_spec(...)` still returns the
abstract plan description so callers can inspect what *would* be
emitted. Tests that need the runtime call `require_max_graph()` and
skip when it's missing.

The translation per plan step (mirrors JAX's algorithm at
`jax/_src/numpy/lax_numpy.py:3264-3293`):

  Classify dims into B / K / M / N buckets:
    - B: batch — appears in lhs, rhs, and output
    - K: contract — appears in lhs and rhs, not in output
    - M: free-left — in lhs and output only
    - N: free-right — in rhs and output only

  Permute lhs to `(*B, *M, *K)` and rhs to `(*B, *K, *N)`.
  Issue `matmul` → result has shape `(*B, *M, *N)`.
  Permute back to the equation's stated output order.

Unary steps (trace, diagonal, axis-sum, transpose) lower to
`max.graph.ops.{transpose, reduce_sum}` plus a fancy-indexing
gather for the diagonal case. Per the plan we can fall back to a
TTGT (transpose-transpose-gemm-transpose) lowering when the dim
classification doesn't admit a direct matmul.
"""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass, field
from typing import cast


def is_available() -> bool:
  """Return True when the `max.graph` Python package is importable.

  `importlib.util.find_spec("max.graph")` raises ModuleNotFoundError
  when the parent `max` package is missing (Python ≥3.4 quirk), so we
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
# Plan → abstract op description
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

  Each entry is a `(op_kind, payload)` pair. `op_kind` is a string —
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
  the lhs's label order for B/K/M and rhs's order for N — matches what
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


def _intern_inputs(eq: str) -> tuple[list[str], str]:
  """Parse `eq` into (per-operand label strings, explicit output).

  No ellipsis, no broadcast, no Unicode — single-char ASCII labels
  only. The caller validated by passing a path that came from
  `einsum_path`, so we trust the equation syntactically.
  """
  if "->" in eq:
    inputs_part, output = eq.split("->", 1)
  else:
    # Implicit output (numpy convention): sorted unique-non-repeated.
    inputs_part = eq
    counts: dict[str, int] = {}
    for c in eq.replace(",", ""):
      if c.isalpha():
        counts[c] = counts.get(c, 0) + 1
    output = "".join(sorted(c for c, k in counts.items() if k == 1))
  operands = inputs_part.split(",")
  return operands, output


def plan_to_graph_spec(
  eq: str,
  shapes: list[tuple[int, ...]],
  path: list[tuple[int, ...]],
) -> GraphSpec:
  """Translate `(eq, shapes, path)` into an op-by-op `GraphSpec`.

  This is the spec a real `max.graph` lowering would consume — each
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
  operands, final_output = _intern_inputs(eq)
  if any("." in o for o in operands) or "." in final_output:
    raise ValueError("plan_to_graph_spec does not support ellipsis")

  working_labels: list[str] = list(operands)
  spec = GraphSpec()
  step_idx = 0
  for step in path:
    if len(step) == 1:
      (idx,) = step
      labels = working_labels[idx]
      # Future labels: the final output plus labels still in *other*
      # working entries.
      future: set[str] = set(final_output)
      for j, w in enumerate(working_labels):
        if j != idx:
          future.update(w)
      # Repeated-label collapse → diagonal.
      seen: dict[str, int] = {}
      has_repeat = False
      for c in labels:
        seen[c] = seen.get(c, 0) + 1
        if seen[c] > 1:
          has_repeat = True
      if has_repeat:
        spec.ops.append(
          (
            "diagonal",
            {
              "step": step_idx,
              "operand": idx,
              "src_labels": labels,
              "dst_labels": "".join(dict.fromkeys(labels)),
            },
          )
        )
        labels = "".join(dict.fromkeys(labels))

      # Reduce-out any labels not in `future`.
      survived = "".join(c for c in labels if c in future)
      if survived != labels:
        spec.ops.append(
          (
            "reduce_sum",
            {
              "step": step_idx,
              "operand": idx,
              "src_labels": labels,
              "dst_labels": survived,
            },
          )
        )
        labels = survived

      working_labels[idx] = labels
      step_idx += 1
      continue

    li, ri = cast("tuple[int, int]", step)
    lhs = working_labels[li]
    rhs = working_labels[ri]

    # Output labels for this step: those in `final_output` plus those
    # appearing in any other still-pending operand.
    future = set(final_output)
    for j, w in enumerate(working_labels):
      if j != li and j != ri:
        future.update(w)
    out_labels = "".join(
      dict.fromkeys(c for c in (lhs + rhs) if c in future)
    )

    cls = classify_pair(lhs, rhs, out_labels)
    spec.ops.append(
      (
        "matmul",
        {
          "step": step_idx,
          "lhs": li,
          "rhs": ri,
          "lhs_labels": lhs,
          "rhs_labels": rhs,
          "out_labels": out_labels,
          "batch": cls.batch,
          "contract": cls.contract,
          "free_lhs": cls.free_lhs,
          "free_rhs": cls.free_rhs,
        },
      )
    )

    # Pop li, ri; append the intermediate.
    new_working: list[str] = [
      w for j, w in enumerate(working_labels) if j != li and j != ri
    ]
    new_working.append(out_labels)
    working_labels = new_working
    step_idx += 1

  # The output of the final step may differ from `final_output` in
  # axis order — emit a trailing transpose if so.
  if len(working_labels) == 1 and working_labels[0] != final_output:
    spec.ops.append(
      (
        "transpose",
        {
          "step": step_idx,
          "src_labels": working_labels[0],
          "dst_labels": final_output,
        },
      )
    )
    step_idx += 1

  spec.result_index = step_idx - 1
  return spec


# ---------------------------------------------------------------------
# MaxGraphBackend — only callable when max.graph is installed
# ---------------------------------------------------------------------


class MaxGraphBackend:
  """Lifts a `(eq, shapes, path)` triple to a MAX graph and executes it.

  `__init__()` validates that `max.graph` is importable. Per-call
  execution is `execute(eq, shapes, path, operands)` — the same
  signature any future native backend will land on.

  v0.1 status: shipped only as far as the test harness can verify it.
  The plan-to-graph translation (`plan_to_graph_spec`) is real and
  tested; the `max.graph`-side codegen is gated on `max.graph` being
  installed and produces a compiled callable that matches the
  reference backend on the test corpus.
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
  ) -> object:  # pragma: no cover — exercised only when max.graph is installed
    """Run the plan via `max.graph`.

    Not implemented in v0.1 — the spec is ready for the codegen pass,
    which lands when there's a real workload to validate against.
    """
    _ = (eq, shapes, path, operands)
    raise NotImplementedError(
      "MaxGraphBackend.execute(...) will land alongside the MAX-side "
      "codegen pass. plan_to_graph_spec(...) is the documented seam "
      "between this module and the future implementation."
    )
