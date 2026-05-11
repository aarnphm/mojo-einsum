"""`MaxGraphBackend` skeleton — lifts a contraction plan to a MAX graph.

P14 stretch: the default `reference` and (forthcoming) `max` backends
execute each plan step eagerly. `MaxGraphBackend` instead assembles
the whole plan as a `max.graph.Graph`, hands it to MAX's compiler, and
returns a callable that fuses everything — including any elementwise
ops the caller composes around the einsum — into one kernel.

This file is intentionally a structural placeholder; the actual graph
construction lands when there's a real workload that benefits from
whole-graph fusion (training loop, repeated inference, etc.). For now
`is_available()` is the public surface — callers (e.g. a future
`einsum(..., backend="max_graph")` path) can detect availability and
fail with a precise error message rather than an ImportError stack.
"""

from __future__ import annotations

import importlib.util
from typing import Any


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


def require_max_graph() -> Any:
  """Import and return `max.graph` or raise a clear `ImportError`."""
  if not is_available():
    raise ImportError(
      "MaxGraphBackend requires the `max` Python package "
      "(install with `uv pip install -e '.[max]'` and re-import). "
      "See docs/comparisons.md for context."
    )
  import max.graph as max_graph

  return max_graph


class MaxGraphBackend:
  """Lifts a `ContractionPlan` to a MAX graph and executes via MAX.

  The Mojo side delivers `(plan, operands)` — a backend-agnostic
  `ContractionPlan` plus the per-operand numpy arrays. This Python
  shim translates each pairwise plan step to `max.graph.ops.matmul`
  with appropriate reshape / transpose ops, hands the whole graph
  to `max.graph.Graph.compile()`, and runs the result.

  Not implemented in v0.1 — `__init__` raises until the FFI exposes
  the plan handle and MAX is on the install path. The class exists to
  make the architectural seam concrete for downstream code that wants
  to thread a `backend=` argument through.
  """

  def __init__(self) -> None:
    raise NotImplementedError(
      "MaxGraphBackend is a stretch deliverable (P14). v0.1 ships the "
      "reference backend only; the max_kernels and native backends "
      "land first. See README.md roadmap."
    )

  def execute(self, plan: Any, operands: list[Any]) -> Any:  # pragma: no cover
    raise NotImplementedError
