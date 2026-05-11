"""Backend stub tests - exercise the architectural seam.

`MaxGraphBackend` is a stretch deliverable (P14). The Python-side
plan-to-graph translation (`plan_to_graph_spec`) is real and shipped;
the `max.graph`-side codegen is gated on the optional `max` package
being installed. Tests cover:

  - availability detection matches `importlib.util.find_spec`
  - `__init__` raises when `max.graph` is missing
  - `plan_to_graph_spec` produces structurally correct op lists for
    the canonical einsum shapes (BMM, matmul, trace, reduce, identity)
  - the dim classifier matches JAX's B/K/M/N split on hand-known cases

`backend="max"` and its `max_graph` alias are executable for the
BMM-lowerable subset through MAX Graph.
"""

from __future__ import annotations

import importlib

import pytest
from moeinsum._max_graph import (
  MaxGraphBackend,
  classify_pair,
  is_available,
  plan_to_graph_spec,
  require_max_graph,
)

# ---------------------------------------------------------------------
# Availability + error surface
# ---------------------------------------------------------------------


def test_is_available_matches_import_spec() -> None:
  try:
    spec_present = importlib.util.find_spec("max.graph") is not None
  except ModuleNotFoundError:
    spec_present = False
  assert is_available() is spec_present


def test_require_max_graph_error_when_missing() -> None:
  if is_available():
    pytest.skip("max.graph installed; cannot test the missing-dep path")
  with pytest.raises(ImportError, match="MaxGraphBackend requires"):
    require_max_graph()


def test_max_graph_backend_init_requires_max() -> None:
  if is_available():
    # The real package is installed - __init__ should succeed.
    backend = MaxGraphBackend()
    assert backend is not None
  else:
    with pytest.raises(ImportError, match="MaxGraphBackend requires"):
      MaxGraphBackend()


# ---------------------------------------------------------------------
# Dim classification (JAX-mirroring)
# ---------------------------------------------------------------------


def test_classify_pair_matmul() -> None:
  """`ij,jk->ik`: free_lhs=[i], contract=[j], free_rhs=[k], batch=[]."""
  c = classify_pair("ij", "jk", "ik")
  assert c.batch == ()
  assert c.contract == ("j",)
  assert c.free_lhs == ("i",)
  assert c.free_rhs == ("k",)


def test_classify_pair_batched_matmul() -> None:
  """`bij,bjk->bik`: batch=[b], contract=[j], free_lhs=[i], free_rhs=[k]."""
  c = classify_pair("bij", "bjk", "bik")
  assert c.batch == ("b",)
  assert c.contract == ("j",)
  assert c.free_lhs == ("i",)
  assert c.free_rhs == ("k",)


def test_classify_pair_frobenius() -> None:
  """`ij,ij->`: contract=[i, j], free=[], batch=[]."""
  c = classify_pair("ij", "ij", "")
  assert c.batch == ()
  assert c.contract == ("i", "j")
  assert c.free_lhs == ()
  assert c.free_rhs == ()


def test_classify_pair_outer_product() -> None:
  """`i,j->ij`: contract=[], free_lhs=[i], free_rhs=[j]."""
  c = classify_pair("i", "j", "ij")
  assert c.batch == ()
  assert c.contract == ()
  assert c.free_lhs == ("i",)
  assert c.free_rhs == ("j",)


def test_classify_pair_label_order_follows_lhs() -> None:
  """When labels appear in different orders on lhs/rhs, the bucket
  ordering follows lhs for B/K/M and rhs for N - matches numpy.einsum.
  """
  # batch label `b` listed second on lhs but first on rhs.
  c = classify_pair("ab", "ba", "ab")
  assert c.batch == ("a", "b")
  assert c.contract == ()
  assert c.free_lhs == ()
  assert c.free_rhs == ()


# ---------------------------------------------------------------------
# plan_to_graph_spec - structural validation
# ---------------------------------------------------------------------


def test_spec_matmul_one_step() -> None:
  spec = plan_to_graph_spec("ij,jk->ik", [(2, 3), (3, 4)], [(0, 1)])
  assert len(spec.ops) == 1
  kind, payload = spec.ops[0]
  assert kind == "matmul"
  assert payload["lhs_labels"] == "ij"
  assert payload["rhs_labels"] == "jk"
  assert payload["out_labels"] == "ik"
  assert payload["batch"] == ()
  assert payload["contract"] == ("j",)


def test_spec_three_operand_chain() -> None:
  """`ij,jk,kl->il` with greedy/optimal path = [(1,2), (0,1)] emits two
  matmuls and no trailing transpose."""
  spec = plan_to_graph_spec(
    "ij,jk,kl->il",
    [(100, 1), (1, 100_000), (100_000, 1)],
    [(1, 2), (0, 1)],
  )
  kinds = [k for k, _ in spec.ops]
  assert kinds == ["matmul", "matmul"]
  # First contraction is `jk,kl->jl`; second is `ij,jl->il`.
  assert spec.ops[0][1]["out_labels"] == "jl"
  assert spec.ops[1][1]["out_labels"] == "il"


def test_spec_unary_trace() -> None:
  """`ii->` emits a diagonal then a reduce_sum (collapse to scalar)."""
  spec = plan_to_graph_spec("ii->", [(4, 4)], [(0,)])
  kinds = [k for k, _ in spec.ops]
  # diagonal collapses 'ii' to 'i'; reduce_sum drops 'i' to ''.
  assert kinds == ["diagonal", "reduce_sum"]


def test_spec_unary_full_sum() -> None:
  spec = plan_to_graph_spec("ij->", [(3, 5)], [(0,)])
  kinds = [k for k, _ in spec.ops]
  assert kinds == ["reduce_sum"]
  assert spec.ops[0][1]["dst_labels"] == ""


def test_spec_trailing_transpose_when_axis_order_differs() -> None:
  """`ji,jk->ik` requires the path's final intermediate to be transposed
  into the equation's stated output order."""
  spec = plan_to_graph_spec("ji,jk->ki", [(3, 2), (3, 5)], [(0, 1)])
  kinds = [k for k, _ in spec.ops]
  # The matmul itself produces 'ik'; we then transpose to 'ki'.
  assert kinds[0] == "matmul"
  assert spec.ops[0][1]["out_labels"] == "ik"
  assert kinds[-1] == "transpose"
  assert spec.ops[-1][1]["dst_labels"] == "ki"


def test_spec_ellipsis_unsupported() -> None:
  with pytest.raises(ValueError, match="ellipsis"):
    plan_to_graph_spec("...ij,jk->...ik", [(2, 3, 4), (4, 5)], [(0, 1)])


def test_spec_unary_identity_emits_no_ops() -> None:
  """`ij->ij` on a 2D operand with the natural axis order - the unary
  step has no reduce-outs, no repeats, and matches `final_output`,
  so the spec is empty."""
  spec = plan_to_graph_spec("ij->ij", [(3, 5)], [(0,)])
  assert spec.ops == []


# ---------------------------------------------------------------------
# Planned-but-unimplemented backend dispatch
# ---------------------------------------------------------------------


def test_einsum_native_backend_phase_aware_error() -> None:
  """`backend="native"` is planned (P11/P12) but not wired. Callers
  should get a NotImplementedError naming the phase and the skeleton
  file, not an opaque "unknown backend" string."""
  import moeinsum
  import numpy as np

  with pytest.raises(NotImplementedError, match="P11"):
    moeinsum.einsum("ij,jk->ik", np.eye(3), np.eye(3), backend="native")


def test_einsum_max_backend_matmul_when_available() -> None:
  """`backend="max:cpu"` runs through MAX Graph for the supported subset."""
  import moeinsum
  import numpy as np

  if not is_available():
    pytest.skip("max.graph not installed")

  a = np.arange(12, dtype=np.float32).reshape(3, 4)
  b = np.arange(20, dtype=np.float32).reshape(4, 5)
  actual = moeinsum.einsum("ij,jk->ik", a, b, backend="max:cpu")
  np.testing.assert_allclose(actual, a @ b, atol=1e-5, rtol=1e-5)


def test_einsum_max_backend_repeated_labels_not_supported_yet() -> None:
  """Diagonal/trace lowering is still outside the executable MAX subset."""
  import moeinsum
  import numpy as np

  if not is_available():
    pytest.skip("max.graph not installed")

  with pytest.raises(NotImplementedError, match="repeated labels"):
    moeinsum.einsum("ii->", np.eye(3, dtype=np.float32), backend="max:cpu")


def test_einsum_max_graph_alias_when_available() -> None:
  """`backend="max_graph"` is a compatibility alias for the MAX Graph path."""
  import moeinsum
  import numpy as np

  if not is_available():
    pytest.skip("max.graph not installed")

  a = np.eye(3, dtype=np.float32)
  actual = moeinsum.einsum("ij,jk->ik", a, a, backend="max_graph")
  np.testing.assert_allclose(actual, a @ a, atol=1e-5, rtol=1e-5)


def test_einsum_unknown_backend_value_error() -> None:
  """Backends that aren't even in the plan still get the legacy
  ValueError surface, not a phase-aware NotImplementedError."""
  import moeinsum
  import numpy as np

  with pytest.raises(ValueError, match="unknown backend"):
    moeinsum.einsum("ij,jk->ik", np.eye(3), np.eye(3), backend="quantum")
