"""P1 smoke tests — exercise parser + reference backend against numpy.einsum.

These are intentionally narrow: they validate that the P1 stupid einsum
returns numerically identical results to numpy on a curated set of
equations. The full ≥150-case parity suite arrives in P8.
"""

from __future__ import annotations

import moeinsum
import numpy as np
import pytest

# ─────────────────────────────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────────────────────────────


def _assert_parse_error(eq: str) -> None:
  try:
    moeinsum.parse_equation(eq)
  except Exception as exc:
    assert "einsum parse error" in str(exc)
  else:
    pytest.fail(f"expected parse error for {eq!r}")


def test_parse_basic() -> None:
  ir = moeinsum.parse_equation("ij,jk->ik")
  assert ir["n_labels"] == 3
  assert ir["has_explicit_output"] is True
  assert len(ir["inputs"]) == 2
  assert ir["output"] != []


def test_parse_implicit_output() -> None:
  ir = moeinsum.parse_equation("ij,jk")
  assert ir["has_explicit_output"] is False
  # Implicit output is sorted unique non-repeated labels: i and k.
  assert len(ir["output"]) == 2


def test_parse_trace() -> None:
  ir = moeinsum.parse_equation("ii->")
  assert ir["has_explicit_output"] is True
  assert ir["output"] == []


def test_parse_ellipsis() -> None:
  ir = moeinsum.parse_equation("...ij,jk->...ik")
  # Pre-expansion: the ellipsis label is -1.
  assert -1 in ir["inputs"][0]
  assert -1 in ir["output"]


def test_parse_invalid_chars() -> None:
  _assert_parse_error("i$j,jk->ik")


def test_parse_double_dot() -> None:
  _assert_parse_error("..ij,jk->ik")


# ─────────────────────────────────────────────────────────────────────
# Reference backend — numpy parity
# ─────────────────────────────────────────────────────────────────────

# Canonical curated cases. Each row: (eq, shapes). Operands are seeded
# from np.random.default_rng(0).standard_normal for reproducibility.
_CASES = [
  # ── single-operand ──
  ("ii->", [(4, 4)]),  # trace
  ("ii->i", [(4, 4)]),  # diagonal
  ("i->", [(7,)]),  # 1D sum
  ("ij->", [(3, 5)]),  # full sum
  ("ij->ji", [(3, 5)]),  # transpose
  ("ijk->ikj", [(2, 3, 4)]),  # 3D transpose
  ("ijk->", [(2, 3, 4)]),  # 3D full sum
  # ── two-operand ──
  ("i,i->", [(5,), (5,)]),  # inner product
  ("i,j->ij", [(3,), (4,)]),  # outer product
  ("ij,j->i", [(3, 5), (5,)]),  # matvec
  ("ij,jk->ik", [(3, 5), (5, 4)]),  # matmul
  ("bij,bjk->bik", [(2, 3, 5), (2, 5, 4)]),  # batched matmul
  ("ij,ij->", [(3, 5), (3, 5)]),  # double contraction (Frobenius)
  ("ij,ji->", [(3, 5), (5, 3)]),  # trace of product
  # ── multi-operand ──
  ("ij,jk,kl->il", [(2, 3), (3, 4), (4, 5)]),
  ("ij,jk,kl,lm->im", [(2, 3), (3, 4), (4, 5), (5, 6)]),
]


@pytest.mark.parametrize(("eq", "shapes"), _CASES)
def test_numpy_parity(eq: str, shapes: list[tuple[int, ...]]) -> None:
  rng = np.random.default_rng(0)
  arrays = [rng.standard_normal(s) for s in shapes]

  expected = np.einsum(eq, *arrays, optimize=True)
  actual = moeinsum.einsum(eq, *arrays)

  np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


# ─────────────────────────────────────────────────────────────────────
# einsum_path
# ─────────────────────────────────────────────────────────────────────


def test_path_naive_left_to_right() -> None:
  # 3 operands → 2 pairwise steps. Naive ordering pairs (0,1) then
  # (0,1) again (accumulator at slot 0, next operand at slot 1).
  path = moeinsum.einsum_path("ij,jk,kl->il", (2, 3), (3, 4), (4, 5), optimize="naive")
  assert len(path) == 2
  assert path[0] == (0, 1)
  assert path[1] == (0, 1)


def test_path_optimize_is_honored() -> None:
  eq = "ab,cd,bc->ad"
  shapes = ((3, 4), (5, 6), (4, 5))

  naive = moeinsum.einsum_path(eq, *shapes, optimize="naive")
  greedy = moeinsum.einsum_path(eq, *shapes, optimize="greedy")
  optimal = moeinsum.einsum_path(eq, *shapes, optimize="optimal")
  auto = moeinsum.einsum_path(eq, *shapes, optimize="auto")

  assert naive == [(0, 1), (0, 1)]
  assert greedy == [(1, 2), (0, 1)]
  assert optimal == [(0, 2), (0, 1)]
  assert auto == optimal


def test_path_single_operand() -> None:
  path = moeinsum.einsum_path("ii->", (4, 4))
  # Single operand: one unary step.
  assert len(path) == 1
  assert path[0] == (0,)


def test_path_bellman_chain() -> None:
  # Classic matrix-chain demo from docs/notation.md: A:100×1, B:1×10^5,
  # C:10^5×1. Naive (AB)C costs ~2×10^7 flops with a huge intermediate;
  # A(BC) costs ~10^5. Greedy / optimal / auto must all pick A(BC) —
  # which in working-set indices is (1, 2) then (0, 1).
  shapes = ((100, 1), (1, 100_000), (100_000, 1))
  for algo in ("greedy", "optimal", "auto"):
    path = moeinsum.einsum_path("ij,jk,kl->il", *shapes, optimize=algo)
    assert path == [(1, 2), (0, 1)], f"{algo} chose {path}"
  # Naive stays left-to-right.
  assert moeinsum.einsum_path("ij,jk,kl->il", *shapes, optimize="naive") == [
    (0, 1),
    (0, 1),
  ]


def test_path_optimizer_known_invalid() -> None:
  with pytest.raises(ValueError, match="unknown optimize"):
    moeinsum.einsum_path("ij,jk->ik", (2, 3), (3, 4), optimize="branch-2")


def test_path_random_greedy_matches_optimal_on_bellman() -> None:
  # random-greedy should at least match greedy on easy cases.
  shapes = ((100, 1), (1, 100_000), (100_000, 1))
  rg = moeinsum.einsum_path("ij,jk,kl->il", *shapes, optimize="random-greedy")
  optimal = moeinsum.einsum_path("ij,jk,kl->il", *shapes, optimize="optimal")
  assert rg == optimal
