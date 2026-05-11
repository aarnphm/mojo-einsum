"""P1 smoke tests — exercise parser + reference backend against numpy.einsum.

These are intentionally narrow: they validate that the P1 stupid einsum
returns numerically identical results to numpy on a curated set of
equations. The full ≥150-case parity suite arrives in P8.
"""

from __future__ import annotations

import numpy as np
import pytest

import mojo_einsum


# ─────────────────────────────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────────────────────────────


def test_parse_basic():
    ir = mojo_einsum.parse_equation("ij,jk->ik")
    assert ir["n_labels"] == 3
    assert ir["has_explicit_output"] is True
    assert len(ir["inputs"]) == 2
    assert ir["output"] != []


def test_parse_implicit_output():
    ir = mojo_einsum.parse_equation("ij,jk")
    assert ir["has_explicit_output"] is False
    # Implicit output is sorted unique non-repeated labels: i and k.
    assert len(ir["output"]) == 2


def test_parse_trace():
    ir = mojo_einsum.parse_equation("ii->")
    assert ir["has_explicit_output"] is True
    assert ir["output"] == []


def test_parse_ellipsis():
    ir = mojo_einsum.parse_equation("...ij,jk->...ik")
    # Pre-expansion: the ellipsis label is -1.
    assert -1 in ir["inputs"][0]
    assert -1 in ir["output"]


def test_parse_invalid_chars():
    with pytest.raises(Exception):
        mojo_einsum.parse_equation("i$j,jk->ik")


def test_parse_double_dot():
    with pytest.raises(Exception):
        mojo_einsum.parse_equation("..ij,jk->ik")


# ─────────────────────────────────────────────────────────────────────
# Reference backend — numpy parity
# ─────────────────────────────────────────────────────────────────────

# Canonical curated cases. Each row: (eq, shapes). Operands are seeded
# from np.random.default_rng(0).standard_normal for reproducibility.
_CASES = [
    # ── single-operand ──
    ("ii->", [(4, 4)]),                      # trace
    ("ii->i", [(4, 4)]),                     # diagonal
    ("i->", [(7,)]),                         # 1D sum
    ("ij->", [(3, 5)]),                      # full sum
    ("ij->ji", [(3, 5)]),                    # transpose
    ("ijk->ikj", [(2, 3, 4)]),               # 3D transpose
    ("ijk->", [(2, 3, 4)]),                  # 3D full sum
    # ── two-operand ──
    ("i,i->", [(5,), (5,)]),                 # inner product
    ("i,j->ij", [(3,), (4,)]),               # outer product
    ("ij,j->i", [(3, 5), (5,)]),             # matvec
    ("ij,jk->ik", [(3, 5), (5, 4)]),         # matmul
    ("bij,bjk->bik", [(2, 3, 5), (2, 5, 4)]),# batched matmul
    ("ij,ij->", [(3, 5), (3, 5)]),           # double contraction (Frobenius)
    ("ij,ji->", [(3, 5), (5, 3)]),           # trace of product
    # ── multi-operand ──
    ("ij,jk,kl->il", [(2, 3), (3, 4), (4, 5)]),
    ("ij,jk,kl,lm->im", [(2, 3), (3, 4), (4, 5), (5, 6)]),
]


@pytest.mark.parametrize(("eq", "shapes"), _CASES)
def test_numpy_parity(eq: str, shapes: list[tuple[int, ...]]):
    rng = np.random.default_rng(0)
    arrays = [rng.standard_normal(s) for s in shapes]

    expected = np.einsum(eq, *arrays, optimize=True)
    actual = mojo_einsum.einsum(eq, *arrays)

    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


# ─────────────────────────────────────────────────────────────────────
# einsum_path
# ─────────────────────────────────────────────────────────────────────


def test_path_naive_left_to_right():
    # 3 operands → 2 pairwise steps. Naive ordering pairs (0,1) then
    # (0,1) again (accumulator at slot 0, next operand at slot 1).
    path = mojo_einsum.einsum_path("ij,jk,kl->il", (2, 3), (3, 4), (4, 5))
    assert len(path) == 2
    assert path[0] == (0, 1)
    assert path[1] == (0, 1)


def test_path_single_operand():
    path = mojo_einsum.einsum_path("ii->", (4, 4))
    # Single operand: one unary step.
    assert len(path) == 1
    assert path[0] == (0,)
