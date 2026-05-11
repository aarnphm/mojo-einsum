"""JAX einsum-corpus parity (mirrors `~/workspace/jax/tests/lax_numpy_einsum_test.py`).

This is the P6 acceptance gate the plan calls for: "Tests against full
JAX einsum test suite (target >=80% case parity)." The corpus mirrors
JAX's hand-authored test cases plus the parametrized dask-derived list
JAX inherits from `https://github.com/dask/dask/pull/3412`.

Cases cover:
  - 1/2/3/4/5-operand contractions
  - leading, trailing, middle, and full-axis ellipses
  - repeated labels (trace, diagonal, multi-axis diagonal)
  - mixed trace + matmul
  - broadcast (size-1 dim matching against larger)
  - rank-6 dense contractions (`abcdef,bcdfg->abcdeg`)

Every case runs numpy.einsum with optimize=True as the oracle. atol=1e-9
on fp64 - the reference backend uses scalar reductions so we don't pay
the bf16 sqrtK drift.
"""

from __future__ import annotations

import moeinsum
import numpy as np
import pytest

# ---------------------------------------------------------------------
# JAX hand-authored cases (lax_numpy_einsum_test.py:40-218)
# ---------------------------------------------------------------------

# Format: (label, equation, [(shape, ...)]). The label keeps pytest's
# parametrize IDs human-readable.
_JAX_HAND = [
  # three_operands
  ("three_outer", "i,j,k->ijk", [(3,), (4,), (5,)]),
  # two_operands
  ("matvec", "ij,j->i", [(3, 4), (4,)]),
  ("reduce_contract", "ijk,j->i", [(3, 4, 5), (4,)]),
  ("repeated_lhs_contract", "iji,i->j", [(3, 4, 3), (3,)]),
  ("frobenius", "ij,ij->", [(3, 4), (3, 4)]),
  ("bmm_single_batch", "nij,jk->nik", [(10, 2, 3), (3, 4)]),
  ("broadcast_size_one", "sa,shb->shab", [(2, 1), (2, 3, 4)]),
  # one_operand
  ("axis_reduce_to_middle", "ijk->j", [(3, 4, 5)]),
  ("axis_permute_kij", "ijk->kij", [(3, 4, 5)]),
  ("axis_permute_drop_ki", "ijk->ki", [(3, 4, 5)]),
  ("ellipsis_last_two_swap", "...ijk->...ki", [(2, 3, 4, 5)]),
  ("ellipsis_collapse_to_ki", "...ijk->ki", [(3, 4, 5)]),
  ("trace_simple", "ii->", [(3, 3)]),
  ("sum_2d", "ij->", [(3, 3)]),
  ("triple_diag_to_scalar", "iii->", [(3, 3, 3)]),
  ("diag_keep_i", "ii->i", [(3, 3)]),
  ("diag_with_extra", "iij->i", [(3, 3, 4)]),
  ("triple_diag_keep_i", "iii->i", [(3, 3, 3)]),
  ("two_diagonals_reduce", "iijkk->i", [(3, 3, 5, 4, 4)]),
  ("two_diagonals_keep_ik", "iijkk->ik", [(3, 3, 5, 4, 4)]),
  ("diag_with_reduce_il", "iijkl->il", [(3, 3, 5, 4, 4)]),
  ("identity_2d", "ij->ij", [(3, 3)]),
  # tf-unsupported (JAX supports these - we must too)
  ("trailing_ellipsis_matmul", "ij...,jk...->ik...", [(2, 3, 5, 1), (3, 4, 5, 1)]),
  ("diag_lhs_outer", "ijj,k->ik", [(2, 3, 3), (4,)]),
  ("three_op_partial_outer", "ij,ij,jk->ik", [(2, 3), (2, 3), (3, 4)]),
]


@pytest.mark.parametrize(("label", "eq", "shapes"), _JAX_HAND, ids=lambda x: x if isinstance(x, str) else None)
def test_jax_hand_corpus(label: str, eq: str, shapes: list[tuple[int, ...]]) -> None:
  rng = np.random.default_rng(0)
  arrays = [rng.standard_normal(s) for s in shapes]
  expected = np.einsum(eq, *arrays, optimize=True)
  actual = moeinsum.einsum(eq, *arrays)
  np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-9)


# ---------------------------------------------------------------------
# JAX dask-derived corpus (lax_numpy_einsum_test.py:220-256)
# ---------------------------------------------------------------------


def _shapes_for(eq: str) -> list[tuple[int, ...]]:
  """Build shapes that satisfy the equation's label-size constraints.

  Walk each operand, assign each label a small dim if unseen, reuse
  the previously-assigned dim otherwise. Ellipsis '...' is mapped to
  a single broadcast dim of size 2.
  """
  inputs_part = eq.split("->")[0]
  pieces = inputs_part.split(",")
  size_for: dict[str, int] = {}
  default_sizes = (3, 4, 5, 6, 7, 8, 9)
  next_size = iter(default_sizes)
  shapes: list[tuple[int, ...]] = []
  for piece in pieces:
    dims: list[int] = []
    i = 0
    while i < len(piece):
      if piece[i] == ".":
        # consume ...
        dims.append(2)  # ellipsis stand-in
        i += 3
        continue
      label = piece[i]
      if label not in size_for:
        try:
          size_for[label] = next(next_size)
        except StopIteration:
          size_for[label] = 3
      dims.append(size_for[label])
      i += 1
    shapes.append(tuple(dims))
  return shapes


_DASK_EQUATIONS = [
  "abc,bad->abcd",
  "abcdef,bcdfg->abcdeg",
  "ea,fb,abcd,gc,hd->efgh",
  "ab,b",
  "aa",
  "a,a->",
  "a,a->a",
  "a,a",
  "a,b",
  "a,b,c",
  "a",
  "ba,b",
  "ba,b->",
  "defab,fedbc->defac",
  "ab...,bc...->ac...",
  "a...a",
  "abc...->cba...",
  "...ab->...a",
  "a...a->a...",
  "...abc,...abcd->...d",
  "ab...,b->ab...",
  "aa->a",
  "ab,ab,c->c",
  "aab,bc->ac",
  "aab,bcc->ac",
  "fdf,cdd,ccd,afe->ae",
  "fff,fae,bef,def->abd",
]


@pytest.mark.parametrize("eq", _DASK_EQUATIONS, ids=lambda eq: eq)
def test_jax_dask_corpus_fp64(eq: str) -> None:
  shapes = _shapes_for(eq)
  rng = np.random.default_rng(0)
  arrays = [rng.standard_normal(s) for s in shapes]
  expected = np.einsum(eq, *arrays, optimize=True)
  actual = moeinsum.einsum(eq, *arrays)
  np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-9)


# ---------------------------------------------------------------------
# Integer-dtype subset (bit-exact)
# ---------------------------------------------------------------------


_INT_SAFE_EQUATIONS = [
  "abc,bad->abcd",
  "ab,b",
  "aa",
  "a,b,c",
  "ab...,bc...->ac...",
  "aab,bc->ac",
  "fdf,cdd,ccd,afe->ae",
]


@pytest.mark.parametrize("eq", _INT_SAFE_EQUATIONS, ids=lambda eq: eq)
def test_jax_dask_corpus_int64(eq: str) -> None:
  shapes = _shapes_for(eq)
  rng = np.random.default_rng(0)
  arrays = [rng.integers(-3, 4, size=s, dtype=np.int64) for s in shapes]
  expected = np.einsum(eq, *arrays, optimize=True)
  actual = moeinsum.einsum(eq, *arrays)
  np.testing.assert_array_equal(actual, expected)
