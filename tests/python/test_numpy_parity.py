"""Comprehensive numpy-parity suite.

Curated from JAX's `tests/lax_numpy_einsum_test.py` plus a handful of
tensor-network shapes. Every case runs against `numpy.einsum(eq, *ops,
optimize=True)` and the moeinsum reference backend; results must
match within atol=1e-10 (fp64).

This is the P2-polish + P6 multi-operand acceptance suite. Path
optimizer variants (greedy / optimal / auto) are also exercised
end-to-end so a regression in `path.mojo` shows up here as a result
mismatch even though the chosen path differs.
"""

from __future__ import annotations

import moeinsum
import numpy as np
import pytest


def _rand(shape: tuple[int, ...], seed: int = 0) -> np.ndarray:
  rng = np.random.default_rng(seed)
  return rng.standard_normal(shape).astype(np.float64)


# Each entry: (label, equation, shape sequence).
SINGLE_OPERAND = [
  ("trace_4", "ii->", [(4, 4)]),
  ("trace_1", "ii->", [(1, 1)]),
  ("diag_4", "ii->i", [(4, 4)]),
  ("sum_1d", "i->", [(7,)]),
  ("sum_2d", "ij->", [(3, 5)]),
  ("sum_3d", "ijk->", [(2, 3, 4)]),
  ("transpose_2d", "ij->ji", [(3, 5)]),
  ("transpose_3d_120", "ijk->jki", [(2, 3, 4)]),
  ("transpose_3d_210", "ijk->kji", [(2, 3, 4)]),
  ("identity_2d", "ij->ij", [(3, 5)]),
  ("partial_sum_3d", "ijk->ij", [(2, 3, 4)]),
  ("partial_sum_3d_to_i", "ijk->i", [(2, 3, 4)]),
  ("partial_sum_3d_to_k", "ijk->k", [(2, 3, 4)]),
  ("diag_3d", "iji->ij", [(3, 4, 3)]),  # repeated i, project to (i, j)
  ("trace_diag", "iij->j", [(3, 3, 5)]),
]


TWO_OPERAND = [
  ("inner_product", "i,i->", [(5,), (5,)]),
  ("outer_product", "i,j->ij", [(3,), (4,)]),
  ("matvec", "ij,j->i", [(3, 5), (5,)]),
  ("vecmat", "i,ij->j", [(3,), (3, 5)]),
  ("matmul_small", "ij,jk->ik", [(3, 5), (5, 4)]),
  ("matmul_square", "ij,jk->ik", [(8, 8), (8, 8)]),
  ("matmul_transpose_b", "ij,kj->ik", [(3, 5), (4, 5)]),
  ("batched_matmul", "bij,bjk->bik", [(2, 3, 5), (2, 5, 4)]),
  ("batched_matmul_3d_batch", "abij,abjk->abik", [(2, 3, 4, 5), (2, 3, 5, 6)]),
  ("frobenius", "ij,ij->", [(3, 5), (3, 5)]),
  ("trace_of_product", "ij,ji->", [(3, 5), (5, 3)]),
  ("contract_inner_only", "ijk,jkl->il", [(2, 3, 4), (3, 4, 5)]),
  ("hadamard", "ij,ij->ij", [(3, 5), (3, 5)]),
  ("outer_3d_2d", "ijk,kl->ijl", [(2, 3, 4), (4, 5)]),
  ("broadcast_via_repeat", "ij,ij->i", [(3, 5), (3, 5)]),
  ("batch_inner", "bi,bi->b", [(7, 5), (7, 5)]),
  ("batch_outer", "bi,bj->bij", [(7, 3), (7, 4)]),
  ("attention_scores", "bhqd,bhkd->bhqk", [(2, 4, 6, 8), (2, 4, 5, 8)]),
  ("attention_value_combine", "bhqk,bhkv->bhqv", [(2, 4, 6, 5), (2, 4, 5, 8)]),
]


MULTI_OPERAND = [
  ("three_chain_sym", "ij,jk,kl->il", [(3, 4), (4, 5), (5, 6)]),
  ("four_chain", "ij,jk,kl,lm->im", [(2, 3), (3, 4), (4, 5), (5, 6)]),
  ("five_chain", "ij,jk,kl,lm,mn->in", [(2, 3), (3, 4), (4, 5), (5, 4), (4, 3)]),
  ("bellman_chain", "ij,jk,kl->il", [(8, 2), (2, 10), (10, 3)]),  # asymmetric
  ("three_with_batch", "bij,bjk,bkl->bil", [(2, 3, 4), (2, 4, 5), (2, 5, 6)]),
  ("triangle", "ab,bc,ca->", [(3, 4), (4, 5), (5, 3)]),
  ("kronecker", "i,j,k->ijk", [(3,), (4,), (5,)]),
  ("inner_then_outer", "i,i,j->j", [(4,), (4,), (5,)]),
  ("tensor_train_4", "ai,aj,ak,al->ijkl", [(3, 2), (3, 2), (3, 2), (3, 2)]),
]


_ALL_CASES = SINGLE_OPERAND + TWO_OPERAND + MULTI_OPERAND


@pytest.mark.parametrize(
  ("label", "eq", "shapes"),
  _ALL_CASES,
  ids=[c[0] for c in _ALL_CASES],
)
@pytest.mark.parametrize("optimize", ["naive", "greedy", "optimal", "auto"])
def test_parity(label: str, eq: str, shapes: list[tuple[int, ...]], optimize: str) -> None:
  arrays = [_rand(s, seed=i) for i, s in enumerate(shapes)]
  expected = np.einsum(eq, *arrays, optimize=True)
  actual = moeinsum.einsum(eq, *arrays, optimize=optimize)
  np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


# ─────────────────────────────────────────────────────────────────────
# Output dtype handling
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
  "dtype",
  [np.float32, np.float64, np.int64],
)
def test_output_dtype(dtype) -> None:
  a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
  b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
  out = moeinsum.einsum("ij,jk->ik", a, b, dtype=dtype)
  assert out.dtype == np.dtype(dtype)
  np.testing.assert_allclose(out.astype(np.float64), a @ b, atol=1e-10)
