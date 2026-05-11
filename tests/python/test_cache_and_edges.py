"""JIT plan-cache effectiveness + edge cases.

The cache is keyed by (equation, shape-tuple, optimize). Repeat calls
must short-circuit the FFI. Edge cases cover zero-size dims, single
labels, and the documented error surface.
"""

from __future__ import annotations

import moeinsum
import numpy as np
import pytest

from moeinsum import PLAN_CACHE


def test_cache_hit_short_circuits_ffi() -> None:
  PLAN_CACHE.clear()
  shapes = ((3, 4), (4, 5), (5, 6))
  assert len(PLAN_CACHE) == 0

  first = moeinsum.einsum_path("ij,jk,kl->il", *shapes, optimize="optimal")
  assert len(PLAN_CACHE) == 1

  second = moeinsum.einsum_path("ij,jk,kl->il", *shapes, optimize="optimal")
  assert first == second
  assert len(PLAN_CACHE) == 1  # no new entry — cache hit


def test_cache_distinguishes_optimize() -> None:
  PLAN_CACHE.clear()
  shapes = ((3, 4), (4, 5), (5, 6))

  moeinsum.einsum_path("ij,jk,kl->il", *shapes, optimize="naive")
  moeinsum.einsum_path("ij,jk,kl->il", *shapes, optimize="greedy")
  # Different optimize → distinct cache entries.
  assert len(PLAN_CACHE) == 2


def test_cache_distinguishes_shapes() -> None:
  PLAN_CACHE.clear()
  moeinsum.einsum_path("ij,jk,kl->il", (3, 4), (4, 5), (5, 6))
  moeinsum.einsum_path("ij,jk,kl->il", (3, 4), (4, 5), (5, 7))
  assert len(PLAN_CACHE) == 2


# ─────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────


def test_zero_operand_raises() -> None:
  with pytest.raises(ValueError, match="at least one operand"):
    moeinsum.einsum("ij->i")


def test_unknown_backend_raises() -> None:
  a = np.eye(2)
  with pytest.raises(ValueError, match="unknown backend"):
    moeinsum.einsum("ij,jk->ik", a, a, backend="cuda")  # type: ignore[arg-type]


def test_unknown_optimize_raises() -> None:
  a = np.eye(2)
  with pytest.raises(ValueError, match="unknown optimize"):
    moeinsum.einsum("ij,jk->ik", a, a, optimize="branch-2")  # type: ignore[arg-type]


def test_size_one_dim() -> None:
  # Degenerate batch of size 1 — has bitten broadcasting implementations.
  a = np.arange(6.0).reshape(1, 2, 3)
  b = np.arange(15.0).reshape(1, 3, 5)
  out = moeinsum.einsum("bij,bjk->bik", a, b)
  np.testing.assert_allclose(out, a @ b)


def test_repeated_index_with_trace() -> None:
  # `iij,jk->ik` — diagonal-then-contract. JAX gets this right; ensure
  # we do too.
  a = np.arange(8.0).reshape(2, 2, 2)
  b = np.arange(6.0).reshape(2, 3)
  expected = np.einsum("iij,jk->ik", a, b, optimize=True)
  actual = moeinsum.einsum("iij,jk->ik", a, b)
  np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_repeated_indices_only_in_input() -> None:
  # `ii->` collapses to scalar.
  a = np.array([[1.0, 2.0], [3.0, 4.0]])
  out = moeinsum.einsum("ii->", a)
  assert out == 5.0  # 1 + 4
