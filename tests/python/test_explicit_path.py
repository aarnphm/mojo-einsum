"""Explicit caller-supplied path tests (P4 close-out).

`optimize=[(i, j), ...]` lets a caller skip the planner and pass a
literal contraction path - matches numpy.einsum / opt_einsum's API.
The path is validated against working-set semantics: each pairwise step
removes two operands and appends one result, each unary step is a no-op
on size, and the final working set must contain exactly one tensor.

For the reference backend, path order doesn't affect the numerical
result (it's a global-index loop), so the explicit path is primarily
useful for introspection and forward-compatibility with the
forthcoming MaxBackend.
"""

from __future__ import annotations

import moeinsum
import numpy as np
import pytest


def test_explicit_path_round_trips() -> None:
  """Passing the optimal path back as `optimize=...` returns it verbatim."""
  eq = "ab,cd,bc->ad"
  shapes = ((3, 4), (5, 6), (4, 5))
  optimal = moeinsum.einsum_path(eq, *shapes, optimize="optimal")
  echoed = moeinsum.einsum_path(eq, *shapes, optimize=optimal)
  assert echoed == optimal


def test_explicit_path_accepts_lists_of_lists() -> None:
  """Caller may pass `[[i, j], ...]` (the numpy/opt_einsum convention)."""
  eq = "ij,jk,kl->il"
  shapes = ((2, 3), (3, 4), (4, 5))
  path = moeinsum.einsum_path(eq, *shapes, optimize=[[0, 1], [0, 1]])
  assert path == [(0, 1), (0, 1)]


def test_explicit_path_returns_correct_einsum_result() -> None:
  """End-to-end: explicit path through einsum() matches numpy."""
  rng = np.random.default_rng(0)
  arrays = [
    rng.standard_normal((2, 3)),
    rng.standard_normal((3, 4)),
    rng.standard_normal((4, 5)),
  ]
  expected = np.einsum("ij,jk,kl->il", *arrays, optimize=True)
  # Naive (left-to-right) explicit path.
  actual = moeinsum.einsum("ij,jk,kl->il", *arrays, optimize=[(0, 1), (0, 1)])
  np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_explicit_path_rejects_out_of_range() -> None:
  with pytest.raises(ValueError, match="out of range"):
    moeinsum.einsum_path("ij,jk->ik", (2, 3), (3, 4), optimize=[(0, 5)])


def test_explicit_path_rejects_same_lhs_rhs() -> None:
  with pytest.raises(ValueError, match="both reference"):
    moeinsum.einsum_path("ij,jk,kl->il", (2, 3), (3, 4), (4, 5), optimize=[(1, 1), (0, 1)])


def test_explicit_path_rejects_bad_arity() -> None:
  with pytest.raises(ValueError, match="arity"):
    moeinsum.einsum_path("ij,jk->ik", (2, 3), (3, 4), optimize=[(0, 1, 2)])


def test_explicit_path_rejects_incomplete_path() -> None:
  """A 3-operand contraction needs 2 pairwise steps. One step leaves 2 tensors."""
  with pytest.raises(ValueError, match="leaves 2"):
    moeinsum.einsum_path("ij,jk,kl->il", (2, 3), (3, 4), (4, 5), optimize=[(0, 1)])


def test_explicit_path_single_operand() -> None:
  """A 1-operand einsum can take a single-element unary path."""
  path = moeinsum.einsum_path("ii->", (4, 4), optimize=[(0,)])
  assert path == [(0,)]


def test_explicit_path_bypasses_cache() -> None:
  """The LRU is keyed by the equation+shape+name; explicit paths skip it."""
  moeinsum.PLAN_CACHE.clear()
  eq = "ij,jk->ik"
  shapes = ((2, 3), (3, 4))
  moeinsum.einsum_path(eq, *shapes, optimize=[(0, 1)])
  # Cache should still be empty after an explicit-path call.
  cached = moeinsum.PLAN_CACHE.get(("__einsum_path__", eq, shapes, "auto"))
  assert cached is None
