"""Branch-family path optimizer tests (P4 polish).

`branch-all`, `branch-2`, `branch-1` are opt_einsum's best-first DFS
over the contraction tree, FLOP-pruned by an initial greedy seed.

  - `branch-1` collapses to greedy by construction (top-1 candidate at
    every level == what greedy picks).
  - `branch-2` keeps the top-2 candidates per level - finds non-greedy
    paths when they're nearby in the candidate ranking.
  - `branch-all` is the exhaustive DFS - guaranteed <= greedy FLOPs (and
    typically == optimal for n <= 6).

The greedy seed gives us "no worse than greedy" for free; we test that
invariant across a small zoo of contractions, plus a few hand-known
optimal results.
"""

from __future__ import annotations

import moeinsum

_BELLMAN_EQ = "ij,jk,kl->il"
_BELLMAN_SHAPES = ((100, 1), (1, 100_000), (100_000, 1))


def test_branch_all_matches_optimal_on_bellman() -> None:
  optimal = moeinsum.einsum_path(_BELLMAN_EQ, *_BELLMAN_SHAPES, optimize="optimal")
  branch_all = moeinsum.einsum_path(_BELLMAN_EQ, *_BELLMAN_SHAPES, optimize="branch-all")
  assert branch_all == optimal


def test_branch_2_matches_optimal_on_bellman() -> None:
  optimal = moeinsum.einsum_path(_BELLMAN_EQ, *_BELLMAN_SHAPES, optimize="optimal")
  branch_2 = moeinsum.einsum_path(_BELLMAN_EQ, *_BELLMAN_SHAPES, optimize="branch-2")
  assert branch_2 == optimal


def test_branch_1_equals_greedy() -> None:
  """branch-1 is structurally greedy. Verify across a 4-operand chain."""
  eq = "ab,bc,cd,de->ae"
  shapes = ((3, 4), (4, 5), (5, 6), (6, 7))
  greedy = moeinsum.einsum_path(eq, *shapes, optimize="greedy")
  branch_1 = moeinsum.einsum_path(eq, *shapes, optimize="branch-1")
  assert branch_1 == greedy


def test_branch_all_no_worse_than_greedy_on_zoo() -> None:
  """Across 4-6 operand zoo, branch-all's total FLOPs must <= greedy.

  The greedy seed gives this for free - branch's pruning bound starts
  at greedy's total cost, so any complete path branch returns is
  weakly better. This is a regression check for that property."""
  cases = [
    ("ab,bc,cd,de->ae", ((3, 4), (4, 5), (5, 6), (6, 7))),
    ("ij,jk,kl,lm,mn->in", ((2, 3), (3, 4), (4, 5), (5, 6), (6, 7))),
    ("ab,bc,ca->", ((4, 5), (5, 6), (6, 4))),
    ("ab,cd,bc,de,ea->", ((3, 4), (5, 6), (4, 5), (6, 7), (7, 3))),
  ]
  for eq, shapes in cases:
    greedy_p = moeinsum.einsum_path(eq, *shapes, optimize="greedy")
    branch_p = moeinsum.einsum_path(eq, *shapes, optimize="branch-all")
    greedy_cost = moeinsum.path_cost(eq, shapes, greedy_p)
    branch_cost = moeinsum.path_cost(eq, shapes, branch_p)
    assert branch_cost["total_flops"] <= greedy_cost["total_flops"], (
      f"{eq}: greedy={greedy_cost['total_flops']} branch-all={branch_cost['total_flops']}"
    )


def test_branch_in_auto_threshold_n5() -> None:
  """For n=5 operands, opt_einsum's auto picks `branch-all`."""
  eq = "ij,jk,kl,lm,mn->in"
  shapes = ((2, 3), (3, 4), (4, 5), (5, 6), (6, 7))
  auto = moeinsum.einsum_path(eq, *shapes, optimize="auto")
  branch_all = moeinsum.einsum_path(eq, *shapes, optimize="branch-all")
  assert auto == branch_all


def test_branch_in_auto_threshold_n7() -> None:
  """For n=7 operands, opt_einsum's auto picks `branch-2`."""
  eq = "ab,bc,cd,de,ef,fg,gh->ah"
  shapes = ((2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9))
  auto = moeinsum.einsum_path(eq, *shapes, optimize="auto")
  branch_2 = moeinsum.einsum_path(eq, *shapes, optimize="branch-2")
  assert auto == branch_2


def test_branch_on_two_operands_is_trivial() -> None:
  """With n=2 there's only one pair - branch trivially returns it."""
  for algo in ("branch-all", "branch-2", "branch-1"):
    path = moeinsum.einsum_path("ij,jk->ik", (3, 4), (4, 5), optimize=algo)
    assert path == [(0, 1)], f"{algo} on 2 operands: {path}"


def test_branch_compute_correctness() -> None:
  """`einsum(..., optimize='branch-all')` must produce numerically
  identical output to numpy on a 4-operand contraction.

  Path order doesn't affect numerical result for matmul-shaped
  contractions; this is a smoke check that the branch dispatch
  doesn't leak a bad path through to the reference backend.
  """
  import numpy as np

  rng = np.random.default_rng(0)
  arrays = [
    rng.standard_normal((2, 3)),
    rng.standard_normal((3, 4)),
    rng.standard_normal((4, 5)),
    rng.standard_normal((5, 6)),
  ]
  expected = np.einsum("ij,jk,kl,lm->im", *arrays, optimize=True)
  for algo in ("branch-all", "branch-2", "branch-1"):
    actual = moeinsum.einsum("ij,jk,kl,lm->im", *arrays, optimize=algo)
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
