"""path_cost: FLOP + peak-intermediate accounting per contraction path."""

from __future__ import annotations

import moeinsum


def test_matmul_chain_path_cost_matches_path_choice() -> None:
  """For the Bellman chain, optimal/greedy should cost less than naive."""
  eq = "ij,jk,kl->il"
  shapes = [(100, 1), (1, 100_000), (100_000, 1)]

  naive_path = moeinsum.einsum_path(eq, *shapes, optimize="naive")
  optimal_path = moeinsum.einsum_path(eq, *shapes, optimize="optimal")

  naive_cost = moeinsum.path_cost(eq, shapes, naive_path)
  optimal_cost = moeinsum.path_cost(eq, shapes, optimal_path)

  # Optimal should be at least 100x cheaper in FLOPs and 10^5x cheaper
  # in peak intermediate memory. (The docs round to "10^7x" because
  # they include the 1×1 scalar intermediate; this number is from
  # actual computed ratios.)
  ratio_flops = naive_cost["total_flops"] / optimal_cost["total_flops"]
  ratio_peak = naive_cost["peak_intermediate"] / optimal_cost["peak_intermediate"]
  assert ratio_flops > 100, f"flops ratio {ratio_flops}"
  assert ratio_peak >= 100_000, f"peak ratio {ratio_peak}"


def test_path_cost_step_breakdown() -> None:
  """Per-step records line up with the path length."""
  eq = "ij,jk,kl->il"
  shapes = [(3, 4), (4, 5), (5, 6)]
  path = moeinsum.einsum_path(eq, *shapes, optimize="optimal")
  cost = moeinsum.path_cost(eq, shapes, path)
  steps = cost["steps"]
  assert isinstance(steps, list)
  assert len(steps) == len(path)
  for step in steps:
    assert "flops" in step and step["flops"] > 0


def test_single_operand_unary_cost() -> None:
  """A unary path step on `ii->` is a diagonal-then-sum.

  After dedup, the input carries one label `i` of size 4, so the
  FLOP count is 4 (sum over the diagonal). Peak intermediate is 1
  (the scalar output).
  """
  eq = "ii->"
  shapes = [(4, 4)]
  path = moeinsum.einsum_path(eq, *shapes)
  cost = moeinsum.path_cost(eq, shapes, path)
  assert cost["total_flops"] == 4
  assert cost["peak_intermediate"] == 1
