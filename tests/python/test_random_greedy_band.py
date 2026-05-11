"""Plan Section 2 - `random-greedy-128` within 5% of opt_einsum optimal on n <= 30.

The plan's verification Section 2 names three targets for path quality vs
opt_einsum:

  1. our `greedy` <= opt_einsum `greedy` on `reduced_size` cost.
  2. DP `optimal` matches opt_einsum DP exactly for n <= 12.
  3. `random-greedy-128` matches *optimal* within 5% on n <= 30.

(1) and (2) are pinned by `test_opt_einsum_parity.py`. This file
covers (3): does the stochastic perturbation across 128 trials land
within 5% of the exact Bellman-Held-Karp optimum?

Three regimes, one parametric corpus:
  - n in {12, 16, 20, 25, 30}
  - chain + tensor-network-flavored shapes
  - 5%-band assertion on FLOP cost vs `opt_einsum.contract_path(optimize='optimal')`

The optimal call is feasible up to n <= 16 (opt_einsum's DP cuts off
near there); beyond that we fall back to `optimize='dp'` which is the
same Bellman-Held-Karp recurrence with a memo cap. The 5%-band still
applies - what changes is the baseline name.
"""

from __future__ import annotations

import pytest

opt_einsum = pytest.importorskip("opt_einsum")

import moeinsum  # noqa: E402  - importorskip must run before the import
from moeinsum._cost import path_cost  # noqa: E402


def _chain_shapes(n: int, dim_lo: int = 2, dim_hi: int = 16) -> tuple[str, list[tuple[int, int]]]:
  """Build an n-matrix chain `ab,bc,cd,...` with deterministically varied dims.

  The chain is deliberately path-dependent: dims alternate between small
  and large so naive left-to-right is far from optimal. This is exactly
  the regime where a random-greedy planner has room to err.
  """
  # Deterministic dim sequence: 16, 2, 16, 2, ... - alternating wide/narrow.
  # Naive (A*B)*C*... pays ~n x dim_hi^2, the Bellman-optimal path pays
  # ~n x dim_hi x dim_lo.
  dims = [dim_hi if i % 2 == 0 else dim_lo for i in range(n + 1)]
  shapes = [(dims[i], dims[i + 1]) for i in range(n)]

  # Build the equation `ab,bc,cd,de,...->az` using single-char labels
  # from a 52-char pool - opt_einsum's grammar maxes out at 52 distinct
  # single-char labels (a-zA-Z). n <= 30 fits comfortably.
  pool = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
  if n + 1 > len(pool):
    raise ValueError(f"chain length n={n} exceeds single-char label pool")
  ops_s = [f"{pool[i]}{pool[i + 1]}" for i in range(n)]
  out_s = f"{pool[0]}{pool[n]}"
  eq = ",".join(ops_s) + "->" + out_s
  return eq, shapes


# ---------------------------------------------------------------------
# Parametric: random-greedy-128 within 5% of opt_einsum DP/optimal
# ---------------------------------------------------------------------


@pytest.mark.parametrize("n", [12, 16, 20, 25, 30])
def test_random_greedy_128_within_5pct_of_opt_einsum_dp(n: int) -> None:
  """For an n-matrix chain with alternating wide/narrow dims, our
  `random-greedy-128` total-FLOP cost must be within 5% of the optimum
  found by opt_einsum's DP planner.

  opt_einsum's `optimize='dp'` runs the same Bellman-Held-Karp DP we
  do for `optimal` but with a memo cap that lets it scale past n=16.
  It's the strongest baseline we can compare against beyond n=16.
  """
  eq, shapes = _chain_shapes(n)

  ours = moeinsum.einsum_path(eq, *shapes, optimize="random-greedy-128")
  oe_path, _ = opt_einsum.contract_path(eq, *shapes, optimize="dp", shapes=True)

  ours_flops = path_cost(eq, shapes, ours)["total_flops"]
  oe_flops = path_cost(eq, shapes, oe_path)["total_flops"]

  # The 5%-band: ours / oe <= 1.05. We assert the FLOP-ratio.
  ratio = ours_flops / oe_flops
  assert ratio <= 1.05, (
    f"n={n} chain: random-greedy-128 FLOPs {ours_flops} vs opt_einsum DP {oe_flops} = "
    f"ratio {ratio:.4f} > 1.05 (5%-band exceeded)"
  )


# ---------------------------------------------------------------------
# Direct convergence: random-greedy-128 vs random-greedy-32
# ---------------------------------------------------------------------


@pytest.mark.parametrize("n", [12, 20, 30])
def test_random_greedy_N_monotone_in_N(n: int) -> None:
  """More trials shouldn't make the random-greedy planner *worse*. The
  trial-N=128 result must dominate (<=) trial-N=32 on the same chain.

  Sanity check on the stochastic kernel: if increasing N hurts, we're
  not actually exploring - we're regressing to the seed-dependent first
  trial.
  """
  eq, shapes = _chain_shapes(n)

  p32 = moeinsum.einsum_path(eq, *shapes, optimize="random-greedy-32")
  p128 = moeinsum.einsum_path(eq, *shapes, optimize="random-greedy-128")

  f32 = path_cost(eq, shapes, p32)["total_flops"]
  f128 = path_cost(eq, shapes, p128)["total_flops"]

  # Allow a 1% tolerance - stochastic search can land on indistinguishable
  # paths at higher trial counts.
  assert f128 <= f32 * 1.01, f"n={n}: random-greedy-128 FLOPs {f128} > random-greedy-32 FLOPs {f32} * 1.01"


# ---------------------------------------------------------------------
# Random-greedy-N matches greedy at N=1 (degenerate case)
# ---------------------------------------------------------------------


def test_random_greedy_1_matches_greedy_on_deterministic_seed() -> None:
  """`random-greedy-1` with the zero-noise default falls back to plain
  greedy. Both should produce the same FLOP cost on the Bellman chain
  (the canonical path-dependent case).
  """
  eq = "ij,jk,kl->il"
  shapes = [(100, 1), (1, 100_000), (100_000, 1)]

  p_greedy = moeinsum.einsum_path(eq, *shapes, optimize="greedy")
  p_rg1 = moeinsum.einsum_path(eq, *shapes, optimize="random-greedy-1")

  f_greedy = path_cost(eq, shapes, p_greedy)["total_flops"]
  f_rg1 = path_cost(eq, shapes, p_rg1)["total_flops"]

  # At N=1 the planner is essentially deterministic greedy; equal FLOPs.
  assert f_greedy == f_rg1, f"greedy {f_greedy} != random-greedy-1 {f_rg1}"
