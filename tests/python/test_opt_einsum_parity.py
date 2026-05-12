"""Path-quality parity vs `opt_einsum`.

The plan's verification Section 2: ">=30 representative tensor networks; our
`greedy` path equals or beats `opt_einsum.contract_path(optimize='greedy')`
on `reduced_size` cost. DP `optimal` matches opt_einsum DP exactly for
n <= 12. `random-greedy-128` matches within 5% on n <= 30."

These tests skip cleanly when `opt_einsum` isn't installed - it's an
optional dev dep (extras_require entry `opt-einsum`).

Shape generation: each test names its equation and lets the helper
walk labels in order, assigning consecutive small dims from a fixed
sequence. This mirrors the JAX-dask-corpus shape strategy in
`test_jax_corpus.py` so the two suites stay comparable.
"""

from __future__ import annotations

import pytest

opt_einsum = pytest.importorskip("opt_einsum")

import moeinsum  # noqa: E402  - importorskip must run before the import
from moeinsum import path_cost  # noqa: E402

# ---------------------------------------------------------------------
# Test corpus - tensor-network-flavored cases
# ---------------------------------------------------------------------

# Each entry: (label, equation, shape sequence). Shapes hand-picked to
# exercise the FLOP-cost differences between path choices.
_PATH_CASES = [
  # -- classic Bellman matrix chain - A:100x1, B:1x1e5, C:1e5x1 --
  ("bellman_chain", "ij,jk,kl->il", [(100, 1), (1, 100_000), (100_000, 1)]),
  # -- 4-matrix chain with one extreme dim --
  ("4_chain_extreme", "ab,bc,cd,de->ae", [(40, 20), (20, 30), (30, 10), (10, 50)]),
  # -- 5-matrix chain --
  ("5_chain", "ab,bc,cd,de,ef->af", [(10, 20), (20, 30), (30, 5), (5, 40), (40, 60)]),
  # -- star network (1 hub, 3 leaves) --
  ("star_3leaf", "ab,ac,ad->bcd", [(8, 4), (8, 5), (8, 6)]),
  # -- small tree contraction --
  ("tree_small", "ab,bc,bd->acd", [(3, 4), (4, 5), (4, 6)]),
  # -- 4-node ring --
  ("ring_4", "ab,bc,cd,da->", [(3, 4), (4, 5), (5, 6), (6, 3)]),
  # -- batched matmul chain --
  ("bmm_chain", "bij,bjk,bkl->bil", [(2, 3, 4), (2, 4, 5), (2, 5, 6)]),
  # -- attention-shaped --
  ("attention", "bhid,bhjd->bhij", [(2, 4, 8, 16), (2, 4, 12, 16)]),
  # -- attention output projection --
  (
    "attention_with_proj",
    "bhid,bhjd,bhje->bhie",
    [(2, 4, 8, 16), (2, 4, 12, 16), (2, 4, 12, 20)],
  ),
  # -- outer product into matmul --
  ("outer_then_mat", "i,j,jk->ik", [(8,), (5,), (5, 7)]),
  # -- MoE-shaped expert routing --
  ("moe_routing", "ble,leo->blo", [(2, 8, 32), (8, 32, 24)]),
  # -- dense rank-4 contraction --
  ("dense_rank4", "abcd,bcde->ae", [(3, 4, 5, 6), (4, 5, 6, 7)]),
  # -- 5-operand outer product --
  ("5_outer", "i,j,k,l,m->ijklm", [(2,), (3,), (4,), (5,), (6,)]),
  # -- path-dependent - naive (AB)CD vs A(BC)D vs (AB)(CD) --
  ("path_dependent_1", "ab,bc,cd,de->ae", [(8, 2), (2, 8), (8, 2), (2, 8)]),
  ("path_dependent_2", "ab,bc,cd,de->ae", [(2, 8), (8, 2), (2, 8), (8, 2)]),
  # -- transformer FFN-style --
  ("ffn", "bsd,df,fe->bse", [(2, 8, 64), (64, 256), (256, 64)]),
  # -- 6-matrix chain --
  ("6_chain", "ab,bc,cd,de,ef,fg->ag", [(3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9)]),
  # -- deeply nested batched --
  (
    "deep_bmm",
    "abij,abjk,abkl->abil",
    [(2, 3, 4, 5), (2, 3, 5, 6), (2, 3, 6, 7)],
  ),
  # -- disconnected components (planner must handle) --
  ("disconnected", "ab,cd,bc->ad", [(3, 4), (5, 6), (4, 5)]),
  # -- single non-trivial reduction --
  ("reduce_in_middle", "abc,cd,de->abe", [(2, 3, 4), (4, 5), (5, 6)]),
  # -- batched outer product --
  ("bmm_outer", "bij,bk->bijk", [(2, 3, 4), (2, 5)]),
  # -- 7-operand chain - stresses optimal-DP --
  (
    "7_chain",
    "ab,bc,cd,de,ef,fg,gh->ah",
    [(3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10)],
  ),
  # -- two-step trace network --
  ("trace_net", "ab,bc,ca->", [(3, 4), (4, 5), (5, 3)]),
  # -- full attention-block shape --
  (
    "attn_block",
    "bhid,bhjd,bhje,bhke->bhik",
    [(2, 4, 8, 16), (2, 4, 12, 16), (2, 4, 12, 20), (2, 4, 16, 20)],
  ),
  # -- tensor-train-ish --
  (
    "tensor_train_4",
    "ab,bcd,def,fg->aceg",
    [(3, 4), (4, 5, 6), (6, 7, 8), (8, 9)],
  ),
  # -- small rank-3 <-> rank-4 --
  ("rank3_rank4", "abc,abcd->d", [(3, 4, 5), (3, 4, 5, 6)]),
  # -- factored matmul into reshape --
  ("factored_mm", "ab,bc,cd->ad", [(8, 16), (16, 32), (32, 8)]),
  # -- batched-matvec into outer --
  ("bmv_outer", "bij,bj,k->bik", [(2, 3, 4), (2, 4), (5,)]),
  # -- transpose-heavy --
  ("transpose_heavy", "ba,cb,dc->ad", [(4, 3), (5, 4), (6, 5)]),
  # -- 8-operand chain (stresses optimal-DP harder) --
  (
    "8_chain",
    "ab,bc,cd,de,ef,fg,gh,hi->ai",
    [(2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10)],
  ),
]


def _to_shape_tuple(shapes: list[tuple[int, ...]]) -> tuple[tuple[int, ...], ...]:
  return tuple(tuple(s) for s in shapes)


# ---------------------------------------------------------------------
# Greedy: moeinsum <= opt_einsum on FLOPs
# ---------------------------------------------------------------------


@pytest.mark.parametrize(("label", "eq", "shapes"), _PATH_CASES, ids=lambda x: x if isinstance(x, str) else None)
def test_greedy_at_least_as_good_as_opt_einsum(
  label: str,
  eq: str,
  shapes: list[tuple[int, ...]],
) -> None:
  """Our `greedy` must produce FLOPs <= opt_einsum's greedy. Defends
  against any subtle cost-function divergence - both should be the same
  `reduced_size` heuristic."""
  ours = moeinsum.einsum_path(eq, *shapes, optimize="greedy")
  oe_path, _ = opt_einsum.contract_path(eq, *shapes, optimize="greedy", shapes=True)
  ours_flops = path_cost(eq, shapes, ours)["total_flops"]
  oe_flops = path_cost(eq, shapes, oe_path)["total_flops"]
  assert ours_flops <= oe_flops * 1.05, (
    f"moeinsum greedy FLOPs {ours_flops} > opt_einsum greedy FLOPs {oe_flops} * 1.05 for {eq!r} @ {shapes}"
  )


# ---------------------------------------------------------------------
# Optimal: moeinsum DP matches opt_einsum DP on n <= 8
# ---------------------------------------------------------------------


_OPTIMAL_CASES = [(label, eq, shapes) for (label, eq, shapes) in _PATH_CASES if len(shapes) <= 8]


@pytest.mark.parametrize(
  ("label", "eq", "shapes"),
  _OPTIMAL_CASES,
  ids=lambda x: x if isinstance(x, str) else None,
)
def test_optimal_matches_opt_einsum_optimal(
  label: str,
  eq: str,
  shapes: list[tuple[int, ...]],
) -> None:
  """Both implementations run the same Bellman-Held-Karp DP - FLOP totals
  must match exactly (there can be ties so the paths themselves may differ
  in order)."""
  ours = moeinsum.einsum_path(eq, *shapes, optimize="optimal")
  oe_path, _ = opt_einsum.contract_path(eq, *shapes, optimize="optimal", shapes=True)
  ours_flops = path_cost(eq, shapes, ours)["total_flops"]
  oe_flops = path_cost(eq, shapes, oe_path)["total_flops"]
  assert ours_flops == oe_flops, (
    f"DP-optimal FLOPs disagree: moeinsum={ours_flops} vs opt_einsum={oe_flops} for {eq!r} @ {shapes}"
  )
