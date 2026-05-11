"""Integration smoke for `compute_path` on long chains.

Run with:
    mojo run -I src tests/mojo/smoke_compute_path.mojo

`tests/mojo/smoke_path_cost.mojo` already pins the cost helpers
(`_flop_cost`, `_reduced_size_cost`). This file plugs the gap between
those unit tests and `smoke_parse.mojo`'s n ≤ 4 path checks — it runs
the planner glue end-to-end on n ∈ {12, 16, 20} matrix chains across
the algorithm family and asserts the returned path is well-formed.

What "well-formed" means here:
  - exactly `n - 1` pairwise steps (a balanced binary tree over n leaves)
  - every `lhs_idx` and `rhs_idx` is in `[0, working_set_size)` at the
    time of its step (working set shrinks by one per step, since each
    step consumes two operands and appends one intermediate)
  - `lhs_idx != rhs_idx` (no self-pairing)

It deliberately does *not* assert path optimality — that's covered by
the Python side's `test_random_greedy_band.py` (random-greedy-128 within
5% of opt_einsum DP) and `test_opt_einsum_parity.py` (greedy / optimal
parity against opt_einsum). The Mojo smoke is here to catch the kind
of regression where the planner returns a path with the wrong shape or
out-of-bound indices — bugs that wouldn't survive the Python parity
suite, but might survive cost-helper unit tests.

Shapes use alternating-dim chains (`16, 2, 16, 2, ...`) so the optimal
contraction order materially differs from naive left-to-right; this is
the regime where planner bugs in operand selection tend to manifest.
"""

from einsum.parse import parse
from einsum.path import compute_path, ContractionStep
from einsum.backends.reference import _resolve_label_sizes


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _alternating_shapes(n: Int) raises -> List[List[Int]]:
    """For an n-matrix chain, build shapes `[(16,2), (2,16), ...]`.

    The naive left-to-right path on this dim sequence pays roughly
    n × 16² intermediates; the Bellman-optimal path pays n × 16 × 2.
    A wide gap — useful for catching planners that secretly fall back
    to naive.
    """
    var shapes = List[List[Int]]()
    for i in range(n):
        var s = List[Int]()
        if i % 2 == 0:
            s.append(16)
            s.append(2)
        else:
            s.append(2)
            s.append(16)
        shapes.append(s^)
    return shapes^


def _validate_path(
    path: List[ContractionStep],
    n: Int,
    label: String,
) raises:
    """Step count + index-range + no-self-pair invariants."""
    if len(path) != n - 1:
        raise Error(
            String(
                label,
                ": expected ",
                n - 1,
                " steps for n=",
                n,
                ", got ",
                len(path),
            )
        )
    for i in range(len(path)):
        # Working set has `n - i` operands when step i starts; step i
        # removes two and appends one, so the next step sees `n - i - 1`.
        var ws_size = n - i
        var lhs = path[i].lhs_idx
        var rhs = path[i].rhs_idx
        if lhs < 0 or lhs >= ws_size:
            raise Error(
                String(
                    label,
                    " step ",
                    i,
                    ": lhs_idx ",
                    lhs,
                    " out of [0, ",
                    ws_size,
                    ")",
                )
            )
        if rhs < 0 or rhs >= ws_size:
            raise Error(
                String(
                    label,
                    " step ",
                    i,
                    ": rhs_idx ",
                    rhs,
                    " out of [0, ",
                    ws_size,
                    ")",
                )
            )
        if lhs == rhs:
            raise Error(
                String(
                    label,
                    " step ",
                    i,
                    ": self-pair (lhs_idx == rhs_idx == ",
                    lhs,
                    ")",
                )
            )


# ─────────────────────────────────────────────────────────────────────
# n = 12 — greedy / auto / random-greedy / branch-1
# ─────────────────────────────────────────────────────────────────────


def check_chain_n12_algorithm_family() raises:
    """12-operand chain, four algorithm dispatches; each must produce a
    well-formed 11-step path."""
    var eq = parse(String("ab,bc,cd,de,ef,fg,gh,hi,ij,jk,kl,lm->am"))
    var shapes = _alternating_shapes(12)
    var sizes = _resolve_label_sizes(eq, shapes)

    var greedy = compute_path(eq, sizes, String("greedy"))
    _validate_path(greedy, 12, String("n12-greedy"))

    var auto = compute_path(eq, sizes, String("auto"))
    _validate_path(auto, 12, String("n12-auto"))

    var rg = compute_path(eq, sizes, String("random-greedy"))
    _validate_path(rg, 12, String("n12-random-greedy"))

    var rg128 = compute_path(eq, sizes, String("random-greedy-128"))
    _validate_path(rg128, 12, String("n12-random-greedy-128"))

    var branch1 = compute_path(eq, sizes, String("branch-1"))
    _validate_path(branch1, 12, String("n12-branch-1"))

    var naive = compute_path(eq, sizes, String("naive"))
    _validate_path(naive, 12, String("n12-naive"))

    print("check_chain_n12_algorithm_family: OK")


# ─────────────────────────────────────────────────────────────────────
# n = 16 — push optimal (still tractable per opt_einsum's threshold)
# ─────────────────────────────────────────────────────────────────────


def check_chain_n16_with_optimal() raises:
    """16-operand chain — at opt_einsum's optimal-DP cutoff. Verifies
    the planner doesn't blow up at the boundary."""
    var eq = parse(
        String(
            "ab,bc,cd,de,ef,fg,gh,hi,ij,jk,kl,lm,mn,no,op,pq->aq",
        )
    )
    var shapes = _alternating_shapes(16)
    var sizes = _resolve_label_sizes(eq, shapes)

    var greedy = compute_path(eq, sizes, String("greedy"))
    _validate_path(greedy, 16, String("n16-greedy"))

    var optimal = compute_path(eq, sizes, String("optimal"))
    _validate_path(optimal, 16, String("n16-optimal"))

    var auto = compute_path(eq, sizes, String("auto"))
    _validate_path(auto, 16, String("n16-auto"))

    var branch2 = compute_path(eq, sizes, String("branch-2"))
    _validate_path(branch2, 16, String("n16-branch-2"))

    print("check_chain_n16_with_optimal: OK")


# ─────────────────────────────────────────────────────────────────────
# n = 20 — past optimal-DP's tractable range; greedy/random-greedy only
# ─────────────────────────────────────────────────────────────────────


def check_chain_n20_greedy_and_random_greedy() raises:
    """20-operand chain — verifies the cheaper algorithms still produce
    well-formed paths at scale. Optimal-DP is intentionally skipped
    (the n=20 subset enumeration is 2²⁰ ≈ 10⁶ states, slow enough that
    a smoke test is the wrong venue)."""
    var eq = parse(
        String(
            "ab,bc,cd,de,ef,fg,gh,hi,ij,jk,kl,lm,mn,no,op,pq,qr,rs,st,tu->au",
        )
    )
    var shapes = _alternating_shapes(20)
    var sizes = _resolve_label_sizes(eq, shapes)

    var greedy = compute_path(eq, sizes, String("greedy"))
    _validate_path(greedy, 20, String("n20-greedy"))

    var auto = compute_path(eq, sizes, String("auto"))
    _validate_path(auto, 20, String("n20-auto"))

    var rg128 = compute_path(eq, sizes, String("random-greedy-128"))
    _validate_path(rg128, 20, String("n20-random-greedy-128"))

    print("check_chain_n20_greedy_and_random_greedy: OK")


# ─────────────────────────────────────────────────────────────────────
# Cross-algorithm consistency: branch-1 must equal greedy
# ─────────────────────────────────────────────────────────────────────


def check_branch_1_equals_greedy_on_long_chain() raises:
    """`branch-1` is greedy with a width-1 DFS — it must produce the
    *same* path as plain greedy on any input. Catches regressions where
    branch's tie-breaking diverges from greedy's."""
    var eq = parse(String("ab,bc,cd,de,ef,fg,gh,hi,ij,jk,kl,lm->am"))
    var shapes = _alternating_shapes(12)
    var sizes = _resolve_label_sizes(eq, shapes)

    var greedy = compute_path(eq, sizes, String("greedy"))
    var branch1 = compute_path(eq, sizes, String("branch-1"))

    if len(greedy) != len(branch1):
        raise Error(
            String(
                "branch-1: step count ",
                len(branch1),
                " ≠ greedy ",
                len(greedy),
            )
        )
    for i in range(len(greedy)):
        if (
            greedy[i].lhs_idx != branch1[i].lhs_idx
            or greedy[i].rhs_idx != branch1[i].rhs_idx
        ):
            raise Error(
                String(
                    "branch-1: step ",
                    i,
                    " differs from greedy: (",
                    branch1[i].lhs_idx,
                    ", ",
                    branch1[i].rhs_idx,
                    ") vs (",
                    greedy[i].lhs_idx,
                    ", ",
                    greedy[i].rhs_idx,
                    ")",
                )
            )
    print("check_branch_1_equals_greedy_on_long_chain: OK")


def main() raises:
    check_chain_n12_algorithm_family()
    check_chain_n16_with_optimal()
    check_chain_n20_greedy_and_random_greedy()
    check_branch_1_equals_greedy_on_long_chain()
    print("all compute_path integration smoke tests passed")
