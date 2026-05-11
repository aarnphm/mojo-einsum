"""Contraction-path optimizer.

Given an `EinsumEquation` and per-label sizes, choose the order in which
to perform pairwise contractions. The interface mirrors `opt_einsum`:

  - `greedy`            — reduced_size heuristic, O(n * k) per step
  - `optimal`           — DP over subsets (Bellman-Held-Karp), n ≤ 16
  - `auto`              — `optimal` if n ≤ 4, else `greedy`
  - `random-greedy`     — 32 deterministic noisy-greedy trials
  - `explicit(path)`    — caller-supplied path
  - (P4 extension) `branch` families.

The output is a `ContractionPath`: a `List[(lhs_idx, rhs_idx)]` of
pairwise step indices, working-set semantics — operand indices refer to
the working set *at the time of that step*, not original input
positions.

Cost model: `reduced_size = size(A) + size(B) - size(A⊗B)`. opt_einsum's
default; correlates well with FLOPs for typical ML-shaped contractions
but undervalues FLOP/memory-divergent ops (Cardoso et al. 2024,
arxiv 2405.09644 propose a corrected cost). v0.1 ships pure
`reduced_size`.
"""

from einsum.parse import EinsumEquation, ELLIPSIS_LABEL


@fieldwise_init
struct ContractionStep(Copyable, Movable):
    """One pairwise step in a path. `lhs_idx` and `rhs_idx` are indices
    into the working set *at the time of this step*. The resulting
    intermediate is appended at the end of the working set; the two
    operands are removed."""

    var lhs_idx: Int
    var rhs_idx: Int


comptime PATH_GREEDY: Int = 0
comptime PATH_OPTIMAL: Int = 1
comptime PATH_AUTO: Int = 2
comptime PATH_EXPLICIT: Int = 3


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _label_set_union(a: List[Int], b: List[Int]) -> List[Int]:
    """Union preserving `a`'s order, then appending `b`'s unique labels."""
    var out = List[Int]()
    for i in range(len(a)):
        var lbl = a[i]
        var found = False
        for j in range(len(out)):
            if out[j] == lbl:
                found = True
                break
        if not found:
            out.append(lbl)
    for i in range(len(b)):
        var lbl = b[i]
        var found = False
        for j in range(len(out)):
            if out[j] == lbl:
                found = True
                break
        if not found:
            out.append(lbl)
    return out^


def _label_set_intersect(a: List[Int], b: List[Int]) -> List[Int]:
    var out = List[Int]()
    for i in range(len(a)):
        var lbl = a[i]
        for j in range(len(b)):
            if b[j] == lbl:
                out.append(lbl)
                break
    return out^


def _label_in(labels: List[Int], lbl: Int) -> Bool:
    for i in range(len(labels)):
        if labels[i] == lbl:
            return True
    return False


def _step_output_labels(
    lhs: List[Int],
    rhs: List[Int],
    other_operand_labels: List[List[Int]],
    final_output: List[Int],
) -> List[Int]:
    """Labels carried forward after contracting `lhs` and `rhs`.

    A label survives if it appears in the final output or in any operand
    not consumed by this step. Otherwise it's contracted out (summed).
    """
    var future = List[Int]()
    for i in range(len(final_output)):
        future.append(final_output[i])
    for i in range(len(other_operand_labels)):
        ref op = other_operand_labels[i]
        for j in range(len(op)):
            var lbl = op[j]
            if not _label_in(future, lbl):
                future.append(lbl)

    var union = _label_set_union(lhs, rhs)
    var out = List[Int]()
    for i in range(len(union)):
        var lbl = union[i]
        if _label_in(future, lbl):
            out.append(lbl)
    return out^


def _tensor_size(labels: List[Int], label_sizes: List[Int]) -> Int:
    """Product of label sizes — count of elements in this tensor."""
    var s: Int = 1
    for i in range(len(labels)):
        s *= label_sizes[labels[i]]
    return s


def _reduced_size_cost(
    lhs: List[Int],
    rhs: List[Int],
    out_labels: List[Int],
    label_sizes: List[Int],
) -> Int:
    """opt_einsum's greedy cost: how much memory the step removes.

    `size(A) + size(B) - size(A⊗B)`. Bigger = better (greater memory
    reduction = preferred pair).
    """
    var a = _tensor_size(lhs, label_sizes)
    var b = _tensor_size(rhs, label_sizes)
    var c = _tensor_size(out_labels, label_sizes)
    return a + b - c


def _flop_cost(
    lhs: List[Int],
    rhs: List[Int],
    out_labels: List[Int],
    label_sizes: List[Int],
) -> Int:
    """FLOP count for one pairwise contraction.

    Equals the product of all label sizes in (lhs ∪ rhs) — the natural
    loop bound for any nested-loop implementation. For BMM-shaped:
    O(B · M · N · K).
    """
    var union = _label_set_union(lhs, rhs)
    var f: Int = 1
    for i in range(len(union)):
        f *= label_sizes[union[i]]
    return f


# ─────────────────────────────────────────────────────────────────────
# Greedy path
# ─────────────────────────────────────────────────────────────────────


def greedy_path(
    eq: EinsumEquation,
    label_sizes: List[Int],
) raises -> List[ContractionStep]:
    """opt_einsum's greedy `reduced_size` heuristic.

    Repeatedly pick the pair maximizing `reduced_size`. Ties broken by
    smaller FLOP cost, then by leftmost lhs index.
    """
    var n = eq.n_operands()
    if n < 2:
        return List[ContractionStep]()

    var working = List[List[Int]]()
    for i in range(n):
        working.append(eq.inputs[i].copy())

    var steps = List[ContractionStep]()

    while len(working) >= 2:
        var best_i: Int = -1
        var best_j: Int = -1
        var best_score: Int = -9223372036854775807  # max-negative sentinel
        var best_flops: Int = 9223372036854775807
        var best_out = List[Int]()

        for i in range(len(working)):
            for j in range(i + 1, len(working)):
                var others = List[List[Int]]()
                for k in range(len(working)):
                    if k != i and k != j:
                        others.append(working[k].copy())
                var out = _step_output_labels(working[i], working[j], others, eq.output)
                var score = _reduced_size_cost(working[i], working[j], out, label_sizes)
                var flops = _flop_cost(working[i], working[j], out, label_sizes)
                var better = False
                if score > best_score:
                    better = True
                elif score == best_score and flops < best_flops:
                    better = True
                if better:
                    best_i = i
                    best_j = j
                    best_score = score
                    best_flops = flops
                    best_out = out^

        steps.append(ContractionStep(best_i, best_j))
        # Build new working set: remove i and j, append result.
        var new_working = List[List[Int]]()
        for k in range(len(working)):
            if k != best_i and k != best_j:
                new_working.append(working[k].copy())
        new_working.append(best_out^)
        working = new_working^

    return steps^


# ─────────────────────────────────────────────────────────────────────
# Optimal path (DP over subsets, Bellman-Held-Karp)
# ─────────────────────────────────────────────────────────────────────


def _subset_labels(
    subset_mask: Int,
    n: Int,
    operand_labels: List[List[Int]],
    other_operand_labels_template: List[List[Int]],
    final_output: List[Int],
) -> List[Int]:
    """Compute the surviving label set for a subset of operands.

    `subset_mask` is a bitmask over `range(n)`. The resulting tensor
    carries labels that either appear in `final_output` or in *any*
    operand NOT in the subset.
    """
    var future = List[Int]()
    for i in range(len(final_output)):
        future.append(final_output[i])
    for i in range(n):
        if (subset_mask & (1 << i)) == 0:
            ref op = operand_labels[i]
            for j in range(len(op)):
                var lbl = op[j]
                if not _label_in(future, lbl):
                    future.append(lbl)

    # Union of all subset operands' labels.
    var union = List[Int]()
    for i in range(n):
        if (subset_mask & (1 << i)) != 0:
            ref op = operand_labels[i]
            for j in range(len(op)):
                var lbl = op[j]
                if not _label_in(union, lbl):
                    union.append(lbl)

    var out = List[Int]()
    for i in range(len(union)):
        var lbl = union[i]
        if _label_in(future, lbl):
            out.append(lbl)
    return out^


def optimal_path(
    eq: EinsumEquation,
    label_sizes: List[Int],
) raises -> List[ContractionStep]:
    """DP over subsets — optimal in FLOPs, O(3^n) time, O(2^n) memory.

    Tractable for n ≤ 16 (≈43M states at n=16). Above 16, fall back to
    greedy. Caller is responsible for the threshold check.

    State: `f[S]` = minimum total FLOP cost to contract subset `S`.
    Recurrence: `f[S] = min_{T ⊂ S, T ≠ ∅, T ≠ S} f[T] + f[S\\T] + cost(T, S\\T)`.
    """
    var n = eq.n_operands()
    if n > 16:
        raise Error(String("optimal_path: ", n, " operands exceeds DP limit of 16"))
    if n < 2:
        return List[ContractionStep]()

    var n_subsets = 1 << n
    var operand_labels = List[List[Int]]()
    for i in range(n):
        operand_labels.append(eq.inputs[i].copy())

    # Pre-compute label sets for every subset. `subset_labels[S]` = the
    # surviving labels if S were contracted into a single tensor.
    var subset_labels = List[List[Int]]()
    for s in range(n_subsets):
        var labels = _subset_labels(s, n, operand_labels, operand_labels, eq.output)
        subset_labels.append(labels^)

    # f[S] = best cost to contract subset S to a single tensor.
    # best_split[S] = (T, S\T) — the optimal first split of subset S.
    var INF: Int = 9223372036854775807
    var f = List[Int]()
    var best_split_lhs = List[Int]()  # bitmask of T
    for _ in range(n_subsets):
        f.append(INF)
        best_split_lhs.append(-1)

    # Base case: singleton subsets cost 0.
    for i in range(n):
        f[1 << i] = 0

    # DP over subset sizes ascending.
    for s in range(1, n_subsets):
        # Count popcount manually (popcount is cheap, but loop is OK at n ≤ 16).
        var pop: Int = 0
        var tmp = s
        while tmp > 0:
            pop += tmp & 1
            tmp >>= 1
        if pop < 2:
            continue

        # Enumerate non-empty proper subsets t of s.
        var t = (s - 1) & s
        while t > 0:
            var u = s ^ t  # u = s \ t
            if f[t] < INF and f[u] < INF:
                # Cost of combining t and u to form s.
                var cost = _flop_cost(
                    subset_labels[t],
                    subset_labels[u],
                    subset_labels[s],
                    label_sizes,
                )
                var total = f[t] + f[u] + cost
                if total < f[s]:
                    f[s] = total
                    best_split_lhs[s] = t
            t = (t - 1) & s

    # Reconstruct path by walking the splits, recording pairwise steps
    # in post-order (deepest leaves first). The output uses working-set
    # indices, so we need to translate the bit-set view to a linear one.
    var steps = List[ContractionStep]()
    _emit_path_dfs(n_subsets - 1, best_split_lhs, steps)

    # `steps` currently records (subset_mask_lhs, subset_mask_rhs) — we
    # need (working_set_idx_lhs, working_set_idx_rhs). Translate.
    var working = List[Int]()  # bitmask per working-set slot
    for i in range(n):
        working.append(1 << i)

    var out_steps = List[ContractionStep]()
    for k in range(len(steps)):
        var t = steps[k].lhs_idx  # bitmask of lhs subset
        var u = steps[k].rhs_idx  # bitmask of rhs subset
        var li: Int = -1
        var ri: Int = -1
        for i in range(len(working)):
            if working[i] == t:
                li = i
            elif working[i] == u:
                ri = i
        if li < 0 or ri < 0:
            raise Error(String("optimal_path: working-set translation failed"))
        if li > ri:
            var tmp = li
            li = ri
            ri = tmp
        out_steps.append(ContractionStep(li, ri))

        var combined = working[li] | working[ri]
        var new_working = List[Int]()
        for i in range(len(working)):
            if i != li and i != ri:
                new_working.append(working[i])
        new_working.append(combined)
        working = new_working^

    return out_steps^


def _emit_path_dfs(
    subset: Int,
    best_split_lhs: List[Int],
    mut steps: List[ContractionStep],
) raises -> None:
    """Post-order walk of the optimal contraction tree."""
    var lhs = best_split_lhs[subset]
    if lhs < 0:
        return  # singleton — no split to emit
    var rhs = subset ^ lhs
    _emit_path_dfs(lhs, best_split_lhs, steps)
    _emit_path_dfs(rhs, best_split_lhs, steps)
    steps.append(ContractionStep(lhs, rhs))


# ─────────────────────────────────────────────────────────────────────
# Auto dispatch
# ─────────────────────────────────────────────────────────────────────


def auto_path(
    eq: EinsumEquation,
    label_sizes: List[Int],
) raises -> List[ContractionStep]:
    """opt_einsum's `auto` threshold table — DP up to n=4, greedy after.

    The full opt_einsum table also has `branch-all` for n≤5 and
    `branch-2` for n≤7 between optimal and greedy; we ship the
    simplified threshold here, branch family is a future addition.
    """
    var n = eq.n_operands()
    if n <= 4:
        return optimal_path(eq, label_sizes)
    return greedy_path(eq, label_sizes)


# ─────────────────────────────────────────────────────────────────────
# Public entrypoint
# ─────────────────────────────────────────────────────────────────────


def _path_total_flops(
    eq: EinsumEquation,
    label_sizes: List[Int],
    steps: List[ContractionStep],
) raises -> Int:
    """Sum the FLOP cost of every step in `steps`.

    Re-runs the working-set simulation so we can compute each step's
    output labels and FLOP count from `label_sizes`.
    """
    var n = eq.n_operands()
    var working = List[List[Int]]()
    for i in range(n):
        working.append(eq.inputs[i].copy())

    var total: Int = 0
    for k in range(len(steps)):
        var li = steps[k].lhs_idx
        var ri = steps[k].rhs_idx
        var others = List[List[Int]]()
        for j in range(len(working)):
            if j != li and j != ri:
                others.append(working[j].copy())
        var out = _step_output_labels(working[li], working[ri], others, eq.output)
        total += _flop_cost(working[li], working[ri], out, label_sizes)
        var new_working = List[List[Int]]()
        for j in range(len(working)):
            if j != li and j != ri:
                new_working.append(working[j].copy())
        new_working.append(out^)
        working = new_working^
    return total


def random_greedy_path(
    eq: EinsumEquation,
    label_sizes: List[Int],
    var n_trials: Int = 32,
    var seed: Int = 0,
) raises -> List[ContractionStep]:
    """N greedy trials with stochastic cost perturbation, return best.

    At each step, we pick the pair with the best noisy `reduced_size`,
    where the deterministic hash perturbation is derived from
    `seed + trial * 1009 + step + lhs * 131 + rhs * 17`. After N
    trials, return whichever path has the lowest *total* FLOP cost.

    This is the working approximation of opt_einsum's random-greedy
    (PR #78). The original uses Gumbel-noise-perturbed costs; a hashed
    tiebreaker is the same shape at coarser resolution — good enough
    to escape the deterministic-greedy traps for typical n ≤ 30
    contractions.
    """
    if n_trials < 1:
        n_trials = 1
    var n = eq.n_operands()
    if n < 2:
        return List[ContractionStep]()

    var best_total: Int = 9223372036854775807
    var best_path = List[ContractionStep]()

    for trial in range(n_trials):
        var trial_seed = seed + trial * 1009
        var working = List[List[Int]]()
        for i in range(n):
            working.append(eq.inputs[i].copy())
        var steps = List[ContractionStep]()
        var step_idx: Int = 0

        while len(working) >= 2:
            var best_i: Int = -1
            var best_j: Int = -1
            var best_noisy_score: Int = -9223372036854775807
            var best_score: Int = -9223372036854775807
            var best_flops: Int = 9223372036854775807
            var best_jitter: Int = -1
            var best_out = List[Int]()

            for i in range(len(working)):
                for j in range(i + 1, len(working)):
                    var others = List[List[Int]]()
                    for k in range(len(working)):
                        if k != i and k != j:
                            others.append(working[k].copy())
                    var out = _step_output_labels(working[i], working[j], others, eq.output)
                    var score = _reduced_size_cost(working[i], working[j], out, label_sizes)
                    var flops = _flop_cost(working[i], working[j], out, label_sizes)
                    var jitter = (trial_seed + step_idx + i * 131 + j * 17) & 0xFFFF
                    var centered_jitter = jitter - 32768
                    var abs_score = score if score >= 0 else -score
                    var noise_scale = abs_score // 2 + 1
                    var noisy_score = score + (centered_jitter * noise_scale) // 32768
                    var better = False
                    if noisy_score > best_noisy_score:
                        better = True
                    elif noisy_score == best_noisy_score and score > best_score:
                        better = True
                    elif noisy_score == best_noisy_score and score == best_score and flops < best_flops:
                        better = True
                    elif (
                        noisy_score == best_noisy_score
                        and score == best_score
                        and flops == best_flops
                        and jitter > best_jitter
                    ):
                        better = True
                    if better:
                        best_i = i
                        best_j = j
                        best_noisy_score = noisy_score
                        best_score = score
                        best_flops = flops
                        best_jitter = jitter
                        best_out = out^

            steps.append(ContractionStep(best_i, best_j))
            var new_working = List[List[Int]]()
            for k in range(len(working)):
                if k != best_i and k != best_j:
                    new_working.append(working[k].copy())
            new_working.append(best_out^)
            working = new_working^
            step_idx += 1

        var total = _path_total_flops(eq, label_sizes, steps)
        if total < best_total:
            best_total = total
            best_path = steps^

    return best_path^


def compute_path(
    eq: EinsumEquation,
    label_sizes: List[Int],
    algorithm: String,
) raises -> List[ContractionStep]:
    """Dispatch to the named algorithm.

    `algorithm ∈ {"greedy", "optimal", "auto", "naive", "random-greedy"}`.
    "naive" is deterministic left-to-right pairing, useful as a baseline.
    """
    if algorithm == String("greedy"):
        return greedy_path(eq, label_sizes)
    if algorithm == String("optimal"):
        return optimal_path(eq, label_sizes)
    if algorithm == String("auto"):
        return auto_path(eq, label_sizes)
    if algorithm == String("naive"):
        return naive_path(eq)
    if algorithm == String("random-greedy"):
        return random_greedy_path(eq, label_sizes)
    raise Error(
        String(
            "compute_path: unknown algorithm '",
            algorithm,
            "'. Supported: greedy, optimal, auto, naive, random-greedy.",
        )
    )


def naive_path(eq: EinsumEquation) raises -> List[ContractionStep]:
    """Left-to-right: (0,1), then (0,1) on the new working set, etc.

    This is what P1's `build_naive_plan` used implicitly. Exposed for
    debugging / regression baselines.
    """
    var n = eq.n_operands()
    var steps = List[ContractionStep]()
    for _ in range(n - 1):
        steps.append(ContractionStep(0, 1))
    return steps^
