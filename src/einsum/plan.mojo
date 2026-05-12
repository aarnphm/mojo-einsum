"""Contraction plan IR.

The plan is a backend-agnostic record of what to compute. It is built from an
`EinsumEquation` and consumed by any backend: reference, max, or native.

A plan is an ordered list of `PlanStep`s. Each step is one of:
  - `UnaryStep`: a single-operand op (reduce / diagonal / trace / transpose).
  - `PairwiseStep`: a two-operand contraction whose dim classification
    (B / K / M / N) is precomputed here so backends just lower.

The path optimizer (`path.mojo`) produces the order of pairwise steps.

Dim role taxonomy (B/K/M/N), mirroring JAX's `_einsum` algorithm at
`jax/_src/numpy/lax_numpy.py:3264-3293`:

  B  batch     - present in lhs, rhs, and output of this step
  K  contract  - present in lhs and rhs, summed out
  M  free-left - present in lhs and step output, not in rhs
  N  free-rt   - present in rhs and step output, not in lhs
"""

from std.utils import Variant

from einsum.parse import EinsumEquation, ELLIPSIS_LABEL


# Unary-op kind tags.
comptime UNARY_REDUCE_SUM: Int = 0
comptime UNARY_DIAGONAL: Int = 1
comptime UNARY_TRACE: Int = 2
comptime UNARY_TRANSPOSE: Int = 3


@fieldwise_init
struct UnaryStep(Copyable, Movable):
    """A single-operand step.

    `operand_idx` is the working-set position to consume. `kind`
    dispatches into the four families above; reducers / diag readers
    use `reduce_axes` / `diag_axes` respectively, and the final
    `out_permutation` reorders the post-reduce / post-diag tensor's axes
    to match `out_labels`.
    """
    var operand_idx: Int
    var kind: Int
    var in_labels: List[Int]
    var out_labels: List[Int]
    var reduce_axes: List[Int]
    var diag_axes: List[List[Int]]
    var out_permutation: List[Int]


@fieldwise_init
struct PairwiseStep(Copyable, Movable):
    """A two-operand step.

    All position lists are indices into the operand's current label
    sequence. The natural BMM-lowering target is `(*B, *M, *K) x
    (*B, *K, *N) -> (*B, *M, *N)`; `out_permutation` then reorders the
    BMM-output axes to `out_labels`.
    """
    var lhs_idx: Int
    var rhs_idx: Int
    var lhs_labels: List[Int]
    var rhs_labels: List[Int]
    var out_labels: List[Int]
    var batch_axes_lhs: List[Int]
    var batch_axes_rhs: List[Int]
    var contract_axes_lhs: List[Int]
    var contract_axes_rhs: List[Int]
    var free_axes_lhs: List[Int]
    var free_axes_rhs: List[Int]
    var out_permutation: List[Int]

# Tagged union via `Variant`, the stdlib idiom.
comptime PlanStep = Variant[UnaryStep, PairwiseStep]


@fieldwise_init
struct ContractionPlan(Copyable, Movable):
    """Ordered sequence of plan steps.

    Working-set semantics: step `k`'s operand indices refer to the
    working set *after* step `k-1`. A pairwise step removes its two
    operands and appends the result. A unary step replaces its operand
    in-place. The final operand is the overall result.
    """
    var steps: List[PlanStep]
    var n_input_operands: Int
    var final_labels: List[Int]


# ---------------------------------------------------------------------
# Dim classification (B / K / M / N)
# ---------------------------------------------------------------------


def _label_in(labels: List[Int], lbl: Int) -> Bool:
    for i in range(len(labels)):
        if labels[i] == lbl:
            return True
    return False


def _index_of(labels: List[Int], lbl: Int) -> Int:
    for i in range(len(labels)):
        if labels[i] == lbl:
            return i
    return -1


def _append_unique(mut out: List[Int], labels: List[Int]) -> None:
    for i in range(len(labels)):
        var lbl = labels[i]
        if not _label_in(out, lbl):
            out.append(lbl)


def _labels_equal(lhs: List[Int], rhs: List[Int]) -> Bool:
    if len(lhs) != len(rhs):
        return False
    for i in range(len(lhs)):
        if lhs[i] != rhs[i]:
            return False
    return True


def _dedupe_labels(labels: List[Int]) -> List[Int]:
    var out = List[Int]()
    for i in range(len(labels)):
        var lbl = labels[i]
        if not _label_in(out, lbl):
            out.append(lbl)
    return out^


def _has_repeated_labels(labels: List[Int]) -> Bool:
    var seen = List[Int]()
    for i in range(len(labels)):
        var lbl = labels[i]
        if _label_in(seen, lbl):
            return True
        seen.append(lbl)
    return False


def _filter_labels(labels: List[Int], keep: List[Int]) -> List[Int]:
    var out = List[Int]()
    for i in range(len(labels)):
        var lbl = labels[i]
        if _label_in(keep, lbl):
            out.append(lbl)
    return out^


def _future_needed_labels(
    working: List[List[Int]],
    final_labels: List[Int],
    skip_a: Int,
    skip_b: Int,
) -> List[Int]:
    var future = List[Int]()
    _append_unique(future, final_labels)
    for i in range(len(working)):
        if i == skip_a or i == skip_b:
            continue
        _append_unique(future, working[i])
    return future^


def _step_output_labels(
    lhs: List[Int],
    rhs: List[Int],
    future: List[Int],
) -> List[Int]:
    var out = List[Int]()
    for i in range(len(lhs)):
        var lbl = lhs[i]
        if _label_in(future, lbl) and not _label_in(out, lbl):
            out.append(lbl)
    for i in range(len(rhs)):
        var lbl = rhs[i]
        if _label_in(future, lbl) and not _label_in(out, lbl):
            out.append(lbl)
    return out^


def _append_diagonal_cleanup_step(
    mut steps: List[PlanStep],
    mut working: List[List[Int]],
    operand_idx: Int,
) raises -> None:
    var labels = working[operand_idx].copy()
    if not _has_repeated_labels(labels):
        return
    var out_labels = _dedupe_labels(labels)
    var unary = build_unary_step(
        operand_idx,
        labels,
        out_labels.copy(),
    )
    steps.append(PlanStep(unary^))
    working[operand_idx] = out_labels^


def classify_pair(
    lhs_labels: List[Int],
    rhs_labels: List[Int],
    out_labels: List[Int],
) raises -> PairwiseStep:
    """Build the B/K/M/N classification for one pairwise step.

    Mirrors JAX `_einsum` at `lax_numpy.py:3264-3293`:
      - batch: in the lhs/rhs/out intersection
      - contract: in the lhs/rhs intersection, not in out
      - free-left (M): in lhs, in out, not in rhs
      - free-right (N): in rhs, in out, not in lhs
    """
    var batch_label_order = List[Int]()
    for i in range(len(out_labels)):
        var lbl = out_labels[i]
        if _label_in(lhs_labels, lbl) and _label_in(rhs_labels, lbl):
            batch_label_order.append(lbl)

    var contract_label_order = List[Int]()
    for i in range(len(lhs_labels)):
        var lbl = lhs_labels[i]
        if _label_in(rhs_labels, lbl) and not _label_in(out_labels, lbl):
            if not _label_in(contract_label_order, lbl):
                contract_label_order.append(lbl)

    var m_label_order = List[Int]()
    for i in range(len(out_labels)):
        var lbl = out_labels[i]
        if _label_in(lhs_labels, lbl) and not _label_in(rhs_labels, lbl):
            m_label_order.append(lbl)

    var n_label_order = List[Int]()
    for i in range(len(out_labels)):
        var lbl = out_labels[i]
        if _label_in(rhs_labels, lbl) and not _label_in(lhs_labels, lbl):
            n_label_order.append(lbl)

    var batch_axes_lhs = List[Int]()
    var batch_axes_rhs = List[Int]()
    for i in range(len(batch_label_order)):
        var lbl = batch_label_order[i]
        batch_axes_lhs.append(_index_of(lhs_labels, lbl))
        batch_axes_rhs.append(_index_of(rhs_labels, lbl))

    var contract_axes_lhs = List[Int]()
    var contract_axes_rhs = List[Int]()
    for i in range(len(contract_label_order)):
        var lbl = contract_label_order[i]
        contract_axes_lhs.append(_index_of(lhs_labels, lbl))
        contract_axes_rhs.append(_index_of(rhs_labels, lbl))

    var free_axes_lhs = List[Int]()
    for i in range(len(m_label_order)):
        free_axes_lhs.append(_index_of(lhs_labels, m_label_order[i]))

    var free_axes_rhs = List[Int]()
    for i in range(len(n_label_order)):
        free_axes_rhs.append(_index_of(rhs_labels, n_label_order[i]))

    # `(*B, *M, *N)` natural BMM-output label order.
    var bmn_labels = List[Int]()
    for i in range(len(batch_label_order)):
        bmn_labels.append(batch_label_order[i])
    for i in range(len(m_label_order)):
        bmn_labels.append(m_label_order[i])
    for i in range(len(n_label_order)):
        bmn_labels.append(n_label_order[i])

    if len(bmn_labels) != len(out_labels):
        raise Error(
            String(
                "classify_pair: bmn label count ",
                len(bmn_labels),
                " != out_labels ",
                len(out_labels),
            )
        )

    var perm = List[Int]()
    for i in range(len(out_labels)):
        var p = _index_of(bmn_labels, out_labels[i])
        if p < 0:
            raise Error(
                String(
                    "classify_pair: out label ",
                    out_labels[i],
                    " not produced by BMM step",
                )
            )
        perm.append(p)

    return PairwiseStep(
        -1,
        -1,
        lhs_labels.copy(),
        rhs_labels.copy(),
        out_labels.copy(),
        batch_axes_lhs^,
        batch_axes_rhs^,
        contract_axes_lhs^,
        contract_axes_rhs^,
        free_axes_lhs^,
        free_axes_rhs^,
        perm^,
    )


# ---------------------------------------------------------------------
# Unary step builder
# ---------------------------------------------------------------------


def build_unary_step(
    operand_idx: Int,
    in_labels: List[Int],
    out_labels: List[Int],
) raises -> UnaryStep:
    """Detect DIAGONAL / TRACE / REDUCE_SUM / TRANSPOSE for one operand."""

    var diag_axes = List[List[Int]]()
    var seen = List[Int]()
    for i in range(len(in_labels)):
        var lbl = in_labels[i]
        if _label_in(seen, lbl):
            continue
        seen.append(lbl)
        var positions = List[Int]()
        for j in range(len(in_labels)):
            if in_labels[j] == lbl:
                positions.append(j)
        if len(positions) >= 2:
            diag_axes.append(positions^)

    var post_diag_labels = List[Int]()
    var post_diag_used = List[Int]()
    for i in range(len(in_labels)):
        var lbl = in_labels[i]
        if _label_in(post_diag_used, lbl):
            continue
        post_diag_used.append(lbl)
        post_diag_labels.append(lbl)

    var reduce_axes = List[Int]()
    for i in range(len(post_diag_labels)):
        if not _label_in(out_labels, post_diag_labels[i]):
            reduce_axes.append(i)

    var post_reduce_labels = List[Int]()
    for i in range(len(post_diag_labels)):
        var lbl = post_diag_labels[i]
        if _label_in(out_labels, lbl):
            post_reduce_labels.append(lbl)

    var perm = List[Int]()
    for i in range(len(out_labels)):
        var p = _index_of(post_reduce_labels, out_labels[i])
        if p < 0:
            raise Error(
                String(
                    "build_unary_step: out label ",
                    out_labels[i],
                    " not produced by reduce/diag pipeline",
                )
            )
        perm.append(p)

    var kind: Int
    if len(diag_axes) > 0 and len(reduce_axes) > 0:
        kind = UNARY_TRACE
    elif len(diag_axes) > 0:
        kind = UNARY_DIAGONAL
    elif len(reduce_axes) > 0:
        kind = UNARY_REDUCE_SUM
    else:
        kind = UNARY_TRANSPOSE

    return UnaryStep(
        operand_idx,
        kind,
        in_labels.copy(),
        out_labels.copy(),
        reduce_axes^,
        diag_axes^,
        perm^,
    )


# ---------------------------------------------------------------------
# Naive left-to-right path
# ---------------------------------------------------------------------


def build_naive_plan(eq: EinsumEquation) raises -> ContractionPlan:
    """Build a plan that contracts operands left-to-right.

    For each operand `i >= 1`, the step output carries labels in
    the accumulator/operand union still needed downstream - either in
    `eq.output` or in any later operand.
    """
    var steps = List[PlanStep]()
    var n = eq.n_operands()
    if n == 0:
        raise Error(String("build_naive_plan: zero operands"))

    if n == 1:
        var u = build_unary_step(0, eq.inputs[0].copy(), eq.output.copy())
        steps.append(PlanStep(u^))
        return ContractionPlan(steps^, 1, eq.output.copy())

    var acc_labels = eq.inputs[0].copy()

    for i in range(1, n):
        var future_needed = List[Int]()
        for k in range(len(eq.output)):
            future_needed.append(eq.output[k])
        for j in range(i + 1, n):
            ref op_j = eq.inputs[j]
            for k in range(len(op_j)):
                var lbl = op_j[k]
                if not _label_in(future_needed, lbl):
                    future_needed.append(lbl)

        var rhs_labels = eq.inputs[i].copy()

        var step_out = List[Int]()
        for k in range(len(acc_labels)):
            var lbl = acc_labels[k]
            if _label_in(future_needed, lbl) and not _label_in(step_out, lbl):
                step_out.append(lbl)
        for k in range(len(rhs_labels)):
            var lbl = rhs_labels[k]
            if _label_in(future_needed, lbl) and not _label_in(step_out, lbl):
                step_out.append(lbl)

        var ps = classify_pair(
            acc_labels.copy(), rhs_labels.copy(), step_out.copy()
        )
        ps.lhs_idx = 0
        ps.rhs_idx = 1
        steps.append(PlanStep(ps^))

        acc_labels = step_out^

    return ContractionPlan(steps^, n, eq.output.copy())


def build_plan_from_path(
    eq: EinsumEquation,
    path: List[List[Int]],
) raises -> ContractionPlan:
    """Build a `ContractionPlan` from explicit working-set path steps.

    Each path element has arity 1 for a unary cleanup step or arity 2 for a
    pairwise contraction. Pairwise output labels are the lhs/rhs union that
    survives into either the final equation output or a later working-set
    operand. Unary steps perform the same diagonal/reduce/transpose cleanup for
    their operand.
    """
    var steps = List[PlanStep]()
    var working = List[List[Int]]()
    for i in range(eq.n_operands()):
        working.append(eq.inputs[i].copy())

    for path_idx in range(len(path)):
        ref raw_step = path[path_idx]
        if len(raw_step) == 1:
            var operand_idx = raw_step[0]
            if operand_idx < 0 or operand_idx >= len(working):
                raise Error(
                    String(
                        "build_plan_from_path: unary step ",
                        path_idx,
                        " operand index ",
                        operand_idx,
                        " out of range",
                    )
                )
            var labels = working[operand_idx].copy()
            var future = _future_needed_labels(
                working,
                eq.output,
                operand_idx,
                -1,
            )
            var deduped = _dedupe_labels(labels)
            var out_labels = _filter_labels(deduped, future)
            var unary = build_unary_step(
                operand_idx,
                labels,
                out_labels.copy(),
            )
            steps.append(PlanStep(unary^))
            working[operand_idx] = out_labels^
        elif len(raw_step) == 2:
            var lhs_idx = raw_step[0]
            var rhs_idx = raw_step[1]
            if lhs_idx == rhs_idx:
                raise Error(
                    String(
                        "build_plan_from_path: pairwise step ",
                        path_idx,
                        " uses the same operand twice",
                    )
                )
            if (
                lhs_idx < 0
                or lhs_idx >= len(working)
                or rhs_idx < 0
                or rhs_idx >= len(working)
            ):
                raise Error(
                    String(
                        "build_plan_from_path: pairwise step ",
                        path_idx,
                        " index out of range",
                    )
                )
            _append_diagonal_cleanup_step(steps, working, lhs_idx)
            _append_diagonal_cleanup_step(steps, working, rhs_idx)
            var lhs = working[lhs_idx].copy()
            var rhs = working[rhs_idx].copy()
            var future = _future_needed_labels(
                working,
                eq.output,
                lhs_idx,
                rhs_idx,
            )
            var out_labels = _step_output_labels(lhs, rhs, future)
            var pair = classify_pair(lhs, rhs, out_labels.copy())
            pair.lhs_idx = lhs_idx
            pair.rhs_idx = rhs_idx
            steps.append(PlanStep(pair^))

            var next_working = List[List[Int]]()
            for i in range(len(working)):
                if i != lhs_idx and i != rhs_idx:
                    next_working.append(working[i].copy())
            next_working.append(out_labels^)
            working = next_working^
        else:
            raise Error(
                String(
                    "build_plan_from_path: step ",
                    path_idx,
                    " has arity ",
                    len(raw_step),
                    ", expected 1 or 2",
                )
            )

    if len(working) != 1:
        raise Error(
            String(
                "build_plan_from_path: path leaves ",
                len(working),
                " tensors; expected 1",
            )
        )

    if not _labels_equal(working[0], eq.output):
        var final_unary = build_unary_step(
            0,
            working[0].copy(),
            eq.output.copy(),
        )
        steps.append(PlanStep(final_unary^))

    return ContractionPlan(steps^, eq.n_operands(), eq.output.copy())
