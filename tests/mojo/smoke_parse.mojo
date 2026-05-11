"""Mojo-side parser smoke test.

Run with:
    mojo run -I src tests/mojo/smoke_parse.mojo

Exercises parse() + expand_ellipsis() on a curated set of equations
without touching Python or MAX. The first build target to make work.
"""

from einsum.parse import parse, expand_ellipsis, ELLIPSIS_LABEL, EinsumEquation
from einsum.plan import build_naive_plan, ContractionPlan


def check_basic() raises:
    var eq = parse(String("ij,jk->ik"))
    if eq.n_operands() != 2:
        raise Error(String("expected 2 operands, got ", eq.n_operands()))
    if eq.n_labels != 3:
        raise Error(String("expected 3 labels, got ", eq.n_labels))
    if not eq.has_explicit_output:
        raise Error(String("expected explicit output"))
    print("check_basic: OK")


def check_trace() raises:
    var eq = parse(String("ii->"))
    if eq.n_operands() != 1:
        raise Error(String("trace: expected 1 operand"))
    if len(eq.output) != 0:
        raise Error(String("trace: expected empty output"))
    print("check_trace: OK")


def check_implicit_output() raises:
    var eq = parse(String("ij,jk"))
    if eq.has_explicit_output:
        raise Error(String("implicit: expected has_explicit_output=False"))
    if len(eq.output) != 2:
        raise Error(String("implicit: expected 2 output labels, got ", len(eq.output)))
    print("check_implicit_output: OK")


def check_ellipsis() raises:
    var eq = parse(String("...ij,jk->...ik"))
    var has_ellipsis = False
    for j in range(len(eq.inputs[0])):
        if eq.inputs[0][j] == ELLIPSIS_LABEL:
            has_ellipsis = True
    if not has_ellipsis:
        raise Error(String("ellipsis: lhs should contain ELLIPSIS_LABEL"))
    var ranks = List[Int]()
    ranks.append(4)  # ...ij with 4-D input → ellipsis is 2 dims
    ranks.append(2)  # jk with 2-D
    expand_ellipsis(eq, ranks)
    # After expansion: lhs has 4 labels, output has 4 labels.
    if len(eq.inputs[0]) != 4:
        raise Error(
            String(
                "ellipsis expand: expected 4 labels in lhs, got ",
                len(eq.inputs[0]),
            )
        )
    print("check_ellipsis: OK")


def check_naive_plan() raises:
    var eq = parse(String("ij,jk,kl->il"))
    var plan = build_naive_plan(eq)
    if len(plan.steps) != 2:
        raise Error(String("naive plan: expected 2 steps, got ", len(plan.steps)))
    print("check_naive_plan: OK")


def main() raises:
    check_basic()
    check_trace()
    check_implicit_output()
    check_ellipsis()
    check_naive_plan()
    print("all parser smoke tests passed")
