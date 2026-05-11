"""Mojo-side parser smoke test.

Run with:
    mojo run -I src tests/mojo/smoke_parse.mojo

Exercises parse() + expand_ellipsis() on a curated set of equations
without touching Python or MAX. The first build target to make work.
"""

from einsum.parse import parse, expand_ellipsis, ELLIPSIS_LABEL, EinsumEquation
from einsum.plan import build_naive_plan, ContractionPlan
from einsum.path import compute_path, ContractionStep
from einsum.backends.reference import _resolve_label_sizes


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


def check_path_greedy() raises:
    # Bellman matrix-chain: A:100x1, B:1x100000, C:100000x1.
    # Greedy / optimal / auto must all pick A(BC) = [(1,2), (0,1)],
    # not the naive (AB)C = [(0,1), (0,1)].
    var eq = parse(String("ij,jk,kl->il"))
    var shapes = List[List[Int]]()
    var s0 = List[Int]()
    s0.append(100)
    s0.append(1)
    shapes.append(s0^)
    var s1 = List[Int]()
    s1.append(1)
    s1.append(100000)
    shapes.append(s1^)
    var s2 = List[Int]()
    s2.append(100000)
    s2.append(1)
    shapes.append(s2^)
    var sizes = _resolve_label_sizes(eq, shapes)
    var path = compute_path(eq, sizes, String("greedy"))
    if len(path) != 2:
        raise Error(String("greedy: expected 2 steps, got ", len(path)))
    if path[0].lhs_idx != 1 or path[0].rhs_idx != 2:
        raise Error(
            String(
                "greedy: expected step 0 = (1, 2), got (",
                path[0].lhs_idx,
                ", ",
                path[0].rhs_idx,
                ")",
            )
        )
    if path[1].lhs_idx != 0 or path[1].rhs_idx != 1:
        raise Error(String("greedy: expected step 1 = (0, 1)"))
    print("check_path_greedy: OK")


def check_path_branch() raises:
    # 4-operand matrix chain — exercise branch-{all,2,1} dispatch.
    # branch-1 must equal greedy by construction. branch-all and
    # branch-2 must produce a valid path (3 steps for n=4).
    var eq = parse(String("ab,bc,cd,de->ae"))
    var shapes = List[List[Int]]()
    var s0 = List[Int]()
    s0.append(3)
    s0.append(4)
    shapes.append(s0^)
    var s1 = List[Int]()
    s1.append(4)
    s1.append(5)
    shapes.append(s1^)
    var s2 = List[Int]()
    s2.append(5)
    s2.append(6)
    shapes.append(s2^)
    var s3 = List[Int]()
    s3.append(6)
    s3.append(7)
    shapes.append(s3^)
    var sizes = _resolve_label_sizes(eq, shapes)

    var greedy = compute_path(eq, sizes, String("greedy"))
    var branch_all = compute_path(eq, sizes, String("branch-all"))
    var branch_2 = compute_path(eq, sizes, String("branch-2"))
    var branch_1 = compute_path(eq, sizes, String("branch-1"))

    if len(branch_all) != 3:
        raise Error(String("branch-all: expected 3 steps, got ", len(branch_all)))
    if len(branch_2) != 3:
        raise Error(String("branch-2: expected 3 steps, got ", len(branch_2)))
    if len(branch_1) != len(greedy):
        raise Error(String("branch-1: step count differs from greedy"))
    for i in range(len(branch_1)):
        if (
            branch_1[i].lhs_idx != greedy[i].lhs_idx
            or branch_1[i].rhs_idx != greedy[i].rhs_idx
        ):
            raise Error(String("branch-1 differs from greedy at step ", i))
    print("check_path_branch: OK")


def main() raises:
    check_basic()
    check_trace()
    check_implicit_output()
    check_ellipsis()
    check_naive_plan()
    check_path_greedy()
    check_path_branch()
    print("all parser smoke tests passed")
