"""Mojo-side parser smoke test.

Run with:
    mojo run -I src tests/mojo/smoke_parse.mojo

Exercises parse() + expand_ellipsis() on a curated set of equations
without touching Python or MAX. The first build target to make work.
"""

from einsum.parse import parse, expand_ellipsis, ELLIPSIS_LABEL, EinsumEquation
from einsum.plan import build_naive_plan, ContractionPlan
from einsum.path import compute_path, ContractionStep
from einsum.backends.reference import _resolve_label_sizes, compute_output_shape
# Import-only checks - confirms backend symbols compile.
from einsum.backends.max import execute_max
from einsum.backends.native import execute_native
from std.memory import UnsafePointer
from std.memory.unsafe_pointer import alloc


def _row_major_strides(shape: List[Int]) -> List[Int]:
    var rank = len(shape)
    var strides = List[Int]()
    for _ in range(rank):
        strides.append(0)
    if rank == 0:
        return strides^
    strides[rank - 1] = 1
    var axis = rank - 2
    while axis >= 0:
        strides[axis] = strides[axis + 1] * shape[axis + 1]
        axis -= 1
    return strides^


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
    ranks.append(4)  # ...ij with 4-D input -> ellipsis is 2 dims
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
    # 4-operand matrix chain - exercise branch-{all,2,1} dispatch.
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


def check_path_random_greedy_n() raises:
    # `random-greedy-N` must parse the trailing digits and dispatch to
    # random_greedy_path with N trials. On the Bellman chain, every
    # trial count must agree with greedy/optimal.
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

    var optimal = compute_path(eq, sizes, String("optimal"))
    var rg1 = compute_path(eq, sizes, String("random-greedy-1"))
    var rg64 = compute_path(eq, sizes, String("random-greedy-64"))

    if len(rg1) != 2 or len(rg64) != 2:
        raise Error(String("random-greedy-N: expected 2 steps"))
    if (
        rg1[0].lhs_idx != optimal[0].lhs_idx
        or rg1[0].rhs_idx != optimal[0].rhs_idx
    ):
        raise Error(String("random-greedy-1: disagrees with optimal on Bellman"))
    if (
        rg64[0].lhs_idx != optimal[0].lhs_idx
        or rg64[0].rhs_idx != optimal[0].rhs_idx
    ):
        raise Error(String("random-greedy-64: disagrees with optimal on Bellman"))
    print("check_path_random_greedy_n: OK")


def check_path_invalid_random_greedy() raises:
    # Suffix must be numeric and >= 1.
    var eq = parse(String("ij,jk->ik"))
    var shapes = List[List[Int]]()
    var s0 = List[Int]()
    s0.append(2)
    s0.append(3)
    shapes.append(s0^)
    var s1 = List[Int]()
    s1.append(3)
    s1.append(4)
    shapes.append(s1^)
    var sizes = _resolve_label_sizes(eq, shapes)

    # `random-greedy-0` must raise.
    var raised = False
    try:
        var _path = compute_path(eq, sizes, String("random-greedy-0"))
    except:
        raised = True
    if not raised:
        raise Error(String("random-greedy-0 should have raised"))

    # `random-greedy-abc` must raise.
    raised = False
    try:
        var _path = compute_path(eq, sizes, String("random-greedy-abc"))
    except:
        raised = True
    if not raised:
        raise Error(String("random-greedy-abc should have raised"))

    print("check_path_invalid_random_greedy: OK")


def check_max_backend_matmul() raises:
    var eq = parse(String("ij,jk->ik"))
    var plan = build_naive_plan(eq)

    var shapes = List[List[Int]]()
    var s0 = List[Int]()
    s0.append(2)
    s0.append(3)
    shapes.append(s0^)
    var s1 = List[Int]()
    s1.append(3)
    s1.append(2)
    shapes.append(s1^)

    var a = alloc[Float64](6)
    var b = alloc[Float64](6)
    var out = alloc[Float64](4)
    for i in range(6):
        a[i] = Float64(i + 1)
        b[i] = Float64(i + 7)
    for i in range(4):
        out[i] = 0.0

    var data = List[UnsafePointer[Float64, MutAnyOrigin]]()
    data.append(a)
    data.append(b)

    var strides = List[List[Int]]()
    strides.append(_row_major_strides(shapes[0]))
    strides.append(_row_major_strides(shapes[1]))

    var out_shape = compute_output_shape(eq, shapes)
    var out_strides = _row_major_strides(out_shape)
    execute_max(plan, data, shapes, strides, out, out_shape, out_strides)

    if out[0] != 58.0 or out[1] != 64.0 or out[2] != 139.0 or out[3] != 154.0:
        raise Error(
            String(
                "max backend matmul mismatch: [",
                out[0],
                ", ",
                out[1],
                ", ",
                out[2],
                ", ",
                out[3],
                "]",
            )
        )

    for i in range(4):
        out[i] = 0.0
    execute_native(plan, data, shapes, strides, out, out_shape, out_strides)
    if out[0] != 58.0 or out[1] != 64.0 or out[2] != 139.0 or out[3] != 154.0:
        raise Error(String("native backend matmul mismatch"))

    a.free()
    b.free()
    out.free()
    print("check_max_backend_matmul: OK")
    print("check_native_backend_matmul: OK")


def check_max_backend_unary_transpose() raises:
    var eq = parse(String("ij->ji"))
    var plan = build_naive_plan(eq)

    var shapes = List[List[Int]]()
    var s0 = List[Int]()
    s0.append(2)
    s0.append(3)
    shapes.append(s0^)

    var a = alloc[Float64](6)
    var out = alloc[Float64](6)
    for i in range(6):
        a[i] = Float64(i + 1)
        out[i] = 0.0

    var data = List[UnsafePointer[Float64, MutAnyOrigin]]()
    data.append(a)
    var strides = List[List[Int]]()
    strides.append(_row_major_strides(shapes[0]))

    var out_shape = compute_output_shape(eq, shapes)
    var out_strides = _row_major_strides(out_shape)
    execute_max(plan, data, shapes, strides, out, out_shape, out_strides)

    if (
        out[0] != 1.0
        or out[1] != 4.0
        or out[2] != 2.0
        or out[3] != 5.0
        or out[4] != 3.0
        or out[5] != 6.0
    ):
        raise Error(String("max backend unary transpose mismatch"))

    for i in range(6):
        out[i] = 0.0
    execute_native(plan, data, shapes, strides, out, out_shape, out_strides)
    if (
        out[0] != 1.0
        or out[1] != 4.0
        or out[2] != 2.0
        or out[3] != 5.0
        or out[4] != 3.0
        or out[5] != 6.0
    ):
        raise Error(String("native backend unary transpose mismatch"))

    a.free()
    out.free()
    print("check_max_backend_unary_transpose: OK")
    print("check_native_backend_unary_transpose: OK")


def check_max_backend_final_transpose() raises:
    var eq = parse(String("ji,jk->ki"))
    var plan = build_naive_plan(eq)

    var shapes = List[List[Int]]()
    var s0 = List[Int]()
    s0.append(3)
    s0.append(2)
    shapes.append(s0^)
    var s1 = List[Int]()
    s1.append(3)
    s1.append(5)
    shapes.append(s1^)

    var a = alloc[Float64](6)
    var b = alloc[Float64](15)
    var out = alloc[Float64](10)
    for i in range(6):
        a[i] = Float64(i + 1)
    for i in range(15):
        b[i] = Float64(i + 7)
    for i in range(10):
        out[i] = 0.0

    var data = List[UnsafePointer[Float64, MutAnyOrigin]]()
    data.append(a)
    data.append(b)

    var strides = List[List[Int]]()
    strides.append(_row_major_strides(shapes[0]))
    strides.append(_row_major_strides(shapes[1]))

    var out_shape = compute_output_shape(eq, shapes)
    var out_strides = _row_major_strides(out_shape)
    execute_max(plan, data, shapes, strides, out, out_shape, out_strides)

    if (
        out[0] != 128.0
        or out[1] != 164.0
        or out[2] != 137.0
        or out[3] != 176.0
        or out[4] != 146.0
        or out[5] != 188.0
        or out[6] != 155.0
        or out[7] != 200.0
        or out[8] != 164.0
        or out[9] != 212.0
    ):
        raise Error(String("max backend final transpose mismatch"))

    for i in range(10):
        out[i] = 0.0
    execute_native(plan, data, shapes, strides, out, out_shape, out_strides)
    if (
        out[0] != 128.0
        or out[1] != 164.0
        or out[2] != 137.0
        or out[3] != 176.0
        or out[4] != 146.0
        or out[5] != 188.0
        or out[6] != 155.0
        or out[7] != 200.0
        or out[8] != 164.0
        or out[9] != 212.0
    ):
        raise Error(String("native backend final transpose mismatch"))

    a.free()
    b.free()
    out.free()
    print("check_max_backend_final_transpose: OK")
    print("check_native_backend_final_transpose: OK")


def main() raises:
    check_basic()
    check_trace()
    check_implicit_output()
    check_ellipsis()
    check_naive_plan()
    check_path_greedy()
    check_path_branch()
    check_path_random_greedy_n()
    check_path_invalid_random_greedy()
    check_max_backend_matmul()
    check_max_backend_unary_transpose()
    check_max_backend_final_transpose()
    _ = execute_max
    _ = execute_native
    print("all parser smoke tests passed")
