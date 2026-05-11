"""Mojo-side unit tests for the path-cost helpers.

Run with:
    mojo run -I src tests/mojo/smoke_path_cost.mojo

Plan §-Outstanding (2) — currently `_flop_cost` and `_reduced_size_cost`
in `src/einsum/path.mojo` are exercised end-to-end via the path
optimizer. These direct unit tests catch regressions in the cost
helpers themselves without round-tripping through the planner.

The helpers are intentionally simple (label-set product, four-line
arithmetic), so the tests are mostly about pinning the contract:

  - `_tensor_size(labels, sizes)` = product of label sizes; empty
    label list returns 1 (scalar tensor).
  - `_reduced_size_cost(lhs, rhs, out, sizes)` = a + b - c where
    `a = size(lhs)`, `b = size(rhs)`, `c = size(out)`. Bigger = more
    memory removed by the contraction = better candidate.
  - `_flop_cost(lhs, rhs, out, sizes)` = product of all label sizes
    in `lhs ∪ rhs` — the natural nested-loop bound.

We import the private helpers directly. Underscore prefix is a Mojo
convention, not enforced; if a future refactor makes them truly
private, switch to testing through `compute_path` with hand-verified
expected paths.
"""

from einsum.path import (
    _flop_cost,
    _reduced_size_cost,
    _tensor_size,
    _label_set_union,
    _label_set_intersect,
)


def check_tensor_size_empty() raises:
    """Empty label list = scalar tensor (size 1)."""
    var labels = List[Int]()
    var sizes = List[Int]()
    var n = _tensor_size(labels, sizes)
    if n != 1:
        raise Error(String("tensor_size(empty) expected 1, got ", n))
    print("check_tensor_size_empty: OK")


def check_tensor_size_simple() raises:
    """[0, 1] with sizes [3, 4] → 12."""
    var labels = List[Int]()
    labels.append(0)
    labels.append(1)
    var sizes = List[Int]()
    sizes.append(3)
    sizes.append(4)
    var n = _tensor_size(labels, sizes)
    if n != 12:
        raise Error(String("tensor_size([0,1], [3,4]) expected 12, got ", n))
    print("check_tensor_size_simple: OK")


def check_tensor_size_rank3() raises:
    """[0, 1, 2] with sizes [2, 3, 5] → 30."""
    var labels = List[Int]()
    labels.append(0)
    labels.append(1)
    labels.append(2)
    var sizes = List[Int]()
    sizes.append(2)
    sizes.append(3)
    sizes.append(5)
    var n = _tensor_size(labels, sizes)
    if n != 30:
        raise Error(String("tensor_size([0,1,2], [2,3,5]) expected 30, got ", n))
    print("check_tensor_size_rank3: OK")


def check_flop_cost_matmul() raises:
    """`ij,jk->ik` with i=3, j=5, k=4: 3*5*4 = 60 FLOPs."""
    var lhs = List[Int]()
    lhs.append(0)  # i
    lhs.append(1)  # j
    var rhs = List[Int]()
    rhs.append(1)  # j
    rhs.append(2)  # k
    var out = List[Int]()
    out.append(0)  # i
    out.append(2)  # k
    var sizes = List[Int]()
    sizes.append(3)  # i
    sizes.append(5)  # j
    sizes.append(4)  # k
    var f = _flop_cost(lhs, rhs, out, sizes)
    if f != 60:
        raise Error(String("flop_cost(matmul 3x5*5x4) expected 60, got ", f))
    print("check_flop_cost_matmul: OK")


def check_flop_cost_bmm() raises:
    """`bij,bjk->bik` with b=2, i=3, j=5, k=4: 2*3*5*4 = 120 FLOPs."""
    var lhs = List[Int]()
    lhs.append(0)  # b
    lhs.append(1)  # i
    lhs.append(2)  # j
    var rhs = List[Int]()
    rhs.append(0)  # b
    rhs.append(2)  # j
    rhs.append(3)  # k
    var out = List[Int]()
    out.append(0)  # b
    out.append(1)  # i
    out.append(3)  # k
    var sizes = List[Int]()
    sizes.append(2)  # b
    sizes.append(3)  # i
    sizes.append(5)  # j
    sizes.append(4)  # k
    var f = _flop_cost(lhs, rhs, out, sizes)
    if f != 120:
        raise Error(String("flop_cost(bmm) expected 120, got ", f))
    print("check_flop_cost_bmm: OK")


def check_flop_cost_frobenius() raises:
    """`ij,ij->` with i=4, j=6: 4*6 = 24 FLOPs (no extra k dim)."""
    var lhs = List[Int]()
    lhs.append(0)
    lhs.append(1)
    var rhs = List[Int]()
    rhs.append(0)
    rhs.append(1)
    var out = List[Int]()
    var sizes = List[Int]()
    sizes.append(4)
    sizes.append(6)
    var f = _flop_cost(lhs, rhs, out, sizes)
    if f != 24:
        raise Error(String("flop_cost(frobenius) expected 24, got ", f))
    print("check_flop_cost_frobenius: OK")


def check_reduced_size_matmul() raises:
    """`ij,jk->ik` with i=3, j=5, k=4:
        a = i*j = 15, b = j*k = 20, c = i*k = 12 → 15+20-12 = 23.
    """
    var lhs = List[Int]()
    lhs.append(0)
    lhs.append(1)
    var rhs = List[Int]()
    rhs.append(1)
    rhs.append(2)
    var out = List[Int]()
    out.append(0)
    out.append(2)
    var sizes = List[Int]()
    sizes.append(3)
    sizes.append(5)
    sizes.append(4)
    var r = _reduced_size_cost(lhs, rhs, out, sizes)
    if r != 23:
        raise Error(String("reduced_size(matmul) expected 23, got ", r))
    print("check_reduced_size_matmul: OK")


def check_reduced_size_outer_product() raises:
    """`i,j->ij` with i=3, j=5:
        a = 3, b = 5, c = 15 → 3+5-15 = -7 (outer products *grow* memory).
    """
    var lhs = List[Int]()
    lhs.append(0)
    var rhs = List[Int]()
    rhs.append(1)
    var out = List[Int]()
    out.append(0)
    out.append(1)
    var sizes = List[Int]()
    sizes.append(3)
    sizes.append(5)
    var r = _reduced_size_cost(lhs, rhs, out, sizes)
    if r != -7:
        raise Error(String("reduced_size(outer) expected -7, got ", r))
    print("check_reduced_size_outer_product: OK")


def check_reduced_size_frobenius_is_positive() raises:
    """`ij,ij->` with i=4, j=6:
        a = 24, b = 24, c = 1 → 47 (full reduction is the best move).
    """
    var lhs = List[Int]()
    lhs.append(0)
    lhs.append(1)
    var rhs = List[Int]()
    rhs.append(0)
    rhs.append(1)
    var out = List[Int]()
    var sizes = List[Int]()
    sizes.append(4)
    sizes.append(6)
    var r = _reduced_size_cost(lhs, rhs, out, sizes)
    if r != 47:
        raise Error(String("reduced_size(frobenius) expected 47, got ", r))
    print("check_reduced_size_frobenius_is_positive: OK")


def check_label_set_union_dedup() raises:
    """Union of [0,1,2] and [1,2,3] = [0,1,2,3] (order preserved)."""
    var a = List[Int]()
    a.append(0)
    a.append(1)
    a.append(2)
    var b = List[Int]()
    b.append(1)
    b.append(2)
    b.append(3)
    var u = _label_set_union(a, b)
    if len(u) != 4:
        raise Error(String("union: expected length 4, got ", len(u)))
    if u[0] != 0 or u[1] != 1 or u[2] != 2 or u[3] != 3:
        raise Error(String("union: order wrong, got [", u[0], ",", u[1], ",", u[2], ",", u[3], "]"))
    print("check_label_set_union_dedup: OK")


def check_label_set_intersect() raises:
    """Intersect of [0,1,2] and [1,2,3] = [1,2]."""
    var a = List[Int]()
    a.append(0)
    a.append(1)
    a.append(2)
    var b = List[Int]()
    b.append(1)
    b.append(2)
    b.append(3)
    var i = _label_set_intersect(a, b)
    if len(i) != 2:
        raise Error(String("intersect: expected length 2, got ", len(i)))
    if i[0] != 1 or i[1] != 2:
        raise Error(String("intersect: contents wrong, got [", i[0], ",", i[1], "]"))
    print("check_label_set_intersect: OK")


def main() raises:
    check_tensor_size_empty()
    check_tensor_size_simple()
    check_tensor_size_rank3()
    check_flop_cost_matmul()
    check_flop_cost_bmm()
    check_flop_cost_frobenius()
    check_reduced_size_matmul()
    check_reduced_size_outer_product()
    check_reduced_size_frobenius_is_positive()
    check_label_set_union_dedup()
    check_label_set_intersect()
    print("smoke_path_cost: all 11 checks passed")
