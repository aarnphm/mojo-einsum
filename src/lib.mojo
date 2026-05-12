"""Entrypoint for `moeinsum._native`.

Exposed callables consumed by the Python wrapper:
  - `parse_equation(eq: str) -> dict`, for IR/debug introspection.
  - `einsum_reference(eq, operands_flat, operand_shapes) -> (flat_out, shape)`,
    the row-major flat-list reference backend.
  - `einsum_path(eq, operand_shapes) -> list[tuple[int, ...]]`, the naive
    left-to-right plan builder path.
  - `einsum_compute_path(eq, operand_shapes, algorithm) -> list[tuple[int, int]]`,
    the named path optimizer.
  - `max_graph_spec(eq, operand_shapes, path) -> dict`, the Mojo-owned
    plan-to-graph description used by MAX debug/lowering tools.
"""

from std.os import abort
from std.python import PythonObject, Python
from std.python.bindings import PythonModuleBuilder
from std.memory import UnsafePointer
from std.memory.unsafe_pointer import alloc

from einsum.parse import parse, expand_ellipsis, EinsumEquation
from einsum.plan import (
    UnaryStep,
    PairwiseStep,
    build_naive_plan,
    classify_pair,
)
from einsum.path import compute_path
from einsum.backends.reference import (
    execute_reference,
    compute_output_shape,
    _resolve_label_sizes,
)


@export
def PyInit__native() -> PythonObject:
    try:
        var module = PythonModuleBuilder("_native")
        module.def_function[parse_equation_py]("parse_equation")
        module.def_function[einsum_reference_py]("einsum_reference")
        module.def_function[einsum_path_py]("einsum_path")
        module.def_function[einsum_compute_path_py]("einsum_compute_path")
        module.def_function[max_graph_spec_py]("max_graph_spec")
        return module.finalize()
    except e:
        abort(String("failed to create _native module: ", e))


# ---------------------------------------------------------------------
# parse_equation - IR introspection
# ---------------------------------------------------------------------


def parse_equation_py(eq_obj: PythonObject) raises -> PythonObject:
    var eq_str = String(py=eq_obj)
    var eq = parse(eq_str)

    var d = Python.dict()

    var inputs_py = Python.evaluate("[]")
    for op_idx in range(len(eq.inputs)):
        var inner = Python.evaluate("[]")
        ref op = eq.inputs[op_idx]
        for j in range(len(op)):
            inner.append(PythonObject(op[j]))
        inputs_py.append(inner)
    d["inputs"] = inputs_py

    var output_py = Python.evaluate("[]")
    for j in range(len(eq.output)):
        output_py.append(PythonObject(eq.output[j]))
    d["output"] = output_py

    d["n_labels"] = PythonObject(eq.n_labels)
    d["has_explicit_output"] = PythonObject(eq.has_explicit_output)

    var label_chars_py = Python.evaluate("[]")
    for j in range(len(eq.label_chars)):
        label_chars_py.append(PythonObject(eq.label_chars[j]))
    d["label_chars"] = label_chars_py

    return d


# ---------------------------------------------------------------------
# einsum_reference - flat-list FFI for the reference backend
# ---------------------------------------------------------------------


def _pylist_shapes_to_mojo(
    shapes_obj: PythonObject,
) raises -> List[List[Int]]:
    var out = List[List[Int]]()
    var n = Int(len(shapes_obj))
    for i in range(n):
        var shape_obj = shapes_obj[i]
        var rank = Int(len(shape_obj))
        var shape = List[Int]()
        for j in range(rank):
            shape.append(Int(py=shape_obj[j]))
        out.append(shape^)
    return out^


def _py_path_to_mojo(path_obj: PythonObject) raises -> List[List[Int]]:
    """Convert a Python path list/tuple into Mojo working-set steps."""
    var out = List[List[Int]]()
    var n = Int(len(path_obj))
    for i in range(n):
        var step_obj = path_obj[i]
        var step = List[Int]()
        var m = Int(len(step_obj))
        if m != 1 and m != 2:
            raise Error(
                String(
                    "max_graph_spec: path step ",
                    i,
                    " has arity ",
                    m,
                    ", expected 1 or 2",
                )
            )
        for j in range(m):
            step.append(Int(py=step_obj[j]))
        out.append(step^)
    return out^


def _label_in(labels: List[Int], lbl: Int) -> Bool:
    for i in range(len(labels)):
        if labels[i] == lbl:
            return True
    return False


def _labels_equal(lhs: List[Int], rhs: List[Int]) -> Bool:
    if len(lhs) != len(rhs):
        return False
    for i in range(len(lhs)):
        if lhs[i] != rhs[i]:
            return False
    return True


def _labels_to_string(labels: List[Int], label_chars: List[String]) -> String:
    var out = String()
    for i in range(len(labels)):
        out += label_chars[labels[i]]
    return out^


def _labels_from_axes_to_py_list(
    labels: List[Int],
    axes: List[Int],
    label_chars: List[String],
) raises -> PythonObject:
    var out = Python.evaluate("[]")
    for i in range(len(axes)):
        out.append(PythonObject(label_chars[labels[axes[i]]]))
    return out


def _has_repeated_labels(labels: List[Int]) -> Bool:
    var seen = List[Int]()
    for i in range(len(labels)):
        var lbl = labels[i]
        if _label_in(seen, lbl):
            return True
        seen.append(lbl)
    return False


def _dedupe_labels(labels: List[Int]) -> List[Int]:
    var out = List[Int]()
    for i in range(len(labels)):
        var lbl = labels[i]
        if not _label_in(out, lbl):
            out.append(lbl)
    return out^


def _filter_labels(labels: List[Int], keep: List[Int]) -> List[Int]:
    var out = List[Int]()
    for i in range(len(labels)):
        var lbl = labels[i]
        if _label_in(keep, lbl):
            out.append(lbl)
    return out^


def _step_output_labels_for_spec(
    lhs: List[Int],
    rhs: List[Int],
    future: List[Int],
) -> List[Int]:
    """Labels carried by a pairwise MAX graph step, preserving lhs+rhs order."""
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


def _row_major_strides(shape: List[Int]) -> List[Int]:
    """Row-major (C-order) strides, in *elements*."""
    var n = len(shape)
    var s = List[Int]()
    for _ in range(n):
        s.append(0)
    if n == 0:
        return s^
    s[n - 1] = 1
    var k = n - 2
    while k >= 0:
        s[k] = s[k + 1] * shape[k + 1]
        k -= 1
    return s^


def einsum_reference_py(
    eq_obj: PythonObject,
    operands_flat_obj: PythonObject,
    operand_shapes_obj: PythonObject,
) raises -> PythonObject:
    """Reference einsum over flat Python lists.

    Returns a 2-tuple `(flat_output_list, output_shape)`.
    """
    var eq_str = String(py=eq_obj)
    var eq = parse(eq_str)

    var operand_shapes = _pylist_shapes_to_mojo(operand_shapes_obj)

    var operand_ranks = List[Int]()
    for i in range(len(operand_shapes)):
        operand_ranks.append(len(operand_shapes[i]))
    expand_ellipsis(eq, operand_ranks)

    if eq.n_operands() != Int(len(operands_flat_obj)):
        raise Error(
            String(
                "einsum: equation has ",
                eq.n_operands(),
                " operands but got ",
                Int(len(operands_flat_obj)),
                " arrays",
            )
        )

    # Copy each operand into a Mojo-owned Float64 buffer. Zero-copy DLPack is
    # later FFI work.
    var data_ptrs = List[UnsafePointer[Float64, MutAnyOrigin]]()
    var strides_list = List[List[Int]]()
    for op_idx in range(eq.n_operands()):
        var flat = operands_flat_obj[op_idx]
        var n_elems = Int(len(flat))
        var expected_elems: Int = 1
        for d_idx in range(len(operand_shapes[op_idx])):
            expected_elems *= operand_shapes[op_idx][d_idx]
        if n_elems != expected_elems:
            raise Error(
                String(
                    "einsum: operand ",
                    op_idx,
                    " flat list length ",
                    n_elems,
                    " != shape product ",
                    expected_elems,
                )
            )
        var ptr = alloc[Float64](n_elems)
        for i in range(n_elems):
            ptr[i] = Float64(py=flat[i])
        data_ptrs.append(ptr)
        strides_list.append(_row_major_strides(operand_shapes[op_idx]))

    var out_shape = compute_output_shape(eq, operand_shapes)
    var out_n: Int = 1
    for i in range(len(out_shape)):
        out_n *= out_shape[i]
    var out_alloc_n = out_n if out_n > 0 else 1
    var out_ptr = alloc[Float64](out_alloc_n)
    for i in range(out_alloc_n):
        out_ptr[i] = 0.0

    var out_strides = _row_major_strides(out_shape)

    execute_reference(
        eq,
        data_ptrs,
        operand_shapes,
        strides_list,
        out_ptr,
        out_strides,
    )

    var flat_out_py = Python.evaluate("[]")
    for i in range(out_alloc_n):
        flat_out_py.append(PythonObject(out_ptr[i]))

    var shape_py = Python.evaluate("[]")
    for i in range(len(out_shape)):
        shape_py.append(PythonObject(out_shape[i]))

    for i in range(len(data_ptrs)):
        data_ptrs[i].free()
    out_ptr.free()

    return Python.tuple(flat_out_py, shape_py)


# ---------------------------------------------------------------------
# max_graph_spec - Mojo-owned plan-to-graph description
# ---------------------------------------------------------------------


def _reject_max_graph_ellipsis(eq: EinsumEquation) raises:
    for op_idx in range(eq.n_operands()):
        ref labels = eq.inputs[op_idx]
        for i in range(len(labels)):
            if labels[i] == -1:
                raise Error(String("max_graph_spec does not support ellipsis"))
    for i in range(len(eq.output)):
        if eq.output[i] == -1:
            raise Error(String("max_graph_spec does not support ellipsis"))


def _validate_operand_shapes_for_spec(
    eq: EinsumEquation,
    operand_shapes: List[List[Int]],
) raises:
    if len(operand_shapes) != eq.n_operands():
        raise Error(
            String(
                "max_graph_spec: equation has ",
                eq.n_operands(),
                " operands but got ",
                len(operand_shapes),
                " shapes",
            )
        )
    for op_idx in range(eq.n_operands()):
        if len(eq.inputs[op_idx]) != len(operand_shapes[op_idx]):
            raise Error(
                String(
                    "max_graph_spec: operand ",
                    op_idx,
                    " has ",
                    len(eq.inputs[op_idx]),
                    " labels but shape rank ",
                    len(operand_shapes[op_idx]),
                )
            )


def _append_diagonal_spec(
    mut ops: PythonObject,
    step_idx: Int,
    operand_idx: Int,
    src_labels: List[Int],
    dst_labels: List[Int],
    label_chars: List[String],
) raises:
    var payload = Python.dict()
    payload["step"] = PythonObject(step_idx)
    payload["operand"] = PythonObject(operand_idx)
    payload["src_labels"] = PythonObject(
        _labels_to_string(src_labels, label_chars)
    )
    payload["dst_labels"] = PythonObject(
        _labels_to_string(dst_labels, label_chars)
    )
    ops.append(Python.tuple(PythonObject("diagonal"), payload))


def _append_reduce_sum_spec(
    mut ops: PythonObject,
    step_idx: Int,
    operand_idx: Int,
    src_labels: List[Int],
    dst_labels: List[Int],
    label_chars: List[String],
) raises:
    var payload = Python.dict()
    payload["step"] = PythonObject(step_idx)
    payload["operand"] = PythonObject(operand_idx)
    payload["src_labels"] = PythonObject(
        _labels_to_string(src_labels, label_chars)
    )
    payload["dst_labels"] = PythonObject(
        _labels_to_string(dst_labels, label_chars)
    )
    ops.append(Python.tuple(PythonObject("reduce_sum"), payload))


def _append_matmul_spec(
    mut ops: PythonObject,
    step_idx: Int,
    lhs_idx: Int,
    rhs_idx: Int,
    lhs_labels: List[Int],
    rhs_labels: List[Int],
    out_labels: List[Int],
    label_chars: List[String],
) raises:
    var cls = classify_pair(
        lhs_labels.copy(), rhs_labels.copy(), out_labels.copy()
    )
    var payload = Python.dict()
    payload["step"] = PythonObject(step_idx)
    payload["lhs"] = PythonObject(lhs_idx)
    payload["rhs"] = PythonObject(rhs_idx)
    payload["lhs_labels"] = PythonObject(
        _labels_to_string(lhs_labels, label_chars)
    )
    payload["rhs_labels"] = PythonObject(
        _labels_to_string(rhs_labels, label_chars)
    )
    payload["out_labels"] = PythonObject(
        _labels_to_string(out_labels, label_chars)
    )
    payload["batch"] = _labels_from_axes_to_py_list(
        lhs_labels, cls.batch_axes_lhs, label_chars
    )
    payload["contract"] = _labels_from_axes_to_py_list(
        lhs_labels, cls.contract_axes_lhs, label_chars
    )
    payload["free_lhs"] = _labels_from_axes_to_py_list(
        lhs_labels, cls.free_axes_lhs, label_chars
    )
    payload["free_rhs"] = _labels_from_axes_to_py_list(
        rhs_labels, cls.free_axes_rhs, label_chars
    )
    ops.append(Python.tuple(PythonObject("matmul"), payload))


def _append_transpose_spec(
    mut ops: PythonObject,
    step_idx: Int,
    src_labels: List[Int],
    dst_labels: List[Int],
    label_chars: List[String],
) raises:
    var payload = Python.dict()
    payload["step"] = PythonObject(step_idx)
    payload["src_labels"] = PythonObject(
        _labels_to_string(src_labels, label_chars)
    )
    payload["dst_labels"] = PythonObject(
        _labels_to_string(dst_labels, label_chars)
    )
    ops.append(Python.tuple(PythonObject("transpose"), payload))


def max_graph_spec_py(
    eq_obj: PythonObject,
    operand_shapes_obj: PythonObject,
    path_obj: PythonObject,
) raises -> PythonObject:
    """Return the MAX Graph op spec for `(eq, shapes, path)`.

    Python still owns actual `max.graph.Graph` object construction because that
    compiler API is Python-facing. The contraction semantics live here: parsing,
    working-set path semantics, diagonal/reduce detection, and B/K/M/N
    classification all route through the Mojo IR.
    """
    var eq_str = String(py=eq_obj)
    var eq = parse(eq_str)
    _reject_max_graph_ellipsis(eq)

    var operand_shapes = _pylist_shapes_to_mojo(operand_shapes_obj)
    _validate_operand_shapes_for_spec(eq, operand_shapes)
    var path = _py_path_to_mojo(path_obj)

    var working = List[List[Int]]()
    for i in range(eq.n_operands()):
        working.append(eq.inputs[i].copy())

    var ops = Python.evaluate("[]")
    var step_idx = 0
    for path_idx in range(len(path)):
        ref step = path[path_idx]
        if len(step) == 1:
            var idx = step[0]
            if idx < 0 or idx >= len(working):
                raise Error(
                    String(
                        "max_graph_spec: unary path index ",
                        idx,
                        " out of range",
                    )
                )
            var labels = working[idx].copy()
            var future = eq.output.copy()
            for j in range(len(working)):
                if j == idx:
                    continue
                ref other = working[j]
                for k in range(len(other)):
                    if not _label_in(future, other[k]):
                        future.append(other[k])

            if _has_repeated_labels(labels):
                var deduped = _dedupe_labels(labels)
                _append_diagonal_spec(
                    ops, step_idx, idx, labels, deduped, eq.label_chars
                )
                labels = deduped^

            var survived = _filter_labels(labels, future)
            if not _labels_equal(labels, survived):
                _append_reduce_sum_spec(
                    ops, step_idx, idx, labels, survived, eq.label_chars
                )
                labels = survived^

            working[idx] = labels^
            step_idx += 1
            continue

        var li = step[0]
        var ri = step[1]
        if li < 0 or li >= len(working) or ri < 0 or ri >= len(working):
            raise Error(
                String("max_graph_spec: pairwise path index out of range")
            )
        if li == ri:
            raise Error(
                String("max_graph_spec: pairwise path indices are equal")
            )

        var lhs = working[li].copy()
        var rhs = working[ri].copy()
        var future = eq.output.copy()
        for j in range(len(working)):
            if j == li or j == ri:
                continue
            ref other = working[j]
            for k in range(len(other)):
                if not _label_in(future, other[k]):
                    future.append(other[k])

        var out_labels = _step_output_labels_for_spec(lhs, rhs, future)
        _append_matmul_spec(
            ops, step_idx, li, ri, lhs, rhs, out_labels, eq.label_chars
        )

        var new_working = List[List[Int]]()
        for j in range(len(working)):
            if j != li and j != ri:
                new_working.append(working[j].copy())
        new_working.append(out_labels^)
        working = new_working^
        step_idx += 1

    if len(working) == 1 and not _labels_equal(working[0], eq.output):
        _append_transpose_spec(
            ops, step_idx, working[0], eq.output, eq.label_chars
        )
        step_idx += 1

    var result = Python.dict()
    result["ops"] = ops
    result["result_index"] = PythonObject(step_idx - 1)
    return result


# ---------------------------------------------------------------------
# einsum_path - pair sequence introspection
# ---------------------------------------------------------------------


def einsum_path_py(
    eq_obj: PythonObject,
    operand_shapes_obj: PythonObject,
) raises -> PythonObject:
    """Return the naive (left-to-right) pair sequence from build_naive_plan.

    Each tuple is `(lhs_idx, rhs_idx)` for a pairwise step or `(idx,)`
    for a unary step, where indices refer to the working-set position
    at the time of that step.
    """
    var eq_str = String(py=eq_obj)
    var eq = parse(eq_str)

    var operand_shapes = _pylist_shapes_to_mojo(operand_shapes_obj)
    var operand_ranks = List[Int]()
    for i in range(len(operand_shapes)):
        operand_ranks.append(len(operand_shapes[i]))
    expand_ellipsis(eq, operand_ranks)

    var plan = build_naive_plan(eq)

    var pairs = Python.evaluate("[]")
    for step_idx in range(len(plan.steps)):
        var step = plan.steps[step_idx].copy()
        if step.isa[PairwiseStep]():
            var ps = step.unsafe_get[PairwiseStep]().copy()
            pairs.append(
                Python.tuple(PythonObject(ps.lhs_idx), PythonObject(ps.rhs_idx))
            )
        else:
            var us = step.unsafe_get[UnaryStep]().copy()
            pairs.append(Python.tuple(PythonObject(us.operand_idx)))
    return pairs


def einsum_compute_path_py(
    eq_obj: PythonObject,
    operand_shapes_obj: PythonObject,
    algorithm_obj: PythonObject,
) raises -> PythonObject:
    """Run `path.mojo`'s named algorithm and return its pair sequence.

    Supported algorithms match `compute_path`: greedy, optimal, auto, naive,
    branch-all, branch-2, branch-1, random-greedy, and random-greedy-N. Output is
    a list of `(lhs_idx, rhs_idx)` tuples, pairwise-only. Single-operand
    contractions go through `einsum_path`, which uses `build_naive_plan`.
    """
    var eq_str = String(py=eq_obj)
    var algorithm = String(py=algorithm_obj)
    var eq = parse(eq_str)

    var operand_shapes = _pylist_shapes_to_mojo(operand_shapes_obj)
    var operand_ranks = List[Int]()
    for i in range(len(operand_shapes)):
        operand_ranks.append(len(operand_shapes[i]))
    expand_ellipsis(eq, operand_ranks)

    var label_sizes = _resolve_label_sizes(eq, operand_shapes)
    var steps = compute_path(eq, label_sizes, algorithm)

    var pairs = Python.evaluate("[]")
    for k in range(len(steps)):
        pairs.append(
            Python.tuple(
                PythonObject(steps[k].lhs_idx),
                PythonObject(steps[k].rhs_idx),
            )
        )
    return pairs
