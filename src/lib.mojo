"""Entrypoint for `moeinsum._native`.

Exposed callables consumed by the Python wrapper:
  - `parse_equation(eq: str) -> dict`, for IR/debug introspection.
  - `einsum_reference(eq, operands_flat, operand_shapes) -> (flat_out, shape)`,
    the row-major flat-list reference backend.
  - `einsum_native(eq, operands_flat, operand_shapes, path) -> (flat_out, shape)`,
    the row-major flat-list native backend over a caller-supplied plan path.
  - `einsum_path(eq, operand_shapes) -> list[tuple[int, ...]]`, the naive
    left-to-right plan builder path.
  - `einsum_compute_path(eq, operand_shapes, algorithm) -> list[tuple[int, int]]`,
    the named path optimizer.
  - `path_cost(eq, operand_shapes, path) -> dict`, FLOP and peak-intermediate
    accounting for a working-set path.
"""

from std.os import abort
from std.python import PythonObject, Python
from std.python.bindings import PythonModuleBuilder
from std.memory import UnsafePointer
from std.memory.unsafe_pointer import alloc

from einsum.parse import parse, expand_ellipsis
from einsum.plan import (
    UnaryStep,
    PairwiseStep,
    build_naive_plan,
    build_plan_from_path,
)
from einsum.path import (
    compute_path,
    _flop_cost,
    _step_output_labels,
    _tensor_size,
)
from einsum.backends.reference import (
    execute_reference,
    compute_output_shape,
    _resolve_label_sizes,
)
from einsum.backends.native import execute_native


@export
def PyInit__native() -> PythonObject:
    try:
        var module = PythonModuleBuilder("_native")
        module.def_function[parse_equation_py]("parse_equation")
        module.def_function[einsum_reference_py]("einsum_reference")
        module.def_function[einsum_native_py]("einsum_native")
        module.def_function[einsum_path_py]("einsum_path")
        module.def_function[einsum_compute_path_py]("einsum_compute_path")
        module.def_function[path_cost_py]("path_cost")
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
                    "einsum path step ",
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


def _label_in_cost(labels: List[Int], lbl: Int) -> Bool:
    for i in range(len(labels)):
        if labels[i] == lbl:
            return True
    return False


def _dedupe_labels_for_cost(labels: List[Int]) -> List[Int]:
    var out = List[Int]()
    for i in range(len(labels)):
        var lbl = labels[i]
        if not _label_in_cost(out, lbl):
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


def einsum_native_py(
    eq_obj: PythonObject,
    operands_flat_obj: PythonObject,
    operand_shapes_obj: PythonObject,
    path_obj: PythonObject,
) raises -> PythonObject:
    """Native backend over flat Python lists.

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
                "einsum_native: equation has ",
                eq.n_operands(),
                " operands but got ",
                Int(len(operands_flat_obj)),
                " arrays",
            )
        )

    var path = _py_path_to_mojo(path_obj)
    var plan = build_plan_from_path(eq, path)

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
                    "einsum_native: operand ",
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

    execute_native(
        plan,
        data_ptrs,
        operand_shapes,
        strides_list,
        out_ptr,
        out_shape,
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
            pairs.append(Python.tuple(PythonObject(ps.lhs_idx), PythonObject(ps.rhs_idx)))
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


def path_cost_py(
    eq_obj: PythonObject,
    operand_shapes_obj: PythonObject,
    path_obj: PythonObject,
) raises -> PythonObject:
    """Return FLOP + peak-intermediate accounting for a working-set path."""
    var eq_str = String(py=eq_obj)
    var eq = parse(eq_str)

    var operand_shapes = _pylist_shapes_to_mojo(operand_shapes_obj)
    var operand_ranks = List[Int]()
    for i in range(len(operand_shapes)):
        operand_ranks.append(len(operand_shapes[i]))
    expand_ellipsis(eq, operand_ranks)

    var label_sizes = _resolve_label_sizes(eq, operand_shapes)
    var path = _py_path_to_mojo(path_obj)

    var working = List[List[Int]]()
    for i in range(eq.n_operands()):
        working.append(eq.inputs[i].copy())

    var total_flops: Int = 0
    var peak_intermediate: Int = 0
    var steps_py = Python.evaluate("[]")

    for path_idx in range(len(path)):
        ref raw_step = path[path_idx]
        if len(raw_step) == 1:
            var operand_idx = raw_step[0]
            if operand_idx < 0 or operand_idx >= len(working):
                raise Error(
                    String(
                        "path_cost: unary step ",
                        path_idx,
                        " operand index ",
                        operand_idx,
                        " out of range",
                    )
                )

            var labels = working[operand_idx].copy()
            var future = List[Int]()
            for i in range(len(eq.output)):
                var lbl = eq.output[i]
                if not _label_in_cost(future, lbl):
                    future.append(lbl)
            for i in range(len(working)):
                if i == operand_idx:
                    continue
                ref op = working[i]
                for j in range(len(op)):
                    var lbl = op[j]
                    if not _label_in_cost(future, lbl):
                        future.append(lbl)

            var deduped = _dedupe_labels_for_cost(labels)
            var survived = List[Int]()
            for i in range(len(deduped)):
                var lbl = deduped[i]
                if _label_in_cost(future, lbl):
                    survived.append(lbl)

            var step_flops = _tensor_size(deduped, label_sizes)
            var out_size = _tensor_size(survived, label_sizes)
            total_flops += step_flops
            if out_size > peak_intermediate:
                peak_intermediate = out_size
            working[operand_idx] = survived^

            var step_d = Python.dict()
            step_d["lhs"] = PythonObject(operand_idx)
            step_d["rhs"] = PythonObject(-1)
            step_d["flops"] = PythonObject(step_flops)
            step_d["out_size"] = PythonObject(out_size)
            steps_py.append(step_d)
        elif len(raw_step) == 2:
            var lhs_idx = raw_step[0]
            var rhs_idx = raw_step[1]
            if lhs_idx == rhs_idx:
                raise Error(String("path_cost: pairwise step ", path_idx, " uses the same operand twice"))
            if lhs_idx < 0 or lhs_idx >= len(working) or rhs_idx < 0 or rhs_idx >= len(working):
                raise Error(String("path_cost: pairwise step ", path_idx, " index out of range"))

            var others = List[List[Int]]()
            for i in range(len(working)):
                if i != lhs_idx and i != rhs_idx:
                    others.append(working[i].copy())

            var out_labels = _step_output_labels(working[lhs_idx], working[rhs_idx], others, eq.output)
            var step_flops = _flop_cost(working[lhs_idx], working[rhs_idx], out_labels, label_sizes)
            var out_size = _tensor_size(out_labels, label_sizes)
            total_flops += step_flops
            if out_size > peak_intermediate:
                peak_intermediate = out_size

            var step_d = Python.dict()
            step_d["lhs"] = PythonObject(lhs_idx)
            step_d["rhs"] = PythonObject(rhs_idx)
            step_d["flops"] = PythonObject(step_flops)
            step_d["out_size"] = PythonObject(out_size)
            steps_py.append(step_d)

            var next_working = List[List[Int]]()
            for i in range(len(working)):
                if i != lhs_idx and i != rhs_idx:
                    next_working.append(working[i].copy())
            next_working.append(out_labels^)
            working = next_working^
        else:
            raise Error(
                String(
                    "path_cost: step ",
                    path_idx,
                    " has arity ",
                    len(raw_step),
                    ", expected 1 or 2",
                )
            )

    var rec = Python.dict()
    rec["total_flops"] = PythonObject(total_flops)
    rec["peak_intermediate"] = PythonObject(peak_intermediate)
    rec["steps"] = steps_py
    return rec
