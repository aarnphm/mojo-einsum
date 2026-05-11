"""Entrypoint for `moeinsum._native`.

Exposed callables (consumed by the Python wrapper):
  - `parse_equation(eq: str) -> dict`
        Returns the parsed equation as a dict for debugging.
  - `einsum_reference(eq, operands_flat, operand_shapes) -> (flat_out, out_shape)`
        Reference backend: row-major flat lists in, row-major flat list out.
  - `einsum_path(eq, operand_shapes) -> list[tuple[int, ...]]`
        Path chosen by the naive (P1) plan builder.
  - `einsum_compute_path(eq, operand_shapes, algorithm) -> list[tuple[int, int]]`
        Path chosen by the named optimizer.
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
        return module.finalize()
    except e:
        abort(String("failed to create _native module: ", e))


# ─────────────────────────────────────────────────────────────────────
# parse_equation — IR introspection
# ─────────────────────────────────────────────────────────────────────


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


# ─────────────────────────────────────────────────────────────────────
# einsum_reference — flat-list FFI for the reference backend
# ─────────────────────────────────────────────────────────────────────


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

    # Copy each operand into a Mojo-var Float64 buffer (zero-copy
    # DLPack path is P8 work).
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


# ─────────────────────────────────────────────────────────────────────
# einsum_path — pair sequence introspection
# ─────────────────────────────────────────────────────────────────────


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

    `algorithm ∈ {"greedy", "optimal", "auto", "naive"}`. Output is a
    list of `(lhs_idx, rhs_idx)` tuples — pairwise-only, no unary
    singletons (single-operand contractions go through `einsum_path`
    which uses `build_naive_plan`).
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
