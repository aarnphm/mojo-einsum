"""Python-object FFI helpers for the native MAX TileTensor backend."""

from std.collections import List
from std.memory import UnsafePointer
from std.python import PythonObject, Python

from einsum.backends.max import execute_max
from einsum.parse import parse, expand_ellipsis
from einsum.plan import build_plan_from_path


def _pylist_ints_to_mojo(values_obj: PythonObject) raises -> List[Int]:
    var out = List[Int]()
    var n = Int(len(values_obj))
    for i in range(n):
        out.append(Int(py=values_obj[i]))
    return out^


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


def _numel(shape: List[Int]) -> Int:
    var n: Int = 1
    for i in range(len(shape)):
        n *= shape[i]
    return n


def _pylist_ptrs_to_mojo[
    dtype: DType,
](ptrs_obj: PythonObject, expected: Int,) raises -> List[UnsafePointer[Scalar[dtype], MutAnyOrigin]]:
    var actual = Int(len(ptrs_obj))
    if actual != expected:
        raise Error(
            String(
                "einsum_max: expected ",
                expected,
                " operand pointers but got ",
                actual,
            )
        )
    var out = List[UnsafePointer[Scalar[dtype], MutAnyOrigin]]()
    for i in range(expected):
        var ptr = UnsafePointer[Scalar[dtype], MutExternalOrigin](unsafe_from_address=Int(py=ptrs_obj[i]))
        out.append(ptr.as_any_origin())
    return out^


def _ptr_from_py[
    dtype: DType,
](addr_obj: PythonObject) raises -> UnsafePointer[Scalar[dtype], MutAnyOrigin]:
    var ptr = UnsafePointer[Scalar[dtype], MutExternalOrigin](unsafe_from_address=Int(py=addr_obj))
    return ptr.as_any_origin()


def _validate_max_ptr_payload(
    operand_shapes: List[List[Int]],
    operand_strides: List[List[Int]],
    operand_numels_obj: PythonObject,
    out_shape: List[Int],
    out_strides: List[Int],
    expected_operands: Int,
) raises -> None:
    if len(operand_shapes) != expected_operands:
        raise Error(
            String(
                "einsum_max: equation has ",
                expected_operands,
                " operands but got ",
                len(operand_shapes),
                " shape records",
            )
        )
    if len(operand_strides) != expected_operands:
        raise Error(
            String(
                "einsum_max: equation has ",
                expected_operands,
                " operands but got ",
                len(operand_strides),
                " stride records",
            )
        )
    if Int(len(operand_numels_obj)) != expected_operands:
        raise Error(
            String(
                "einsum_max: equation has ",
                expected_operands,
                " operands but got ",
                Int(len(operand_numels_obj)),
                " numel records",
            )
        )

    for op_idx in range(expected_operands):
        if len(operand_shapes[op_idx]) != len(operand_strides[op_idx]):
            raise Error(
                String(
                    "einsum_max: operand ",
                    op_idx,
                    " shape rank ",
                    len(operand_shapes[op_idx]),
                    " != stride rank ",
                    len(operand_strides[op_idx]),
                )
            )
        var expected_elems = _numel(operand_shapes[op_idx])
        var actual_elems = Int(py=operand_numels_obj[op_idx])
        if expected_elems != actual_elems:
            raise Error(
                String(
                    "einsum_max: operand ",
                    op_idx,
                    " buffer numel ",
                    actual_elems,
                    " != shape product ",
                    expected_elems,
                )
            )

    if len(out_shape) != len(out_strides):
        raise Error(
            String(
                "einsum_max: output shape rank ",
                len(out_shape),
                " != output stride rank ",
                len(out_strides),
            )
        )


def execute_max_ptr_payload[
    dtype: DType,
    target: StaticString,
](eq_obj: PythonObject, payload_obj: PythonObject, path_obj: PythonObject,) raises -> PythonObject:
    var eq_str = String(py=eq_obj)
    var eq = parse(eq_str)

    var operand_shapes = _pylist_shapes_to_mojo(payload_obj["operand_shapes"])
    var operand_strides = _pylist_shapes_to_mojo(payload_obj["operand_strides"])
    var out_shape = _pylist_ints_to_mojo(payload_obj["out_shape"])
    var out_strides = _pylist_ints_to_mojo(payload_obj["out_strides"])

    var operand_ranks = List[Int]()
    for i in range(len(operand_shapes)):
        operand_ranks.append(len(operand_shapes[i]))
    expand_ellipsis(eq, operand_ranks)

    _validate_max_ptr_payload(
        operand_shapes,
        operand_strides,
        payload_obj["operand_numels"],
        out_shape,
        out_strides,
        eq.n_operands(),
    )

    var path = _py_path_to_mojo(path_obj)
    var plan = build_plan_from_path(eq, path)
    var data_ptrs = _pylist_ptrs_to_mojo[dtype](
        payload_obj["operand_ptrs"],
        eq.n_operands(),
    )
    var out_ptr = _ptr_from_py[dtype](payload_obj["out_ptr"])

    execute_max[dtype=dtype, target=target](
        plan,
        data_ptrs,
        operand_shapes,
        operand_strides,
        out_ptr,
        out_shape,
        out_strides,
    )

    return Python.none()
