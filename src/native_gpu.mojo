"""Lazy import module for native MAX GPU pointer exports."""

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from einsum.max_ffi import execute_max_ptr_payload


@export
def PyInit__native_gpu() -> PythonObject:
    try:
        var module = PythonModuleBuilder("_native_gpu")
        module.def_function[einsum_max_f32_gpu_ptrs_py]("einsum_max_f32_gpu_ptrs")
        module.def_function[einsum_max_f64_gpu_ptrs_py]("einsum_max_f64_gpu_ptrs")
        return module.finalize()
    except e:
        abort(String("failed to create _native_gpu module: ", e))


def einsum_max_f32_gpu_ptrs_py(
    eq_obj: PythonObject,
    payload_obj: PythonObject,
    path_obj: PythonObject,
) raises -> PythonObject:
    return execute_max_ptr_payload[dtype=DType.float32, target="gpu"](
        eq_obj,
        payload_obj,
        path_obj,
    )


def einsum_max_f64_gpu_ptrs_py(
    eq_obj: PythonObject,
    payload_obj: PythonObject,
    path_obj: PythonObject,
) raises -> PythonObject:
    return execute_max_ptr_payload[dtype=DType.float64, target="gpu"](
        eq_obj,
        payload_obj,
        path_obj,
    )
