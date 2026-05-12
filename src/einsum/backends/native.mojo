"""Native backend over the current flat-buffer ABI.

This module owns the Mojo flat-buffer plan executor. The public
`backend="native"` path calls it directly through `_native`; the Mojo MAX seam
in `backends/max.mojo` owns its separate TileTensor pairwise lowering. The next
native kernel cutover replaces this module's pairwise inner loop with native
kernels:

  - Phase 11: SIMD-tiled CPU GETT, TBLIS-style fuse-permute-into-pack.
  - Phase 12: SM90 WGMMA GETT, with permutation fused into shared-memory tile
    loads, matching the cuTENSOR GETT family.

The flat executor is intentionally conservative: it is the semantic baseline
that the SIMD/GPU kernels must preserve.

Kernel surface targets, implemented when the corresponding kernel lands:

  - `_pack_lhs[BM, BK]`: pack `(*B, M, K)` from a strided tile into a contiguous
    buffer, applying the permutation inside the loop.
  - `_pack_rhs[BK, BN]`: same for `(*B, K, N)`.
  - `_compute_microkernel`: SIMD `MR x NR` outer-product loop on CPU,
    `TensorCoreAsync[mma_shape=Index(64, 128, 16)]` on SM90 with
    `warpgroup_fence()` bracketing.

The Phase 11/12 design lives in `docs/derivations.md` Section 3.
"""

from std.collections import List
from std.memory import UnsafePointer
from std.memory.unsafe_pointer import alloc

from einsum.plan import (
    ContractionPlan,
    PairwiseStep,
    UnaryStep,
)
from einsum.unary import (
    diagonal_view,
    reduce_sum_axes,
    transpose_view,
)


@fieldwise_init
struct _WorkTensor(Copyable, Movable):
    var data: UnsafePointer[Float64, MutAnyOrigin]
    var shape: List[Int]
    var strides: List[Int]
    var labels: List[Int]


def _numel(shape: List[Int]) -> Int:
    var n: Int = 1
    for i in range(len(shape)):
        n *= shape[i]
    return n


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


def _zeros(n: Int) -> List[Int]:
    var out = List[Int]()
    for _ in range(n):
        out.append(0)
    return out^


def _zero_buffer(ptr: UnsafePointer[Float64, MutAnyOrigin], n: Int) -> None:
    for i in range(n):
        ptr[i] = 0.0


def _advance_index(mut idx: List[Int], shape: List[Int]) -> Bool:
    """Advance a mixed-radix index. Returns False after the final point."""
    if len(shape) == 0:
        return False
    for axis in range(len(shape)):
        idx[axis] += 1
        if idx[axis] < shape[axis]:
            return True
        idx[axis] = 0
    return False


def _offset(idx: List[Int], strides: List[Int]) -> Int:
    var off: Int = 0
    for axis in range(len(idx)):
        off += idx[axis] * strides[axis]
    return off


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


def _single_axis_for_label(
    labels: List[Int],
    lbl: Int,
    operand_name: String,
) raises -> Int:
    var found = -1
    for axis in range(len(labels)):
        if labels[axis] != lbl:
            continue
        if found >= 0:
            raise Error(
                String(
                    "execute_plan_flat: repeated label ",
                    lbl,
                    " in ",
                    operand_name,
                    " is a unary diagonal/trace case; build a UnaryStep first",
                )
            )
        found = axis
    return found


def _merge_dim(previous: Int, dim: Int, lbl: Int) raises -> Int:
    if previous == -1 or previous == dim:
        return dim
    if previous == 1:
        return dim
    if dim == 1:
        return previous
    raise Error(
        String(
            "execute_plan_flat: size mismatch on label ",
            lbl,
            ": ",
            previous,
            " vs ",
            dim,
        )
    )


def _resolved_dim(
    lbl: Int,
    lhs_labels: List[Int],
    lhs_shape: List[Int],
    rhs_labels: List[Int],
    rhs_shape: List[Int],
) raises -> Int:
    var dim = -1
    var lhs_axis = _single_axis_for_label(lhs_labels, lbl, String("lhs"))
    if lhs_axis >= 0:
        dim = _merge_dim(dim, lhs_shape[lhs_axis], lbl)
    var rhs_axis = _single_axis_for_label(rhs_labels, lbl, String("rhs"))
    if rhs_axis >= 0:
        dim = _merge_dim(dim, rhs_shape[rhs_axis], lbl)
    if dim < 0:
        raise Error(String("execute_plan_flat: label ", lbl, " has no source dim"))
    return dim


def _shape_for_labels(
    labels: List[Int],
    lhs_labels: List[Int],
    lhs_shape: List[Int],
    rhs_labels: List[Int],
    rhs_shape: List[Int],
) raises -> List[Int]:
    var out = List[Int]()
    for i in range(len(labels)):
        out.append(
            _resolved_dim(
                labels[i],
                lhs_labels,
                lhs_shape,
                rhs_labels,
                rhs_shape,
            )
        )
    return out^


def _reduction_labels(
    lhs_labels: List[Int],
    rhs_labels: List[Int],
    out_labels: List[Int],
) -> List[Int]:
    """Unique labels from lhs/rhs that are not carried into the step output."""
    var out = List[Int]()
    for axis in range(len(lhs_labels)):
        var lbl = lhs_labels[axis]
        if not _label_in(out_labels, lbl) and not _label_in(out, lbl):
            out.append(lbl)
    for axis in range(len(rhs_labels)):
        var lbl = rhs_labels[axis]
        if not _label_in(out_labels, lbl) and not _label_in(out, lbl):
            out.append(lbl)
    return out^


def _coord_for_label(
    lbl: Int,
    out_labels: List[Int],
    out_idx: List[Int],
    reduction_labels: List[Int],
    reduction_idx: List[Int],
) raises -> Int:
    var pos = _index_of(out_labels, lbl)
    if pos >= 0:
        return out_idx[pos]
    pos = _index_of(reduction_labels, lbl)
    if pos >= 0:
        return reduction_idx[pos]
    raise Error(String("execute_plan_flat: no coordinate for label ", lbl))


def _operand_offset(
    operand_labels: List[Int],
    operand_shape: List[Int],
    operand_strides: List[Int],
    out_labels: List[Int],
    out_idx: List[Int],
    reduction_labels: List[Int],
    reduction_idx: List[Int],
) raises -> Int:
    var off: Int = 0
    for axis in range(len(operand_labels)):
        if operand_shape[axis] == 1:
            continue
        var coord = _coord_for_label(
            operand_labels[axis],
            out_labels,
            out_idx,
            reduction_labels,
            reduction_idx,
        )
        off += coord * operand_strides[axis]
    return off


def _execute_pairwise(
    step: PairwiseStep,
    lhs: _WorkTensor,
    rhs: _WorkTensor,
    mut allocated: List[UnsafePointer[Float64, MutAnyOrigin]],
) raises -> _WorkTensor:
    var out_shape = _shape_for_labels(
        step.out_labels,
        step.lhs_labels,
        lhs.shape,
        step.rhs_labels,
        rhs.shape,
    )
    var out_strides = _row_major_strides(out_shape)
    var out_n = _numel(out_shape)
    var out_alloc_n = out_n if out_n > 0 else 1
    var out_data = alloc[Float64](out_alloc_n)
    _zero_buffer(out_data, out_alloc_n)
    allocated.append(out_data)

    var reduction = _reduction_labels(
        step.lhs_labels,
        step.rhs_labels,
        step.out_labels,
    )
    var reduction_shape = _shape_for_labels(
        reduction,
        step.lhs_labels,
        lhs.shape,
        step.rhs_labels,
        rhs.shape,
    )

    if out_n == 0:
        return _WorkTensor(out_data, out_shape^, out_strides^, step.out_labels.copy())

    var out_idx = _zeros(len(out_shape))
    while True:
        var acc: Float64 = 0.0
        var reduction_idx = _zeros(len(reduction_shape))
        while True:
            var lhs_off = _operand_offset(
                step.lhs_labels,
                lhs.shape,
                lhs.strides,
                step.out_labels,
                out_idx,
                reduction,
                reduction_idx,
            )
            var rhs_off = _operand_offset(
                step.rhs_labels,
                rhs.shape,
                rhs.strides,
                step.out_labels,
                out_idx,
                reduction,
                reduction_idx,
            )
            acc += lhs.data[lhs_off] * rhs.data[rhs_off]
            if not _advance_index(reduction_idx, reduction_shape):
                break

        out_data[_offset(out_idx, out_strides)] = acc
        if not _advance_index(out_idx, out_shape):
            break

    return _WorkTensor(out_data, out_shape^, out_strides^, step.out_labels.copy())


def _shape_after_reduce(shape: List[Int], reduce_axes: List[Int]) -> List[Int]:
    var out = List[Int]()
    for axis in range(len(shape)):
        if not _label_in(reduce_axes, axis):
            out.append(shape[axis])
    return out^


def _execute_unary(
    step: UnaryStep,
    tensor: _WorkTensor,
    mut allocated: List[UnsafePointer[Float64, MutAnyOrigin]],
) raises -> _WorkTensor:
    var data = tensor.data
    var shape = tensor.shape.copy()
    var strides = tensor.strides.copy()

    if len(step.diag_axes) > 0:
        var diag = diagonal_view(shape, strides, step.diag_axes)
        shape = diag.shape.copy()
        strides = diag.strides.copy()

    if len(step.reduce_axes) > 0:
        var reduced_shape = _shape_after_reduce(shape, step.reduce_axes)
        var reduced_strides = _row_major_strides(reduced_shape)
        var n = _numel(reduced_shape)
        var alloc_n = n if n > 0 else 1
        var reduced_data = alloc[Float64](alloc_n)
        _zero_buffer(reduced_data, alloc_n)
        allocated.append(reduced_data)
        if n > 0:
            reduce_sum_axes(
                data,
                shape,
                strides,
                step.reduce_axes,
                reduced_data,
                reduced_strides,
            )
        data = reduced_data
        shape = reduced_shape^
        strides = reduced_strides^

    if len(step.out_permutation) != len(shape):
        raise Error(
            String(
                "execute_plan_flat: unary out_permutation rank ",
                len(step.out_permutation),
                " != tensor rank ",
                len(shape),
            )
        )
    var transposed = transpose_view(shape, strides, step.out_permutation)
    return _WorkTensor(
        data,
        transposed.shape.copy(),
        transposed.strides.copy(),
        step.out_labels.copy(),
    )


def _labels_equal(lhs: List[Int], rhs: List[Int]) -> Bool:
    if len(lhs) != len(rhs):
        return False
    for i in range(len(lhs)):
        if lhs[i] != rhs[i]:
            return False
    return True


def _permutation_for_labels(src_labels: List[Int], dst_labels: List[Int]) raises -> List[Int]:
    var perm = List[Int]()
    for i in range(len(dst_labels)):
        var p = _index_of(src_labels, dst_labels[i])
        if p < 0:
            raise Error(
                String(
                    "execute_plan_flat: final output label ",
                    dst_labels[i],
                    " not present in result labels",
                )
            )
        perm.append(p)
    return perm^


def _transpose_to_final_labels(
    tensor: _WorkTensor,
    final_labels: List[Int],
) raises -> _WorkTensor:
    if _labels_equal(tensor.labels, final_labels):
        return tensor.copy()
    var perm = _permutation_for_labels(tensor.labels, final_labels)
    var transposed = transpose_view(tensor.shape, tensor.strides, perm)
    return _WorkTensor(
        tensor.data,
        transposed.shape.copy(),
        transposed.strides.copy(),
        final_labels.copy(),
    )


def _copy_pair_survivors(
    working: List[_WorkTensor],
    mut next_working: List[_WorkTensor],
    lhs_idx: Int,
    rhs_idx: Int,
) raises -> None:
    if lhs_idx == rhs_idx:
        raise Error(String("execute_plan_flat: pairwise step uses the same operand twice"))
    if lhs_idx < 0 or lhs_idx >= len(working) or rhs_idx < 0 or rhs_idx >= len(working):
        raise Error(String("execute_plan_flat: pairwise working-set index out of range"))
    for i in range(len(working)):
        if i != lhs_idx and i != rhs_idx:
            next_working.append(working[i].copy())


def _copy_to_output(
    tensor: _WorkTensor,
    out_ptr: UnsafePointer[Float64, MutAnyOrigin],
    out_shape: List[Int],
    out_strides: List[Int],
) raises -> None:
    if len(tensor.shape) != len(out_shape):
        raise Error(
            String(
                "execute_plan_flat: result rank ",
                len(tensor.shape),
                " != output rank ",
                len(out_shape),
            )
        )
    for axis in range(len(out_shape)):
        if tensor.shape[axis] != out_shape[axis]:
            raise Error(
                String(
                    "execute_plan_flat: result dim ",
                    axis,
                    " = ",
                    tensor.shape[axis],
                    " != output dim ",
                    out_shape[axis],
                )
            )

    var n = _numel(out_shape)
    if n == 0:
        return
    var idx = _zeros(len(out_shape))
    while True:
        out_ptr[_offset(idx, out_strides)] = tensor.data[_offset(idx, tensor.strides)]
        if not _advance_index(idx, out_shape):
            break


def execute_plan_flat(
    plan: ContractionPlan,
    operand_data: List[UnsafePointer[Float64, MutAnyOrigin]],
    operand_shapes: List[List[Int]],
    operand_strides: List[List[Int]],
    out_ptr: UnsafePointer[Float64, MutAnyOrigin],
    out_shape: List[Int],
    out_strides: List[Int],
) raises:
    """Execute `plan` against flat buffers, writing the result into `out_ptr`.

    This follows the same working-set contract as `ContractionPlan`: pairwise
    steps remove two tensors and append one intermediate; unary steps replace
    their input slot with a view or reduced buffer. All reductions run in a
    deterministic left-to-right mixed-radix order.
    """
    if len(operand_data) != len(operand_shapes) or len(operand_data) != len(operand_strides):
        raise Error(String("execute_plan_flat: operand data/shape/stride length mismatch"))
    if plan.n_input_operands != len(operand_data):
        raise Error(
            String(
                "execute_plan_flat: plan expects ",
                plan.n_input_operands,
                " operands but got ",
                len(operand_data),
            )
        )

    var working = List[_WorkTensor]()
    for i in range(len(operand_data)):
        working.append(
            _WorkTensor(
                operand_data[i],
                operand_shapes[i].copy(),
                operand_strides[i].copy(),
                List[Int](),
            )
        )

    var allocated = List[UnsafePointer[Float64, MutAnyOrigin]]()
    for step_idx in range(len(plan.steps)):
        var step = plan.steps[step_idx].copy()
        if step.isa[PairwiseStep]():
            var ps = step.unsafe_get[PairwiseStep]().copy()
            if ps.lhs_idx < 0 or ps.lhs_idx >= len(working) or ps.rhs_idx < 0 or ps.rhs_idx >= len(working):
                raise Error(String("execute_plan_flat: pairwise step index out of range at step ", step_idx))
            var lhs = working[ps.lhs_idx].copy()
            var rhs = working[ps.rhs_idx].copy()
            var out = _execute_pairwise(ps, lhs, rhs, allocated)
            var next_working = List[_WorkTensor]()
            _copy_pair_survivors(
                working,
                next_working,
                ps.lhs_idx,
                ps.rhs_idx,
            )
            next_working.append(out^)
            working = next_working^
        else:
            var us = step.unsafe_get[UnaryStep]().copy()
            if us.operand_idx < 0 or us.operand_idx >= len(working):
                raise Error(String("execute_plan_flat: unary step index out of range at step ", step_idx))
            var out = _execute_unary(us, working[us.operand_idx].copy(), allocated)
            working[us.operand_idx] = out^

    if len(working) != 1:
        raise Error(
            String(
                "execute_plan_flat: contraction path leaves ",
                len(working),
                " tensors; expected 1",
            )
        )

    var result = _transpose_to_final_labels(working[0], plan.final_labels)
    _copy_to_output(result, out_ptr, out_shape, out_strides)

    for i in range(len(allocated)):
        allocated[i].free()


def execute_native(
    plan: ContractionPlan,
    operand_data: List[UnsafePointer[Float64, MutAnyOrigin]],
    operand_shapes: List[List[Int]],
    operand_strides: List[List[Int]],
    out_ptr: UnsafePointer[Float64, MutAnyOrigin],
    out_shape: List[Int],
    out_strides: List[Int],
) raises:
    """Execute `plan` via the native kernel set.

    Same working-set semantics as `build_naive_plan`: each pairwise step consumes
    two operands and appends the result; each unary step replaces its operand in
    place.
    """
    execute_plan_flat(
        plan,
        operand_data,
        operand_shapes,
        operand_strides,
        out_ptr,
        out_shape,
        out_strides,
    )


# ---------------------------------------------------------------------
# CPU GETT - Phase 11 design (no code yet)
# ---------------------------------------------------------------------
#
# TBLIS approach: fuse the permutation into the inner-most pack pass.
# Pseudocode:
#
#   for outer block of (B, M):
#     for outer block of (K):
#       pack_lhs_with_permute(lhs_tile, A_pack, perm_lhs)
#       for outer block of (N):
#         pack_rhs_with_permute(rhs_tile, B_pack, perm_rhs)
#         microkernel_outer_product(A_pack, B_pack, C_pack)
#       unpack_with_permute(C_pack, out_tile, perm_out)
#
# Notable choices:
#   - per-thread A/B-pack buffers, not shared.
#   - multiple-accumulator ILP in the microkernel, 4-8 FMA accumulators on
#     AVX-512 / SVE / NEON.
#   - tile sizes (BM, BN, BK) derived from CPU cache hierarchy via
#     `BLIS_PACK_BLOCKING_PARAMETERS`-style discovery at startup.
#   - permutations applied per-element during pack, one fma per multiply.


# ---------------------------------------------------------------------
# GPU GETT - Phase 12 design (no code yet)
# ---------------------------------------------------------------------
#
# SM90 warp-specialized matmul with TMA tile loads, WGMMA-issuing warpgroups,
# and permutation fused into the shared-memory pack. Reference:
# `~/workspace/modular/max/kernels/src/linalg/matmul/gpu/sm90/matmul.mojo`.
#
# Critical mojo-perf invariants for this kernel:
#   - `warpgroup_fence()` before and after every `TensorCoreAsync.mma` batch.
#   - `cuda.cp.async.bulk.tensor.shared::cluster.global.tile` for TMA loads.
#   - multiple-accumulator ILP in the WGMMA loop, 4-8 D-tiles per warpgroup so
#     the issue/retire pipeline saturates.
