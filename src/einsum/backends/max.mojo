"""Mojo-side MAX backend over TileTensor pairwise contractions.

The public Python `backend="max[:cpu|gpu]"` path still owns MAX Graph execution
in `python/moeinsum/_max_backend.py`. This module is the Mojo backend seam: it
consumes the repo's `ContractionPlan`, packs pairwise steps into BMM-shaped
TileTensor buffers, and lowers the contraction itself through
`linalg.bmm.batched_matmul`.

The current ABI is still flat `UnsafePointer[Float64]` plus runtime
shape/stride lists, so this implementation materializes TTGT-style packed
intermediates before the TileTensor call. That is the honest cutover point:
runtime-strided operands enter as flat buffers, pairwise math runs through MAX's
TileTensor BMM kernel, and a later zero-copy RuntimeLayout path can remove the
packing when the ABI grows real TileTensor operands.
"""

from std.collections import List
from std.memory import UnsafePointer
from std.memory.unsafe_pointer import alloc

from layout import Coord, Idx, TileTensor, row_major
from linalg.bmm import batched_matmul

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


def _append_labels_for_axes(
    mut out: List[Int],
    labels: List[Int],
    axes: List[Int],
) raises -> None:
    for i in range(len(axes)):
        var axis = axes[i]
        if axis < 0 or axis >= len(labels):
            raise Error(String("execute_max: axis ", axis, " out of range"))
        out.append(labels[axis])


def _labels_for_axes(labels: List[Int], axes: List[Int]) raises -> List[Int]:
    var out = List[Int]()
    _append_labels_for_axes(out, labels, axes)
    return out^


def _concat3(lhs: List[Int], mid: List[Int], rhs: List[Int]) -> List[Int]:
    var out = List[Int]()
    for i in range(len(lhs)):
        out.append(lhs[i])
    for i in range(len(mid)):
        out.append(mid[i])
    for i in range(len(rhs)):
        out.append(rhs[i])
    return out^


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
                    "execute_max: repeated label ",
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
            "execute_max: size mismatch on label ",
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
        raise Error(String("execute_max: label ", lbl, " has no source dim"))
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


def _shape_from_self(
    labels: List[Int],
    source_labels: List[Int],
    source_shape: List[Int],
) raises -> List[Int]:
    var out = List[Int]()
    for i in range(len(labels)):
        var axis = _single_axis_for_label(source_labels, labels[i], String("source"))
        if axis < 0:
            raise Error(String("execute_max: source label ", labels[i], " missing"))
        out.append(source_shape[axis])
    return out^


def _coord_in_group(
    lbl: Int,
    labels: List[Int],
    idx: List[Int],
) -> Int:
    var pos = _index_of(labels, lbl)
    if pos >= 0:
        return idx[pos]
    return -1


def _coord_for_grouped_label(
    lbl: Int,
    group0_labels: List[Int],
    group0_idx: List[Int],
    group1_labels: List[Int],
    group1_idx: List[Int],
    group2_labels: List[Int],
    group2_idx: List[Int],
) raises -> Int:
    var coord = _coord_in_group(lbl, group0_labels, group0_idx)
    if coord >= 0:
        return coord
    coord = _coord_in_group(lbl, group1_labels, group1_idx)
    if coord >= 0:
        return coord
    coord = _coord_in_group(lbl, group2_labels, group2_idx)
    if coord >= 0:
        return coord
    raise Error(String("execute_max: no coordinate for label ", lbl))


def _grouped_operand_offset(
    operand_labels: List[Int],
    operand_shape: List[Int],
    operand_strides: List[Int],
    group0_labels: List[Int],
    group0_idx: List[Int],
    group1_labels: List[Int],
    group1_idx: List[Int],
    group2_labels: List[Int],
    group2_idx: List[Int],
) raises -> Int:
    var off: Int = 0
    for axis in range(len(operand_labels)):
        if operand_shape[axis] == 1:
            continue
        var coord = _coord_for_grouped_label(
            operand_labels[axis],
            group0_labels,
            group0_idx,
            group1_labels,
            group1_idx,
            group2_labels,
            group2_idx,
        )
        off += coord * operand_strides[axis]
    return off


def _pack_lhs_bmk(
    src: _WorkTensor,
    src_labels: List[Int],
    batch_labels: List[Int],
    batch_shape: List[Int],
    m_labels: List[Int],
    m_shape: List[Int],
    k_labels: List[Int],
    k_shape: List[Int],
    dst: UnsafePointer[Float64, MutAnyOrigin],
    m_size: Int,
    k_size: Int,
) raises -> None:
    var batch_strides = _row_major_strides(batch_shape)
    var m_strides = _row_major_strides(m_shape)
    var k_strides = _row_major_strides(k_shape)
    var batch_idx = _zeros(len(batch_shape))
    while True:
        var batch_flat = _offset(batch_idx, batch_strides)
        var m_idx = _zeros(len(m_shape))
        while True:
            var m_flat = _offset(m_idx, m_strides)
            var k_idx = _zeros(len(k_shape))
            while True:
                var k_flat = _offset(k_idx, k_strides)
                var src_off = _grouped_operand_offset(
                    src_labels,
                    src.shape,
                    src.strides,
                    batch_labels,
                    batch_idx,
                    m_labels,
                    m_idx,
                    k_labels,
                    k_idx,
                )
                dst[(batch_flat * m_size + m_flat) * k_size + k_flat] = src.data[src_off]
                if not _advance_index(k_idx, k_shape):
                    break
            if not _advance_index(m_idx, m_shape):
                break
        if not _advance_index(batch_idx, batch_shape):
            break


def _pack_rhs_bkn(
    src: _WorkTensor,
    src_labels: List[Int],
    batch_labels: List[Int],
    batch_shape: List[Int],
    k_labels: List[Int],
    k_shape: List[Int],
    n_labels: List[Int],
    n_shape: List[Int],
    dst: UnsafePointer[Float64, MutAnyOrigin],
    k_size: Int,
    n_size: Int,
) raises -> None:
    var batch_strides = _row_major_strides(batch_shape)
    var k_strides = _row_major_strides(k_shape)
    var n_strides = _row_major_strides(n_shape)
    var batch_idx = _zeros(len(batch_shape))
    while True:
        var batch_flat = _offset(batch_idx, batch_strides)
        var k_idx = _zeros(len(k_shape))
        while True:
            var k_flat = _offset(k_idx, k_strides)
            var n_idx = _zeros(len(n_shape))
            while True:
                var n_flat = _offset(n_idx, n_strides)
                var src_off = _grouped_operand_offset(
                    src_labels,
                    src.shape,
                    src.strides,
                    batch_labels,
                    batch_idx,
                    k_labels,
                    k_idx,
                    n_labels,
                    n_idx,
                )
                dst[(batch_flat * k_size + k_flat) * n_size + n_flat] = src.data[src_off]
                if not _advance_index(n_idx, n_shape):
                    break
            if not _advance_index(k_idx, k_shape):
                break
        if not _advance_index(batch_idx, batch_shape):
            break


def _execute_pairwise[
    target: StaticString = "cpu"
](
    step: PairwiseStep,
    lhs: _WorkTensor,
    rhs: _WorkTensor,
    mut allocated: List[UnsafePointer[Float64, MutAnyOrigin]],
) raises -> _WorkTensor:
    var batch_labels = _labels_for_axes(step.lhs_labels, step.batch_axes_lhs)
    var m_labels = _labels_for_axes(step.lhs_labels, step.free_axes_lhs)
    var k_labels = _labels_for_axes(step.lhs_labels, step.contract_axes_lhs)
    var n_labels = _labels_for_axes(step.rhs_labels, step.free_axes_rhs)

    var batch_shape = _shape_for_labels(
        batch_labels,
        step.lhs_labels,
        lhs.shape,
        step.rhs_labels,
        rhs.shape,
    )
    var m_shape = _shape_from_self(m_labels, step.lhs_labels, lhs.shape)
    var k_shape = _shape_for_labels(
        k_labels,
        step.lhs_labels,
        lhs.shape,
        step.rhs_labels,
        rhs.shape,
    )
    var n_shape = _shape_from_self(n_labels, step.rhs_labels, rhs.shape)

    var batch_size = _numel(batch_shape)
    var m_size = _numel(m_shape)
    var k_size = _numel(k_shape)
    var n_size = _numel(n_shape)

    var natural_labels = _concat3(batch_labels, m_labels, n_labels)
    var natural_shape = _concat3(batch_shape, m_shape, n_shape)
    var natural_strides = _row_major_strides(natural_shape)

    var out_n = batch_size * m_size * n_size
    var out_alloc_n = out_n if out_n > 0 else 1
    var out_data = alloc[Float64](out_alloc_n)
    _zero_buffer(out_data, out_alloc_n)
    allocated.append(out_data)

    if out_n == 0 or k_size == 0:
        return _transpose_to_labels(
            _WorkTensor(out_data, natural_shape^, natural_strides^, natural_labels^),
            step.out_labels,
        )

    var lhs_n = batch_size * m_size * k_size
    var rhs_n = batch_size * k_size * n_size
    var lhs_pack = alloc[Float64](lhs_n)
    var rhs_pack = alloc[Float64](rhs_n)
    allocated.append(lhs_pack)
    allocated.append(rhs_pack)

    _pack_lhs_bmk(
        lhs,
        step.lhs_labels,
        batch_labels,
        batch_shape,
        m_labels,
        m_shape,
        k_labels,
        k_shape,
        lhs_pack,
        m_size,
        k_size,
    )
    _pack_rhs_bkn(
        rhs,
        step.rhs_labels,
        batch_labels,
        batch_shape,
        k_labels,
        k_shape,
        n_labels,
        n_shape,
        rhs_pack,
        k_size,
        n_size,
    )

    var a_layout = row_major(Coord(Idx(batch_size), Idx(m_size), Idx(k_size)))
    var b_layout = row_major(Coord(Idx(batch_size), Idx(k_size), Idx(n_size)))
    var c_layout = row_major(Coord(Idx(batch_size), Idx(m_size), Idx(n_size)))
    var a_tile = TileTensor(lhs_pack, a_layout)
    var b_tile = TileTensor(rhs_pack, b_layout)
    var c_tile = TileTensor(out_data, c_layout)
    batched_matmul[transpose_a=False, transpose_b=False, target=target](c_tile, a_tile, b_tile)

    return _transpose_to_labels(
        _WorkTensor(out_data, natural_shape^, natural_strides^, natural_labels^),
        step.out_labels,
    )


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
                "execute_max: unary out_permutation rank ",
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
                    "execute_max: output label ",
                    dst_labels[i],
                    " not present in result labels",
                )
            )
        perm.append(p)
    return perm^


def _transpose_to_labels(
    tensor: _WorkTensor,
    labels: List[Int],
) raises -> _WorkTensor:
    if _labels_equal(tensor.labels, labels):
        return tensor.copy()
    var perm = _permutation_for_labels(tensor.labels, labels)
    var transposed = transpose_view(tensor.shape, tensor.strides, perm)
    return _WorkTensor(
        tensor.data,
        transposed.shape.copy(),
        transposed.strides.copy(),
        labels.copy(),
    )


def _copy_pair_survivors(
    working: List[_WorkTensor],
    mut next_working: List[_WorkTensor],
    lhs_idx: Int,
    rhs_idx: Int,
) raises -> None:
    if lhs_idx == rhs_idx:
        raise Error(String("execute_max: pairwise step uses the same operand twice"))
    if lhs_idx < 0 or lhs_idx >= len(working) or rhs_idx < 0 or rhs_idx >= len(working):
        raise Error(String("execute_max: pairwise working-set index out of range"))
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
                "execute_max: result rank ",
                len(tensor.shape),
                " != output rank ",
                len(out_shape),
            )
        )
    for axis in range(len(out_shape)):
        if tensor.shape[axis] != out_shape[axis]:
            raise Error(
                String(
                    "execute_max: result dim ",
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


def execute_max[
    target: StaticString = "cpu"
](
    plan: ContractionPlan,
    operand_data: List[UnsafePointer[Float64, MutAnyOrigin]],
    operand_shapes: List[List[Int]],
    operand_strides: List[List[Int]],
    out_ptr: UnsafePointer[Float64, MutAnyOrigin],
    out_shape: List[Int],
    out_strides: List[Int],
) raises:
    """Execute `plan` with TileTensor-backed MAX pairwise contraction steps.

    Same working-set semantics as `ContractionPlan`: pairwise steps consume two
    operands and append one result; unary steps replace their operand in place.
    Pairwise math lowers through `linalg.bmm.batched_matmul` after packing the
    current flat-buffer ABI into BMM-shaped TileTensor buffers. `target`
    selects the MAX kernel target at compile time, matching `linalg.bmm`.
    """
    if len(operand_data) != len(operand_shapes) or len(operand_data) != len(operand_strides):
        raise Error(String("execute_max: operand data/shape/stride length mismatch"))
    if plan.n_input_operands != len(operand_data):
        raise Error(
            String(
                "execute_max: plan expects ",
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
                raise Error(String("execute_max: pairwise step index out of range at step ", step_idx))
            var lhs = working[ps.lhs_idx].copy()
            var rhs = working[ps.rhs_idx].copy()
            var out = _execute_pairwise[target=target](ps, lhs, rhs, allocated)
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
                raise Error(String("execute_max: unary step index out of range at step ", step_idx))
            var out = _execute_unary(us, working[us.operand_idx].copy(), allocated)
            working[us.operand_idx] = out^

    if len(working) != 1:
        raise Error(
            String(
                "execute_max: contraction path leaves ",
                len(working),
                " tensors; expected 1",
            )
        )

    var result = _transpose_to_labels(working[0], plan.final_labels)
    _copy_to_output(result, out_ptr, out_shape, out_strides)

    for i in range(len(allocated)):
        allocated[i].free()
