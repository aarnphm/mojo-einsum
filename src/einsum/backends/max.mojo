"""MAX-kernels backend skeleton.

Consumes a `ContractionPlan` and dispatches each step to MAX's kernel library:
`linalg.batched_matmul` for two-operand BMM-lowered steps, and the unary kernels
from `einsum.unary` for single-operand steps. This backend should inherit CPU,
SM90, SM100, and Apple-Silicon dispatch from MAX, so we do not write
platform-specific kernels at this layer.

The plumbing currently is not wired into the FFI. `src/lib.mojo` still passes
flat-list operands to the reference backend. The Phase 5 work that unblocks this:

  1. FFI accepts numpy ndarray buffers and constructs `TileTensor` views over
     them, using `RuntimeLayout` when shapes are runtime-only.
  2. Each pairwise step builds `Layout` compositions to produce zero-copy
     `(*B, *M, *K)` / `(*B, *K, *N)` views of the operands.
  3. `linalg.batched_matmul[transpose_b=...](c, a, b, ctx)` runs the
     contraction.
  4. The output is permuted via `out_permutation` to `out_labels` order.

For `all_dims_known=False` operands, the common DLPack case, we fall back to
TTGT: materialize the permutation into a fresh buffer before matmul. The native
backend's GETT path avoids this materialization.

This file is intentionally structural: the function shape, dispatch seam, and
lowering notes are correct, but the kernel calls are stubbed until the FFI side
is upgraded. See `docs/derivations.md` Sections 1, 3, and 6 for the BMM
lowering math, GETT notes, and output-permutation choice.
"""

from std.memory import UnsafePointer

from einsum.plan import ContractionPlan


# ---------------------------------------------------------------------
# Backend entry point
# ---------------------------------------------------------------------


def execute_max(
    plan: ContractionPlan,
    operand_data: List[UnsafePointer[Float64, MutAnyOrigin]],
    operand_shapes: List[List[Int]],
    operand_strides: List[List[Int]],
    out_ptr: UnsafePointer[Float64, MutAnyOrigin],
    out_shape: List[Int],
    out_strides: List[Int],
) raises:
    """Execute `plan` against the operands, writing the result into `out_ptr`.

    Working-set semantics matches `build_naive_plan`: each pairwise step consumes
    two operands and appends the result; each unary step replaces its operand in
    place. The final working-set element is written into `out_ptr`.

    v0.1 status: structural stub. MAX-kernel dispatch lands when the FFI accepts
    TileTensor handles.
    """
    raise Error(
        String(
            "execute_max: not yet implemented (Phase 5 work). "
            "Use backend='reference' for now."
        )
    )


# ---------------------------------------------------------------------
# Pairwise-step lowering (Phase 5 implementation target)
# ---------------------------------------------------------------------

# Pseudocode for the eventual pairwise lowering:
#
# def _execute_pairwise(
#     step: PairwiseStep,
#     lhs_tile: TileTensor[...],
#     rhs_tile: TileTensor[...],
#     out_tile: TileTensor[mut=True, ...],
#     ctx: DeviceContextPtr,
# ) raises:
#     # 1. Build (*B, *M, *K) view of lhs via Layout composition over
#     #    step.batch_axes_lhs ++ step.free_axes_lhs ++ step.contract_axes_lhs.
#     #    If the natural memory order already matches, zero-copy.
#     #    Otherwise materialize via TTGT.
#     #
#     # 2. Same for rhs -> (*B, *K, *N).
#     #
#     # 3. Dispatch:
#     #
#     #    linalg.batched_matmul[
#     #        transpose_a=False,
#     #        transpose_b=False,
#     #    ](out_bmm_view, lhs_view, rhs_view, ctx)
#     #
#     # 4. If step.out_permutation is non-identity, transpose out_bmm_view into
#     #    out_tile via copy. Otherwise the BMM result is already in the right
#     #    layout.
#     #
#     # Note: JAX's trick of trying both (lhs, rhs) and (rhs, lhs) orderings to
#     # avoid the output permute is a one-branch optimization here: check whether
#     # the BMM-natural order matches out_labels, and if not, retry with swapped
#     # operands.


# ---------------------------------------------------------------------
# Unary-step lowering (Phase 5)
# ---------------------------------------------------------------------

# Pseudocode:
#
# def _execute_unary(
#     step: UnaryStep,
#     in_tile: TileTensor[...],
#     out_tile: TileTensor[mut=True, ...],
# ) raises:
#     # Compose layout-only transformations where possible, materialize only when
#     # we hit a reduction.
#     if step.kind == UNARY_TRANSPOSE:
#         # Pure layout permutation - write metadata, no copy.
#         transpose_view(...)
#     elif step.kind == UNARY_DIAGONAL:
#         # Stride summation - write metadata, no copy.
#         diagonal_view(..., step.diag_axes, ...)
#     elif step.kind == UNARY_REDUCE_SUM:
#         # Allocate output, call reduce_sum_axes.
#         reduce_sum_axes(in_buf, in_shape, in_strides, step.reduce_axes,
#                         out_buf, out_strides)
#     elif step.kind == UNARY_TRACE:
#         # Compose diagonal_view -> reduce_sum_axes, following NumPy's pattern.
#         var mid_shape = List[Int]()
#         var mid_strides = List[Int]()
#         diagonal_view(in_shape, in_strides, step.diag_axes,
#                       mid_shape, mid_strides)
#         reduce_sum_axes(in_buf, mid_shape, mid_strides, step.reduce_axes,
#                         out_buf, out_strides)
