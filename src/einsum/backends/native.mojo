"""Native-optimized backend skeleton.

Same execution seam as `MaxBackend`, different lowering: instead of calling
`linalg.batched_matmul` from MAX, this backend ships native kernels:

  - Phase 11: SIMD-tiled CPU GETT, TBLIS-style fuse-permute-into-pack.
  - Phase 12: SM90 WGMMA GETT, with permutation fused into shared-memory tile
    loads, matching the cuTENSOR GETT family.

The skeleton lives here so backend selection is concrete from day one and
downstream code can thread the `"native"` choice through cache keys, CLIs, and
bench matrices without conditional plumbing.

Kernel surface targets, implemented when the corresponding kernel lands:

  - `_pack_lhs[BM, BK]`: pack `(*B, M, K)` from a strided tile into a contiguous
    buffer, applying the permutation inside the loop.
  - `_pack_rhs[BK, BN]`: same for `(*B, K, N)`.
  - `_compute_microkernel`: SIMD `MR x NR` outer-product loop on CPU,
    `TensorCoreAsync[mma_shape=Index(64, 128, 16)]` on SM90 with
    `warpgroup_fence()` bracketing.

The Phase 11/12 design lives in `docs/derivations.md` Section 3. This file stays
a stub until the kernel work starts.
"""

from std.memory import UnsafePointer

from einsum.plan import ContractionPlan


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

    v0.1 status: structural skeleton only. The CPU GETT lands in P11, the SM90
    GETT in P12. Both raise `Phase 11/12 work` until then.
    """
    _ = plan
    _ = operand_data
    _ = operand_shapes
    _ = operand_strides
    _ = out_ptr
    _ = out_shape
    _ = out_strides
    raise Error(
        String(
            "execute_native: not yet implemented (Phase 11/12 work). ",
            "P11 = SIMD-tiled CPU GETT (TBLIS), P12 = SM90 WGMMA. ",
            "Use backend='reference' for correctness, or backend='max' ",
            "once P5 lands.",
        )
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
