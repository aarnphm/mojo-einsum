---
title: Performance tuning
date: 2025/05/10
---

_See [[derivations]] for why the rules below hold. This page is the rules._

## backends

moeinsum ships four. `max` is the default and the right answer almost always.

**`reference`** — naive nested loop, fp64 internally. Bit-equivalent to numpy by construction, so use it as the oracle when a result diverges. Also the right choice for tensor totals under ~10⁴ elements, where BMM launch overhead dominates the actual work. It scales O(product-of-all-label-sizes); past ~10⁴ it falls off a cliff.

**`max`** — lowers each pairwise step to `linalg.batched_matmul`. This is the drop-in for `numpy.einsum` / `torch.einsum` / `jax.numpy.einsum`. The matmul dispatcher handles platform selection: vDSP/AMX on Apple, BLIS-style packed AVX-512 on Intel/AMD, NEON on ARM, WGMMA on Hopper, MFMA on AMD CDNA. With `target="gpu"` and a valid `DeviceContext` you get tensor-core kernels for free.

**`native`** — GETT-style kernels. Switch to this when the contraction has a heavy permute that `max` would materialize — concretely, contracted dims non-adjacent and accounting for ≥30% of work — or when you need fp16/bf16/fp8 micro-control so the accumulator-precision choice flows into the WGMMA/MFMA opcode.

**`max_graph`** — builds a MAX graph from the contraction plan and hands it to the MAX compiler. Use when you have several einsums in sequence with elementwise ops between them; whole-graph fusion collapses the lot into one megakernel, which beats BMM-lowering when graph-construction latency amortizes over many calls.

## which `optimize=` algorithm

The path optimizer is orthogonal to the backend. The algorithm family in `path.mojo`:

| Algorithm | When to use                                                            | Cost                              |
| --------- | ---------------------------------------------------------------------- | --------------------------------- |
| `naive`   | Operand order is hand-tuned; you know better than the planner.         | 0                                 |
| `greedy`  | Default for n > 4 operands. Near-optimal for ML-shaped contractions.   | O(n³) planning                    |
| `optimal` | Default for n ≤ 4 (this is what `auto` picks). Truly optimal in FLOPs. | O(3ⁿ) planning, tractable to n=16 |
| `auto`    | Threshold dispatch over the rows above. Default for the public API.    | Per-n as above                    |

For n ≤ 4 the planning cost is negligible — always `optimal` or `auto`. For n in 5–16, `optimal` adds milliseconds to seconds of overhead at n=16, but the JIT cache amortizes it to zero across repeated calls.

For n > 16 I only implement `greedy`. Real tensor-network workloads with n > 30 (quantum simulation, lattice contractions) should use `cotengra` directly and pass the result back:

```python
import cotengra as ctg
path = ctg.array_contract_path(eq, *shapes, optimize=ctg.HyperOptimizer())
result = moeinsum.einsum(eq, *arrays, optimize=path)
```

The explicit-path API lands with P4 polish; before that, call `einsum_path` yourself and pass the result.

## accumulator dtype

Inputs cast up to `accum_dtype` for the reduction, result casts back to `result_dtype`. The default `accum_dtype` is `max(fp32, input_dtype)`.

| Input dtype | Default `accum_dtype` | When to override                                                                                          |
| ----------- | --------------------- | --------------------------------------------------------------------------------------------------------- |
| fp32        | fp32                  | Rarely. fp64 if you have a known stability problem.                                                       |
| fp16 / bf16 | fp32                  | **Never override.** fp16/bf16 accumulation is mathematically broken above K=64 (see `derivations.md` §4). |
| fp64        | fp64                  | n/a                                                                                                       |
| int\*       | int64                 | If you can guarantee no overflow, int32 saves bandwidth.                                                  |

[[derivations#4. Low-precision accumulation|Derivation 4]] shows the √K growth concretely. The headline ratio: bf16 accumulator at K=4096 has ~50% relative error against fp32's ~10⁻⁵ — five orders of magnitude apart, for a 2× input-bandwidth saving you do not want. fp32 accumulation is not negotiable.

## deterministic reductions

`linalg.batched_matmul` parallelizes the K-loop across threads and blocks; summation order is non-deterministic. The error is at the unit-roundoff level — fine for almost everything, fatal for regression-testing, audit trails, or anything published as reproducible.

Pass `deterministic=True` to force serial K-summation. Cost: ~30% throughput on CPU; 5–10× slower on GPU, where you're serializing what the BMM was parallelizing. Use only when the auditor asks for it.

## profiling

Before tuning anything, profile. `moeinsum-bench` (P13) emits per-step timing as JSON:

```bash
moeinsum-bench "ij,jk,kl->il" --shapes 256,256,256,256 --backend max --target cpu
```

```json
{
  "equation": "ij,jk,kl->il",
  "path": [
    [0, 1],
    [0, 1]
  ],
  "steps": [
    { "shape": "[256,256]x[256,256]", "ms": 0.42, "gflops": 80.1 },
    { "shape": "[256,256]x[256,256]", "ms": 0.41, "gflops": 81.4 }
  ],
  "total_ms": 0.83
}
```

The per-step `gflops` is the diagnostic. Below 50% of platform peak, the bottleneck is permute/packing rather than the matmul — switch to `native` or restructure the equation to reduce permute cost.

## cross-platform notes

**Apple Silicon (M-series)** — AMX handles small-batch matmul via Accelerate's `cblas_*`. AMX has native fp32; fp16 pays a ~2× tax for software fp32 expansion; bf16 falls back to vDSP fp32 (AMX has no bf16). Run fp32 inputs on M-series unless you have a specific reason not to.

**NVIDIA Hopper (H100, H200)** — WGMMA + TMA, dispatched via `warp_specialize_gemm_with_multicasting`. Peak fp16/bf16 needs `M ≥ 64`, `N ≥ 128`, `K` a multiple of 16. Below that you hit the small-matrix slow path. When you do, the fix is usually upstream: the planner picked an intermediate with a small free dim. Try a different `optimize=` and inspect the path.

**NVIDIA Blackwell (B100, B200)** — TCGEN05 + UMMA. Same shape rules as Hopper but with stricter alignment. `max` with `target="gpu"` handles it transparently.

**AMD MI300X** — MFMA via Composable Kernel. Peak fp16 needs `M, N` multiples of 32 (vs 64 on Hopper). Odd shapes cost more than they do on NVIDIA; CK's mixed-shape kernels are less mature.

## when `max` is slow

Three diagnostics, in order:

1. **BMM or permute?** Per-step gflops > 50% peak but total time dominated by something else means the something else is the permute. `native`/GETT will help.
2. **Is K tiny?** Tensor-core BMM assumes K ≥ 16. K=4 or K=8 hits a slow path. Fix upstream by combining multiple K-dims into one fatter K, or use the `cublas_compute_32f_fast_16f` (or platform equivalent) variant.
3. **Allocating intermediates in the hot loop?** Each pairwise step allocates a fresh buffer. The `ContractionContext` arena (P6) sizes a single buffer to the peak intermediate up front; reusing the context across calls (when P7+ exposes it) eliminates the allocation entirely.

## comparisons

Long-form table — feature parity, perf on canonical contractions, install / dep complexity — lives in `comparisons.md`. Headline: `max` matches PyTorch / JAX / NumPy on ML-shaped contractions, beats them on call-site overhead via the JIT cache (P7), and loses to cuTENSOR on irregular shapes that need GETT.
