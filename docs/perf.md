# Performance tuning

This page is the user-facing companion to `derivations.md`. The math there explains _why_ certain choices matter; here are the _what_ and _when_: which backend to pick, which `optimize=` setting, which accumulator dtype, and how to read profiler output when something is slower than you expect.

## When to choose which backend

mojo-einsum ships four backends; the default `max_kernels` is right for almost everything. The exceptions are worth knowing.

**`reference`** — naive nested loop, fp64 internally. Use it for:

- Correctness debugging. If a result differs from numpy, run with `backend="reference"` first; that result is bit-equivalent to numpy at fp64 by construction.
- Tiny inputs (≤ 100 elements). The reference loop has no overhead; the BMM-lowered path has constant launch cost.
- Education / single-stepping. The reference walks every label index explicitly, so it's trivial to read.

Never use `reference` for anything bigger than ~10000 total elements. It scales O(product-of-all-label-sizes).

**`max_kernels`** — lowers each pairwise step to `linalg.batched_matmul`. Use it for:

- Everything you'd normally use `numpy.einsum`, `torch.einsum`, or `jax.numpy.einsum` for. ML-shaped contractions, batched matmuls, multi-step chains under ~10 operands.
- CPU (Apple Silicon, AVX-512) — the matmul dispatcher handles platform selection. You get vDSP/AMX on Mac, BLIS-style packed AVX-512 on Intel/AMD, NEON on ARM.
- GPU (SM90, SM100, AMD CDNA) — with `target="gpu"` and a valid `DeviceContext`, the same dispatcher routes to tensor-core kernels. WGMMA on Hopper, MFMA on AMD.

The matmul dispatch is where almost all the engineering effort has gone — that's the win of going through MAX rather than rolling our own. Default to this.

**`native`** — our own GETT-style kernels. Use when:

- The contraction has a heavy permute that `max_kernels` would materialize. Pattern: any contraction where the contracted dims aren't naturally adjacent and they account for ≥ 30% of total work. Profile to confirm.
- You need fp16 / bf16 / fp8 micro-control. The `native` path lets the accumulator-precision choice flow into the WGMMA/MFMA opcode selection in a way that `linalg.batched_matmul`'s parameter doesn't expose granularly.

In v0.1 this backend is a stub; full GETT lands in Phase 11 / 12.

**`max_graph`** — builds a MAX graph from the contraction plan and hands it to the MAX compiler. Use when:

- You have multiple einsums in sequence with elementwise ops between them. MAX's whole-graph fusion can collapse all of that into one kernel — the BMM-lowered path can't.
- Latency cost of MAX graph construction is amortized over many calls (training, inference loop, repeated benchmark iterations).

In v0.1 this backend is a stub; full integration lands in Phase 14.

## Which `optimize=` algorithm

This is the path-optimizer choice, separate from the backend. opt_einsum's algorithm family ships in `path.mojo`:

| Algorithm | When to use                                                            | Cost                              |
| --------- | ---------------------------------------------------------------------- | --------------------------------- |
| `naive`   | Operand order is hand-tuned; you know better than the planner.         | 0                                 |
| `greedy`  | Default for n > 4 operands. Near-optimal for ML-shaped contractions.   | O(n³) planning                    |
| `optimal` | Default for n ≤ 4 (this is what `auto` picks). Truly optimal in FLOPs. | O(3ⁿ) planning, tractable to n=16 |
| `auto`    | Automatic threshold dispatch. Default for the public API.              | Per-n table above                 |

For n ≤ 4, the planning cost is negligible — always use `optimal` or `auto`. For n in the 5–16 range, `optimal` adds noticeable overhead (milliseconds to seconds at n=16), but if the einsum will be called repeatedly, the JIT cache (P7) amortizes it to zero across calls.

For n > 16: `greedy` is the only option in v0.1. If you have a real tensor-network workload with n > 30 (quantum simulation, lattice contractions), use `cotengra` directly via Python to compute the path, then pass it explicitly:

```python
import cotengra as ctg
path = ctg.array_contract_path(eq, *shapes, optimize=ctg.HyperOptimizer())
result = mojo_einsum.einsum(eq, *arrays, optimize=path)
```

(The explicit-path API lands with P4 polish; for v0.1 you can call `einsum_path` yourself and pass the result.)

## Accumulator dtype

Default behavior: inputs cast up to `accum_dtype` for the accumulation, result cast back to `result_dtype`. The default `accum_dtype` is the larger of fp32 and the input dtype.

| Input dtype | Default `accum_dtype` | When to override                                                                                          |
| ----------- | --------------------- | --------------------------------------------------------------------------------------------------------- |
| fp32        | fp32                  | Rarely. fp64 if you have a known stability problem.                                                       |
| fp16 / bf16 | fp32                  | **Never override.** fp16/bf16 accumulation is mathematically broken above K=64 (see `derivations.md` §4). |
| fp64        | fp64                  | n/a                                                                                                       |
| int\*       | int64                 | If you can guarantee no overflow, int32 saves bandwidth.                                                  |

The `derivations.md` §4 derivation shows the √K error growth concretely. The headline number: bf16 accumulator at K=4096 has ~50% relative error. fp32 accumulator at K=4096 has ~10⁻⁵ error. The fp32 accumulator costs 2× bandwidth at the input but is a non-negotiable correctness requirement.

## Deterministic reductions

`linalg.batched_matmul` parallelizes the K-loop across threads / blocks; the order of summation is non-deterministic. For most numerical workloads this is fine — the error is at the unit roundoff level. For some (regression testing, audit trails, reproducibility-required research) it matters.

mojo-einsum exposes a `deterministic=True` flag that forces serial K-summation. Costs about 30% throughput on CPU, can be 5–10× slower on GPU (you serialize the BMM). Use only when needed.

## Profiling

Before tuning anything, profile. A `mojo-einsum-bench` CLI ships in P13 and emits per-step timing as JSON:

```bash
mojo-einsum-bench "ij,jk,kl->il" --shapes 256,256,256,256 --backend max_kernels --target cpu
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

The per-step `gflops` tells you whether the matmul kernel is hot. If it's <50% of your platform's peak, the bottleneck is probably permute / packing, not the BMM — switch to `native` (when shipped) or restructure the equation to reduce permute cost.

## Cross-platform notes

**Apple Silicon (M-series)**: AMX coprocessor handles small-batch matmul efficiently through Accelerate's `cblas_*` calls. Best performance comes from fp32 inputs (AMX has native support); fp16 requires a software fp16→fp32 expansion that costs about 2×. bf16 is unsupported on AMX directly — Accelerate falls back to vDSP fp32. For inference workloads, fp32 is usually the right choice on M-series.

**NVIDIA Hopper (H100, H200)**: WGMMA + TMA are the inner kernel. The `linalg.batched_matmul` GPU path uses `warp_specialize_gemm_with_multicasting`. Peak fp16/bf16 throughput requires `M ≥ 64`, `N ≥ 128`, `K` a multiple of 16. Smaller contractions hit the small-matrix slow path. If your einsum lowers to such a shape, consider whether the path optimizer is choosing a suboptimal intermediate — sometimes a different `optimize=` setting produces a better-shaped intermediate.

**NVIDIA Blackwell (B100, B200)**: TCGEN05 + UMMA. Similar story to Hopper but with stricter alignment requirements. mojo-einsum's `max_kernels` backend handles this transparently when `target="gpu"`.

**AMD MI300X**: MFMA tensor cores via Composable Kernel. Peak fp16 requires `M, N` multiples of 32 (vs 64 on Hopper). The matmul dispatcher knows this. For odd shapes, performance suffers more than on NVIDIA — AMD's mixed-shape kernels are less mature.

## When `max_kernels` is slow

Three diagnostic questions:

1. **Is it the BMM or the permute?** Profile per-step. If the gflops number on the BMM step is reasonable (>50% peak) but total time is dominated by something else, the something else is the permute — `native`/GETT will help.
2. **Is the K dim tiny?** BMM kernels assume K ≥ 16 or so for tensor-core throughput. If your equation has K=4 or K=8, you're hitting a slow path. Often the fix is to reshape: combine multiple K-dims into one fatter K, or use the `cublas_compute_32f_fast_16f` (or platform equivalent) variant.
3. **Are you allocating intermediates in the inner loop?** Each pairwise step in the multi-operand case allocates a fresh intermediate buffer. The `ContractionContext` arena (P6) sizes a single buffer to the peak intermediate up front. If you're in a hot loop, manually reusing context across calls (when the API exposes this in P7+) saves the allocation.

## Comparisons

For the long-form comparison table — feature parity, perf in canonical contractions, install / dep complexity — see `comparisons.md`. Headline: `max_kernels` matches PyTorch / JAX / NumPy on ML-shaped contractions, beats them on call-site overhead (via the JIT cache, P7), and loses to cuTENSOR for the moment on irregular shapes that need GETT.
