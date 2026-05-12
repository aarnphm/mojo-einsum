---
title: Performance tuning
date: 2025/05/10
---

_See [[derivations]] for the math behind these rules._

## backends

moeinsum exposes five backend names. `reference` is still the default because it covers the full grammar with fp64 internal accumulation. `native` is the Mojo plan executor for the full grammar. `max` is the fast executable MAX Graph path for float32, float64, and bfloat16 tensors.

**`reference`** - naive nested loop, fp64 internally. Use it as the oracle when a result diverges. It is also the right choice for tensor totals under $10^4$ elements, where BMM launch overhead can dominate. It scales as O(product-of-all-label-sizes), so large contractions need another backend.

**`max` / `max:cpu` / `max:gpu`** - builds a MAX Graph for each signature and lowers repeated-label operands with `gather_nd`, pairwise contraction steps to batched matmul-shaped graph ops, and unary reductions with `sum` / `squeeze`. It supports matmul, batched matmul, outer products, multi-operand chains, size-1 broadcast, ellipsis, diagonal / trace, unary transpose, and reduce-sum. `max` chooses GPU when MAX reports an accelerator, otherwise CPU; use `max:gpu` or `max:cpu` to force placement.

**`native`** - Mojo flat-buffer plan executor today, GETT-style kernels next. It covers the full grammar with deterministic reductions, then will become the home for contractions with heavy permutes, especially non-adjacent contracted dims accounting for $\ge 30\%$ of work, and for fp16/bf16/fp8 accumulator control at the WGMMA/MFMA opcode.

## which `optimize=` algorithm

The path optimizer is orthogonal to the backend. The algorithm family in `path.mojo`:

| Algorithm | When to use                                                             | Cost                                 |
| --------- | ----------------------------------------------------------------------- | ------------------------------------ |
| `naive`   | Operand order is hand-tuned; you know better than the planner.          | 0                                    |
| `greedy`  | Default for n > 4 operands. Near-optimal for ML-shaped contractions.    | $O(n^3)$ planning                    |
| `optimal` | Default for n <= 4 (this is what `auto` picks). Truly optimal in FLOPs. | $O(3^n)$ planning, tractable to n=16 |
| `auto`    | Threshold dispatch over the rows above. Default for the public API.     | Per-n as above                       |

For n <= 4 the planning cost is negligible - always `optimal` or `auto`. For n in 5-16, `optimal` adds milliseconds to seconds of overhead at n=16, but the JIT cache amortizes it to zero across repeated calls.

For n > 16, use `greedy` or `random-greedy(-N)` instead of exact DP. Real tensor-network workloads with n > 30 (quantum simulation, lattice contractions) should use `cotengra` directly and pass the result back:

```python
import cotengra as ctg
path = ctg.array_contract_path(eq, *shapes, optimize=ctg.HyperOptimizer())
result = moeinsum.einsum(eq, *arrays, optimize=path)
```

The explicit-path API is available now: call `einsum_path` yourself, edit or persist the path if needed, then pass `optimize=[(i, j), ...]`.

## accumulator dtype

`dtype=` controls the public output dtype. `accum_dtype=` is validated at the API boundary, but the current execution story is backend-specific: `reference` accumulates through fp64; MAX Graph uses MAX's matmul accumulation policy (bf16 routes through MAX and accumulates in fp32 internally). The future Mojo `native` / TileTensor cutover is where `accum_dtype` becomes an opcode-level knob.

| Input dtype | Default `accum_dtype` | When to override                                                          |
| ----------- | --------------------- | ------------------------------------------------------------------------- |
| fp32        | fp32                  | Rarely. fp64 if you have a known stability problem.                       |
| fp16 / bf16 | fp32                  | Avoid fp16/bf16 accumulation above K=64 (see `derivations.md` Section 4). |
| fp64        | fp64                  | n/a                                                                       |
| int\*       | int64                 | If you can guarantee no overflow, int32 saves bandwidth.                  |

[[derivations#4. Low-precision accumulation|Derivation 4]] shows the $\sqrt{K}$ growth. At K=4096, bf16 accumulation has ~50% relative error against fp32's ~$10^{-5}$, roughly a $50{,}000\times$ error ratio for a $2\times$ input-bandwidth saving.

## deterministic reductions

`reference` is deterministic because it is a serial scalar loop. The executable MAX Graph path follows MAX's matmul implementation; on GPU this means parallel reduction order and tensor-core math. The error is at the unit-roundoff / TF32 level for fp32 shapes we tested - fine for benchmark work, not a bitwise audit trail.

`deterministic=True` is accepted by the public API and honored by `reference`. The MAX/native backends still need a real deterministic lowering before that flag means "force serial K-summation" outside the reference path.

## profiling

Before tuning anything, profile. `moeinsum-bench` (P13) emits machine-readable JSON:

```bash
moeinsum-bench "ij,jk,kl->il" --shapes 256,256 256,256 256,256 --backend max:cpu --compare-engines numpy,opt_einsum,jax
```

```json
{
  "equation": "ij,jk,kl->il",
  "total_ms_median": 0.83,
  "total_ms_min": 0.81,
  "total_ms_max": 0.87,
  "comparisons": {
    "moeinsum": { "status": "ok", "ms_median": 0.83 },
    "numpy": { "status": "ok", "time_ratio_vs_moeinsum": 0.42 },
    "jax": { "status": "ok", "time_ratio_vs_moeinsum": 0.31 }
  },
  "comparison_fastest": "jax"
}
```

Use `--progress` when running long sweeps interactively; tqdm writes to stderr, while JSON stays on stdout for scripts.

### lowering and IR debugging

Use `moeinsum-lowering` when the question is "what will each backend do?"
It does not execute the contraction or import `max.graph`, so it works on a
local laptop even when MAX is only available on the B200 host:

```bash
moeinsum-lowering "ji,jk->ki" --shapes 3,2 3,5 --backend all
```

The JSON includes parser IR, the chosen working-set path, cost estimates
when the equation has no ellipsis, the Mojo plan graph spec, and per-backend
lowering records. For `max[:cpu|gpu]`, the record shows the concrete B/K/M/N
split, broadcast inserts, BMM shape, whether operands are swapped to avoid a
final transpose, and the MAX Graph op target.

When compiling through MAX, `moeinsum-bench` can enable MAX debug options
before the runtime loads:

```bash
moeinsum-bench "ij,jk->ik" \
  --shapes 1024,1024 1024,1024 \
  --backend max:gpu \
  --max-ir-output-dir .max-ir \
  --max-op-log-level trace \
  --max-source-tracebacks
```

That maps to `MODULAR_DEBUG=ir-output-dir=.max-ir,op-log-level=trace,source-tracebacks`.
Use `--max-device-sync` when a GPU failure reports at the wrong host call
site, and `--max-nan-check` when an accuracy bug needs op-local NaN checks.
These flags intentionally taint timings; use them for diagnosis, then rerun
without them for benchmark numbers.

## cross-platform notes

**Apple Silicon (M-series)** - AMX handles small-batch matmul via Accelerate's `cblas_*`. AMX has native fp32; fp16 pays a ~$2\times$ tax for software fp32 expansion; bf16 falls back to vDSP fp32 (AMX has no bf16). Run fp32 inputs on M-series unless you have a specific reason not to.

**NVIDIA Hopper (H100, H200)** - WGMMA + TMA, dispatched via `warp_specialize_gemm_with_multicasting`. Peak fp16/bf16 needs $M \ge 64$, $N \ge 128$, and `K` a multiple of 16. Below that you hit the small-matrix slow path. When you do, the fix is usually upstream: the planner picked an intermediate with a small free dim. Try a different `optimize=` and inspect the path.

**NVIDIA Blackwell (B100, B200)** - TCGEN05 + UMMA. Same shape rules as Hopper but with stricter alignment. `backend="max:gpu"` exercises this path through MAX Graph today.

**AMD MI300X** - MFMA via Composable Kernel. Peak fp16 needs `M, N` multiples of 32 (vs 64 on Hopper). Odd shapes cost more than they do on NVIDIA; CK's mixed-shape kernels are less mature.

## when `max` is slow

Three diagnostics, in order:

1. **BMM or permute?** If a profiler shows the matmul near platform throughput but total time still dominated elsewhere, the elsewhere is usually permute / packing / graph overhead. `native`/GETT will help.
2. **Is K tiny?** Tensor-core BMM assumes $K \ge 16$. K=4 or K=8 hits a slow path. Fix upstream by combining multiple K-dims into one fatter K, or use the `cublas_compute_32f_fast_16f` (or platform equivalent) variant.
3. **Allocating intermediates in the hot loop?** Each pairwise step allocates through the MAX Graph execution path today. The `ContractionContext` arena design still belongs to the Mojo TileTensor/native cutover.

## comparisons

Feature parity and comparison notes live in `comparisons.md`. Use `max` for ML-shaped contractions today; `native` is the planned path for irregular GETT-heavy shapes.
