---
title: Performance tuning
date: 2025/05/10
---

_See [[derivations]] for the math behind these rules._

## backends

moeinsum exposes four backend:
- `reference` is still the default because it covers the full grammar with fp64 internal accumulation.
	- naive nested loop, and this use double precision by default. 
	- Use it as the oracle when a result diverges.
	- For tensor totals under $10^4$ elements, where BMM launch overhead is significant. 
	- Note that it scales as $O(\text{product-of-all-label-sizes}))$, so large contractions will require difference backend.
- `native` implements the native flat-buffer plan executor in Mojo
	- I have plans to implement GETT-style kernels later
	- It covers full grammar with deterministic reductions
	- should be used for contractions with heavy permutes, esp non-adjacent contracted dims accounting for 30% of the work, and for fp16/bf16/fp8 accumulator control at WGMMA/MFMA opcode.
- `max:cpu` is the native Mojo MAX TileTensor path. 
	-  We will builds the `Buffer` objects and pass the pointer addresses plus shape//strides metadata.
	- for pairwise steps, it will pack operands in $(\text{batch}, m, k)$ and $(\text{batch, k, n})$ TileTensor buffers, then uses MAX's `linalg.bmm.batched_matmul` to dispatch it based on MAX.
- `max:gpu` forces accelerator placement and tries the native GPU extension before MAX Graph.
	- We will only use MAX Graph for signatures and lowered it to `gather_nd`, pairwise contract steps to batched matmul-shape ops, and unary reduction via `sum`/`squeeze`

## which `optimize=` algorithm

The path optimizer is orthogonal to the backend. The algorithm family in `path.mojo`:

| Algorithm | When to use                                                             | Cost                                 |
| --------- | ----------------------------------------------------------------------- | ------------------------------------ |
| `naive`   | Operand order is hand-tuned; you know better than the planner.          | 0                                    |
| `greedy`  | Default for n > 4 operands. Near-optimal for ML-shaped contractions.    | $O(n^3)$ planning                    |
| `optimal` | Default for n <= 4 (this is what `auto` picks). Truly optimal in FLOPs. | $O(3^n)$ planning, tractable to n=16 |
| `auto`    | Threshold dispatch over the rows above. Default for the public API.     | Per-n as above                       |

For n <= 4 the planning cost is negligible, so you should always use `optimal` or `auto`. For n in 5-16, `optimal` adds milliseconds to seconds of overhead at n=16, but the JIT cache amortizes it to zero across repeated calls.

For n > 16, use `greedy` or `random-greedy(-N)` instead of exact DP. Real tensor-network workloads with n > 30 (quantum simulation, lattice contractions) should use `cotengra` directly and pass the result back:

```python
import cotengra as ctg
path = ctg.array_contract_path(eq, *shapes, optimize=ctg.HyperOptimizer())
result = moeinsum.einsum(eq, *arrays, optimize=path)
```

The explicit-path API is available via `einsum_path` (i.e. you can call it yourself, edit or persist the path if needed, then pass `optimize=[(i, j), ...]`)

## accumulator dtype

There are a bit of a confusion between `dtype=` and `accum_dtype=`.

`dtype=` controls the public output dtype, whereas `accum_dtype=` controls the internal precision where the backend exposes. 

For example, `reference` accumulates through fp64. Native MAX pointer entries currently cover fp32/fp64 and preserve that dtype through the borrowed Buffer ABI. 

Passing `accum_dtype=` on a MAX backend routes through the MAX Graph path, which casts pairwise matmul inputs and reduction inputs to fp32 or fp64 before the graph op and rejects fp16/bf16 accumulators. 
> [!note]
> 
> CPU MAX Graph does not support bf16 graph inputs, so graph execution compiles bf16 calls as fp32 graphs and returns bf16 arrays at the API boundary.

| Input dtype | Default `accum_dtype` | When to override                                                          |
| ----------- | --------------------- | ------------------------------------------------------------------------- |
| fp32        | fp32                  | Rarely. fp64 if you have a known stability problem.                       |
| fp16 / bf16 | fp32                  | Avoid fp16/bf16 accumulation above K=64 (see [[derivations#4. Low-precision accumulation\|derivation section 4]]). |
| fp64        | fp64                  | n/a                                                                       |
| int\*       | int64                 | If you can guarantee no overflow, int32 saves bandwidth.                  |

[[derivations#4. Low-precision accumulation|Derivation Section 4]] shows the $\sqrt{K}$ growth, where _at K=4096, bf16 accumulation has ~50% relative error against fp32's ~$10^{-5}$, roughly a $50{,}000\times$ error ratio for a $2\times$ input-bandwidth saving._

## deterministic reductions

_You might have a question wrt to why reduction are non-deterministic?_

`reference` is deterministic because it is a serial scalar loop. MAX Graph and the native MAX TileTensor path follow MAX's matmul implementation; on GPU this means parallel reduction order and tensor-core math. The error is at the unit-roundoff / TF32 level for fp32 shapes I tested

> This is fine for benchmark work, but will cause certain determinism issue whilst using at large.

`deterministic=True` is accepted by the public API and honored by `reference`. The MAX/native backends still need a real deterministic lowering before "forcing serial K-summation" outside the reference path.

## profiling

You can also tune via `moeinsum-bench`, and we emit machine-readable JSON:

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

### lowering and IR debugging

_The IR is relatively simple, but if you are interested_

Use `moeinsum-lowering` when the question is "what will each backend do?"

It does not execute the contraction or import `max.graph`, so it should work everywhere (hopefully):

```bash
moeinsum-lowering "ji,jk->ki" --shapes 3,2 3,5 --backend all
```

The JSON includes parser IR, the chosen working-set path, cost estimates
when the equation has no ellipsis, the Mojo plan graph spec, and per-backend
lowering records. 

For `max[:cpu|gpu]`, the record shows the concrete B/K/M/N split, broadcast inserts, BMM shape, whether operands are swapped to avoid a final transpose, the MAX Graph op target, and the Mojo TileTensor backend seam.

You can also debug the JSON during execution via `ir=True`:

```python
mp.einsum("ij,jk->ik", a, b, backend="max:cpu", ir=True)
```

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

> [!note]- internals
> 
> This maps to `MODULAR_DEBUG=ir-output-dir=.max-ir,op-log-level=trace,source-tracebacks`.
Use `--max-device-sync` when a GPU failure reports at the wrong host call
site, and `--max-nan-check` when an accuracy bug needs op-local NaN checks.
These flags intentionally taint timings; use them for diagnosis, then rerun
without them for benchmark numbers.

## some cross-platform notes

- **Apple Silicon (M-series)** - 
	- AMX handles small-batch matmul via Accelerate's `cblas_*`. 
	- AMX has native fp32; and fp16 pays a ~$2\times$ tax for software fp32 expansion; 
	- bf16 falls back to vDSP fp32 (as AMX has no bf16). 
	- Run fp32 inputs on M-series unless you have a specific reason not to.
- **NVIDIA Hopper (H100, H200)**
	- WGMMA + TMA, dispatched via `warp_specialize_gemm_with_multicasting`. 
	- Peak fp16/bf16 needs $M \ge 64$, $N \ge 128$, and `K` a multiple of 16. 
	- Below that you WILL hit the small-matrix slow path. 
		- If you run into this, probably there is something wrong with my implementation lol (please report as a bug thanks)
		- Try a different `optimize=` and inspect the path.
- **NVIDIA Blackwell (B100, B200)**
	- TCGEN05 + UMMA. Same shape rules as Hopper but with stricter alignment.
	- `backend="max:gpu"` exercises this path through MAX Graph today.
- **AMD MI300X**
	- MFMA via Composable Kernel. 
	- Peak fp16 needs `M, N` multiples of 32 (vs 64 on Hopper). 
	- Odd shapes cost more than they do on NVIDIA; CK's mixed-shape kernels are less mature.

## when `max` is slow

Try this:

1. **BMM or permute?** If a profiler shows the matmul near platform throughput but total time still dominated elsewhere, the elsewhere is usually permute/packing/graph overhead. `native`/GETT will help.
2. **Is K tiny?** Tensor-core BMM assumes $K \ge 16$. K=4 or K=8 hits a slow path. Fix upstream by combining multiple K-dims into one fatter K, or use the `cublas_compute_32f_fast_16f` (or platform equivalent) variant.
3. **Allocating intermediates in the hot loop?** Each pairwise step allocates through the MAX Graph execution path today. The Mojo MAX seam also allocates TTGT pack buffers before its `TileTensor` BMM call. A `ContractionContext` arena is still the next allocation fix.

## comparisons

See also [[comparisons]]. But heuristics you can follow as `max` for ML-shaped contractions, and `native` for irregular GETT-heavy shapes.
