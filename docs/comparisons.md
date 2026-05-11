---
title: Comparison
date: 2026/05/10
---


## Feature parity

| Feature                        | moeinsum (v0.1)            | NumPy                | PyTorch               | JAX                   | cuTENSOR           | TBLIS          |
| ------------------------------ | -------------------------- | -------------------- | --------------------- | --------------------- | ------------------ | -------------- |
| Basic equation grammar         | ✅                         | ✅                   | ✅                    | ✅                    | ✅ (C API)         | ✅ (C API)     |
| Implicit output                | ✅                         | ✅                   | ✅                    | ✅                    | n/a                | n/a            |
| Ellipsis broadcasting          | ✅                         | ✅                   | ✅                    | ✅                    | n/a                | n/a            |
| Trace / diagonal               | ✅                         | ✅                   | ✅                    | ✅                    | n/a                | partial        |
| Multi-operand paths            | ✅                         | needs `optimize=`    | auto (via opt_einsum) | auto (via opt_einsum) | single-step        | single-step    |
| `greedy` algorithm             | ✅                         | ✅ (`einsum_path`)   | ✅ (via opt_einsum)   | ✅ (via opt_einsum)   | n/a                | n/a            |
| `optimal` DP algorithm         | ✅ (n ≤ 16)                | ✅ $n \leq \approx 10$         | ✅ (via opt_einsum)   | ✅ (via opt_einsum)   | n/a                | n/a            |
| `random-greedy`                | ⏳ (P4 polish)             | ❌                   | ✅ (opt_einsum)       | ✅ (opt_einsum)       | n/a                | n/a            |
| `branch` family                | ⏳ (P4 polish)             | ❌                   | ✅ (opt_einsum)       | ✅ (opt_einsum)       | n/a                | n/a            |
| Hypergraph paths (cotengra)    | ❌ (out of v0.1 scope)     | ❌                   | external              | external              | n/a                | n/a            |
| Compile-time-known paths       | ✅ (when shapes are alias) | ❌                   | ❌                    | partial (jit-traced)  | partial (JIT plan) | ❌             |
| Per-call-site kernel cache     | ✅ (P7)                    | ❌                   | ❌                    | ✅ (jit cache)        | ✅ (plan cache)    | ❌             |
| GETT-style fused permute       | ⏳ (P11/P12)               | ❌                   | ❌                    | ❌                    | ✅                 | ✅             |
| Native CPU tensor-core (AMX)   | ✅ (via MAX)               | partial (Accelerate) | partial               | partial               | n/a                | partial        |
| Native GPU tensor-core (WGMMA) | ✅ (via MAX)               | n/a                  | ✅                    | ✅                    | ✅                 | n/a            |
| Configurable accumulator dtype | ✅                         | partial              | partial               | ✅                    | ✅                 | n/a            |
| Deterministic reduction flag   | ✅ (P9)                    | partial              | partial               | partial               | ❌                 | n/a            |
| NumPy interop                  | ✅                         | n/a                  | via conversion        | via conversion        | via cupy           | via conversion |
| PyTorch interop                | ✅ (DLPack, P8)            | n/a                  | n/a                   | via DLPack            | via cupy           | n/a            |
| JAX interop                    | ✅ (DLPack, P8)            | n/a                  | via DLPack            | n/a                   | n/a                | n/a            |
| MLX interop                    | ✅ (DLPack, P8)            | n/a                  | n/a                   | n/a                   | n/a                | n/a            |

Legend: ✅ shipped, ⏳ planned, ❌ not in scope. "n/a" means the comparison doesn't make sense (e.g. NumPy interop with NumPy itself).

## Performance

expectation:

- **Matmul-shaped einsums** (`ij,jk->ik`, `bij,bjk->bik`):
	- Should be within ±5% of PyTorch's BMM, JAX's `dot_general`, and cuBLAS direct calls on equivalent shapes, because we lower directly to MAX's `linalg.batched_matmul`.
	- _call-site overhead_ matters separately; see below.
- **Multi-operand chains** (`ij,jk,kl,lm->im`):
	- Same matmul kernel for each step, plus path-optimizer choice.
	- `optimal` uses the same DP recurrence as opt_einsum.
	- Results should match JAX + opt_einsum on these workloads.
- **Irregular contractions** (heavy permute, awkward strides):
	- moeinsum's `max` backend currently does TTGT and materializes the permute.
	- PyTorch/JAX do the same. cuTENSOR's GETT avoids the materialization and wins by ~1.5-3x on these shapes.
	- The `native` backend's P11/P12 GETT implementation targets parity here.
- **Tensor networks** (n > 20 operands, dense contractions):
	- opt_einsum's greedy is suboptimal.
	- cotengra's hypergraph paths win by orders of magnitude.
- **Call-site overhead** (latency of `einsum("ij,jk->ik", a, b)` over hot cache):
	- PyTorch and JAX both parse the equation, call opt_einsum, classify dims, and dispatch BMM on every call.
	- For small tensors it can dominate the FLOPs.
	- moeinsum's JIT plan cache (P7) hits a hash lookup and dispatches directly to the cached kernel.
		- Expected ~10x reduction in call-site latency for repeated small einsums.

## Gaps

- **No cotengra equivalent.** For tensor-network workloads, you must compute the path externally. This is a deliberate v0.1 scoping decision; opt_einsum's algorithm family covers ≤30 operands well, and that handles all ML use cases. Quantum-circuit simulation and similar genuinely need cotengra.
- **No GETT yet.** Phase 11/12 work. Until then, awkward permutes go through TTGT, with the bandwidth cost that implies.
- **No `random-greedy` or `branch`.** opt_einsum has them; moeinsum's `path.mojo` will get them. Until then, pass an explicit path if one of those algorithms matters.
- **Limited dtypes.** v0.1 ships fp32 / fp64 internally. fp16, bf16, fp8 (e4m3, e5m2), int arrive in P9 with accumulator handling. Until then, callers should pre-cast.
- **No autograd.** PyTorch and JAX wrap their einsum with autograd; moeinsum is a primitive.

## Features

- **Backend-pluggable dispatch.** `reference`, `max`, `native`, and `max_graph` share the same equation and plan IR. PyTorch and JAX have variants of this internally but do not expose the seam to users.
- **Compile-time-known paths.** When operand shapes are compile-time `alias`, the path optimizer runs in `@parameter` evaluation and emits a straight-line sequence of GEMM calls. The v0.1 API still takes runtime equation strings; compile-time overloads arrive when P10 lands.
- **The JIT plan cache.** A per-(equation, dtype-sig, rank-sig, backend) cache makes the runtime API behave like a compile-time-specialized library after the first call. This is the cuTENSOR strategy without the NVRTC tax.

## What I steal from each

This is the moneyball table — given Mojo's unique leverage, where does it pay to mimic vs. innovate.

| From           | Steal                                                                  | Why                                                                           |
| -------------- | ---------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| **opt_einsum** | Path-finding algorithm family (greedy, optimal, random-greedy, branch) | Smith & Gray got this right. Good way for me to learn                        |
| **PyTorch**    | `sumproduct_pair` four-bucket classification                           | The B/K/M/N taxonomy is the right factorization. JAX uses the same algorithm. |
| **JAX**        | The lhs/rhs-swap output-permutation trick                              | Saves a final transpose on ~50% of contractions. One-line code change.        |
| **cuTENSOR**   | GETT algorithm; plan-cache pattern                                     | The right kernel design and the right caching strategy, separately.           |
| **TBLIS**      | BLIS-packing-aware tensor contraction                                  | The CPU implementation of GETT.               |
| **MLX**        | Lazy evaluation for whole-graph fusion                                 | `max_graph` backend P14 — let MAX do the fusion, don't reinvent it.           |
| **cotengra**   | (Future) hypergraph paths                                              | The right algorithm for n > 30. Out of v0.1; potentially Phase 16+.           |

## What I have in mind

**Tensor Comprehensions' polyhedral approach.** Beautiful abstraction, didn't ship to production. The lesson is that schedule-search compilers lose to specialized kernel libraries + simple dispatch — exactly what moeinsum does.

**Halide-style schedule-language separation.** Adds developer surface area without a clear win for einsum specifically. Halide's schedules shine when the algorithm is hard to express; einsum's algorithm is a one-liner, so there's nothing to schedule against.

**A new IR.** moeinsum's `ContractionPlan` is intentionally minimal — a list of B/K/M/N-classified pairwise steps plus permutations. It's not a graph IR. The full graph-level concerns (fusion, layout selection across ops) belong in MAX, which is why `max_graph` is a backend rather than the core.

---

The honest summary: moeinsum is a fresh implementation built to learn from everyone else's choices. The math is the same; the kernels are the same; the algorithm catalog is the same. What's new is Mojo's compile-time leverage — used at the path layer (compile-time paths when shapes are alias) and the kernel layer (one source for CPU and GPU, specialization per dtype/rank signature). v0.1 establishes the architecture and proves correctness; v0.2 ships GETT and closes the irregular-contraction gap to cuTENSOR.
