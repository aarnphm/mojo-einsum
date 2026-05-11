---
title: Comparison
date: 2026/05/10
---

## Feature parity

| Feature                      | moeinsum v0.1     | NumPy                | PyTorch             | JAX                 | cuTENSOR      | TBLIS          |
| ---------------------------- | ----------------- | -------------------- | ------------------- | ------------------- | ------------- | -------------- |
| Equation grammar             | done                | done                   | done                  | done                  | done (C API)    | done (C API)     |
| Implicit output              | done                | done                   | done                  | done                  | n/a           | n/a            |
| Ellipsis broadcasting        | done                | done                   | done                  | done                  | n/a           | n/a            |
| Trace / diagonal             | done                | done                   | done                  | done                  | n/a           | partial        |
| Multi-operand paths          | done                | needs `optimize=`    | auto (opt_einsum)   | auto (opt_einsum)   | single-step   | single-step    |
| `greedy`                     | done                | done                   | via opt_einsum      | via opt_einsum      | n/a           | n/a            |
| `optimal` DP                 | done (n <= 16)      | done (n ~<= 10)          | via opt_einsum      | via opt_einsum      | n/a           | n/a            |
| `random-greedy`              | done (fixed-trial)  | out of scope                   | via opt_einsum      | via opt_einsum      | n/a           | n/a            |
| `branch` (`all` / `2` / `1`) | done                | out of scope                   | via opt_einsum      | via opt_einsum      | n/a           | n/a            |
| Hypergraph paths             | out of scope                | out of scope                   | external (cotengra) | external (cotengra) | n/a           | n/a            |
| Compile-time paths           | done (alias shapes) | out of scope                   | out of scope                  | partial (jit-trace) | partial (JIT) | out of scope             |
| Per-call-site kernel cache   | done (P7)           | out of scope                   | out of scope                  | done (jit cache)      | done (plan)     | out of scope             |
| GETT fused permute           | pending (P11/P12)      | out of scope                   | out of scope                  | out of scope                  | done            | done             |
| CPU tensor-core (AMX)        | done (via MAX)      | partial (Accelerate) | partial             | partial             | n/a           | partial        |
| GPU tensor-core (WGMMA)      | done (via MAX)      | n/a                  | done                  | done                  | done            | n/a            |
| Configurable accumulator     | done                | partial              | partial             | done                  | done            | n/a            |
| Deterministic reduction      | done (P9)           | partial              | partial             | partial             | out of scope            | n/a            |
| NumPy interop                | done                | n/a                  | via conversion      | via conversion      | via cupy      | via conversion |
| PyTorch interop              | done (DLPack, P8)   | n/a                  | n/a                 | via DLPack          | via cupy      | n/a            |
| JAX interop                  | done (DLPack, P8)   | n/a                  | via DLPack          | n/a                 | n/a           | n/a            |
| MLX interop                  | done (DLPack, P8)   | n/a                  | n/a                 | n/a                 | n/a           | n/a            |

`done` = shipped; `pending` = planned; `out of scope` = not planned for v0.1. "n/a" = comparison undefined.

## Performance

| Workload                                    | moeinsum vs. peer                                                                                                                                     |
| ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| Matmul-shaped (`ij,jk->ik`, `bij,bjk->bik`) | $\pm 5\%$ of PyTorch BMM, JAX `dot_general`, cuBLAS - lowers to MAX `linalg.batched_matmul`                                                             |
| Multi-operand chains (`ij,jk,kl,lm->im`)    | Matches JAX + opt_einsum; same DP recurrence, same kernel per step                                                                                    |
| Irregular permutes                          | TTGT today on `max` backend, matching PyTorch/JAX; cuTENSOR GETT wins ~1.5-3x. `native` P11/P12 closes this.                                          |
| Tensor networks (n > 20)                    | opt_einsum greedy is suboptimal; cotengra wins by orders of magnitude                                                                                 |
| Call-site overhead (hot cache)              | ~10x faster than PyTorch/JAX for small repeated einsums - P7 plan cache hashes and dispatches; PyTorch/JAX reparse, reclassify, redispatch every call |

## Gaps

- **No cotengra equivalent.** Tensor-network workloads compute the path externally. v0.1 scoping: opt_einsum's family covers <=30 operands, which handles all ML use cases. Quantum-circuit simulation needs cotengra.
- **No GETT.** P11/P12 work. Awkward permutes go through TTGT until then.
- **No explicit-path parser.** opt_einsum accepts caller-supplied paths; moeinsum exposes named optimizers only.
- **fp32 / fp64 only.** fp16, bf16, fp8 (e4m3, e5m2), int arrive in P9 with accumulator handling. Pre-cast until then.
- **No autograd.** PyTorch and JAX wrap einsum with autograd; moeinsum is a primitive.

## What moeinsum does that others don't

- **Backend-pluggable dispatch.** `reference`, `max`, `native`, `max_graph` share equation and plan IR. PyTorch/JAX keep that seam internal.
- **Compile-time paths.** When shapes are `alias`, the optimizer runs in `@parameter` and emits straight-line GEMM. Runtime equation strings work today; compile-time overloads ship with P10.
- **JIT plan cache.** Per-(equation, dtype-sig, rank-sig, backend) cache. Runtime API behaves like a compile-time-specialized library after the first call - cuTENSOR's strategy without the NVRTC tax.

## What I steal from each

| From       | Steal                                                    | Why                                                               |
| ---------- | -------------------------------------------------------- | ----------------------------------------------------------------- |
| opt_einsum | Path algorithms (greedy, optimal, random-greedy, branch) | Smith & Gray got it right                                         |
| PyTorch    | `sumproduct_pair` B/K/M/N classification                 | Right factorization; JAX uses the same one                        |
| JAX        | lhs/rhs-swap output-permutation trick                    | Saves a final transpose on ~50% of contractions. One-line change. |
| cuTENSOR   | GETT algorithm; plan-cache pattern                       | Right kernel; right caching strategy                              |
| TBLIS      | BLIS-packing-aware tensor contraction                    | The CPU implementation of GETT                                    |
| MLX        | Lazy evaluation for whole-graph fusion                   | `max_graph` P14 - let MAX fuse                                    |
| cotengra   | (future) hypergraph paths                                | Right algorithm for n > 30. Phase 16+.                            |

## What I rejected

- **Tensor Comprehensions' polyhedral search.** Beautiful, never shipped. Schedule-search compilers lose to specialized kernels plus simple dispatch.
- **Halide-style schedule language.** Developer surface area without a clear einsum win - einsum's algorithm is a one-liner, so there's nothing to schedule against.
- **A graph IR.** `ContractionPlan` is a list of B/K/M/N-classified pairwise steps plus permutations. Graph-level fusion and cross-op layout belong in MAX, which is why `max_graph` is a backend, not the core.
