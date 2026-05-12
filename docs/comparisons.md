---
title: Comparison
date: 2026/05/10
---

## Feature parity

| Feature                      | moeinsum v0.1          | NumPy                | PyTorch             | JAX                 | cuTENSOR      | TBLIS          |
| ---------------------------- | ---------------------- | -------------------- | ------------------- | ------------------- | ------------- | -------------- |
| Equation grammar             | done                   | done                 | done                | done                | done (C API)  | done (C API)   |
| Implicit output              | done                   | done                 | done                | done                | n/a           | n/a            |
| Ellipsis broadcasting        | done                   | done                 | done                | done                | n/a           | n/a            |
| Size-1 per-label broadcast   | done                   | done                 | done                | done                | n/a           | n/a            |
| Trace / diagonal             | done                   | done                 | done                | done                | n/a           | partial        |
| Multi-operand paths          | done                   | needs `optimize=`    | auto (opt_einsum)   | auto (opt_einsum)   | single-step   | single-step    |
| `greedy`                     | done                   | done                 | via opt_einsum      | via opt_einsum      | n/a           | n/a            |
| `optimal` DP                 | done (n <= 16)         | done (n ~<= 10)      | via opt_einsum      | via opt_einsum      | n/a           | n/a            |
| `random-greedy`              | done (fixed-trial)     | out of scope         | via opt_einsum      | via opt_einsum      | n/a           | n/a            |
| `branch` (`all` / `2` / `1`) | done                   | out of scope         | via opt_einsum      | via opt_einsum      | n/a           | n/a            |
| Hypergraph paths             | out of scope           | out of scope         | external (cotengra) | external (cotengra) | n/a           | n/a            |
| Compile-time paths           | planned                | out of scope         | out of scope        | partial (jit-trace) | partial (JIT) | out of scope   |
| Per-call-site kernel cache   | done (P7)              | out of scope         | out of scope        | done (jit cache)    | done (plan)   | out of scope   |
| GETT fused permute           | post-v0.1 perf work    | out of scope         | out of scope        | out of scope        | done          | done           |
| CPU tensor-core (AMX)        | done (via MAX)         | partial (Accelerate) | partial             | partial             | n/a           | partial        |
| GPU tensor-core (WGMMA)      | done (via MAX)         | n/a                  | done                | done                | done          | n/a            |
| Configurable accumulator     | fp32/fp64 on MAX       | partial              | partial             | done                | done          | n/a            |
| Deterministic reduction      | reference-only         | partial              | partial             | partial             | out of scope  | n/a            |
| NumPy interop                | done                   | n/a                  | via conversion      | via conversion      | via cupy      | via conversion |
| PyTorch interop              | done (DLPack, P8)      | n/a                  | n/a                 | via DLPack          | via cupy      | n/a            |
| JAX interop                  | done (DLPack, P8)      | n/a                  | via DLPack          | n/a                 | n/a           | n/a            |
| MLX interop                  | done (DLPack, P8)      | n/a                  | n/a                 | n/a                 | n/a           | n/a            |

`done` = shipped; `planned` = not shipped; `out of scope` = not planned for v0.1. "n/a" = comparison undefined.

## Performance

| Workload                                    | moeinsum vs. peer                                                                                                               |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| Matmul-shaped (`ij,jk->ik`, `bij,bjk->bik`) | Executable through MAX Graph today; committed cross-platform ratios still pending                                               |
| Multi-operand chains (`ij,jk,kl,lm->im`)    | Matches JAX + opt_einsum path recurrence; each pairwise step lowers through the same MAX Graph bridge where supported           |
| Irregular permutes                          | TTGT today on `max` backend, matching PyTorch/JAX; cuTENSOR GETT wins ~1.5-3x. `native` P11/P12 closes this.                    |
| Tensor networks (n > 20)                    | Use cotengra for hypergraph path search; opt_einsum greedy is not enough for these cases                                        |
| Call-site overhead (hot cache)              | P7 plan cache removes repeated parse / planning cost; benchmark CLI emits ratios against numpy / opt_einsum / jax / torch / mlx |

## Gaps

- **No cotengra equivalent.** Tensor-network workloads compute the path externally. opt_einsum's family covers the current ML-shaped target cases; quantum-circuit simulation needs cotengra.
- **No optimized GETT yet.** Kernel work remains post-v0.1. Awkward permutes go through flat-buffer/native or TTGT-style MAX lowering until then.
- **MAX diagonal lowering is explicit.** The executable MAX Graph path lowers repeated labels through `gather_nd`, so trace / diagonal cases now run there. It is a materializing graph op today, while the Mojo native path can still become a stride-only view in the GETT cutover.
- **Low precision is backend-limited.** bf16 is covered on the MAX Graph path with fp32 accumulation and bf16 output. fp16, fp8 (e4m3, e5m2), and opcode-level accumulator control wait on dtype-specialized Mojo TileTensor/native kernels.
- **No autograd.** PyTorch and JAX wrap einsum with autograd; moeinsum is a primitive.

## Project seams

- **Backend-pluggable dispatch.** `reference`, `max`, and `native` share equation and plan IR. PyTorch/JAX keep that seam internal.
- **Compile-time paths.** The optimizer is in Mojo and already shares the same recurrence as the runtime API. Shape-specialized straight-line GEMM overloads are still future work.
- **JIT plan cache.** Per-(equation, dtype-sig, rank-sig, backend) cache. Repeated calls reuse the parsed/planned path.

## Design sources

| From       | Source                                                   | Why                                                           |
| ---------- | -------------------------------------------------------- | ------------------------------------------------------------- |
| opt_einsum | Path algorithms (greedy, optimal, random-greedy, branch) | Established contraction-path algorithms                       |
| PyTorch    | `sumproduct_pair` B/K/M/N classification                 | Right factorization; JAX uses the same one                    |
| JAX        | lhs/rhs-swap output-permutation trick                    | Avoids a final transpose for some contractions                |
| cuTENSOR   | GETT algorithm; plan-cache pattern                       | Kernel and caching strategy                                   |
| TBLIS      | BLIS-packing-aware tensor contraction                    | The CPU implementation of GETT                                |
| MLX        | Lazy evaluation for whole-graph fusion                   | Future MAX Graph polish should let MAX fuse beyond one einsum |
| cotengra   | Hypergraph paths                                         | Needed for n > 30 tensor-network cases                        |

## Rejected designs

- **Tensor Comprehensions' polyhedral search.** Schedule-search compiler work is out of scope; the current target is specialized kernels plus simple dispatch.
- **Halide-style schedule language.** Extra developer surface without a clear einsum win.
- **A graph IR.** `ContractionPlan` is a list of B/K/M/N-classified pairwise steps plus permutations. Graph-level fusion and cross-op layout belong in MAX rather than the core IR.
