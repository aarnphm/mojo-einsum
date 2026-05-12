---
title: Derivations
date: 2025/05/10
---

_see also [[notation|Notation]]._

## 1. The BMM lowering

Any two-operand contraction reduces to a batched matmul after at most one permutation per operand. JAX's `einsum`, PyTorch's `sumproduct_pair` (`aten/src/ATen/native/Linear.cpp`), and cuTENSOR all use this shape.

Take a two-operand contraction `lhs,rhs->out` with labels classified into [[notation#four sets]]:

- B - batch (in $\text{lhs} \cap \text{rhs} \cap \text{out}$)
- M - free-left (in $\text{lhs} \cap \text{out}$)
- N - free-right (in $\text{rhs} \cap \text{out}$)
- K - contract (in $\text{lhs} \cap \text{rhs}$)

Pick any orderings $b_1,\ldots,b_{|B|}$, $m_1,\ldots,m_{|M|}$, $n_1,\ldots,n_{|N|}$, $k_1,\ldots,k_{|K|}$. Permute lhs to `(b, m, k)` and rhs to `(b, k, n)`:

$$
L_{(b),(m),(k)} = \text{lhs}_{b, m, k}, \quad R_{(b),(k),(n)} = \text{rhs}_{b, k, n}
$$

$(b)$ flattens the batch dims, $(m)$ flattens all M dims, and so on - each flattened dim is the product of the corresponding axis sizes. The contraction becomes:

$$C_{(b),(m),(n)} = \sum_{(k)} L_{(b),(m),(k)} \, R_{(b),(k),(n)}$$

A batched matmul over $|B|$ batches, each $(|M_{\text{flat}}|, |K_{\text{flat}}|) \times (|K_\text{flat}|, |N_\text{flat}|) \to (|M_\text{flat}|, |N_{\text{flat}}|)$. Reshape `C` to `(b, m, n)` axes and permute to `out_labels` order.

The whole pipeline is: at most one reshape (zero-copy via strides), one permute, one batched GEMM.

### When is the permute free?

A permutation is free _iff_ it can be expressed as a layout change with no data movement.

For row-major contiguous input, the target permutation must coincide with the natural memory order:

- $(M, K) \to (M, K)$ is free.
- $(K, M) \to (M, K)$ needs a physical transpose unless $K=1$ or $M=1$.
- $(B_1, B_2, M, K) \to (B_1 B_2, M, K)$ is free - flattening contiguous dims.
- $(M, B, K) \to (B, M, K)$ needs physical movement.

When the would-be permutation is exactly a 2D transpose of the inner block, the backend dispatches to `linalg.batched_matmul`'s `transpose_a` / `transpose_b` flags - cuBLAS / Apple BLAS handle it as a packing variant rather than a separate kernel. For more general permutations:

1. Materialize the permute into a fresh buffer (TTGT - Transpose-Transpose-GEMM-Transpose).
2. Fuse the permute into the GEMM's tile-loading code (see [[#3. GETT: GEMM-like Tensor-Tensor multiplication|GETT]]).

The executable MAX Graph path and current native MAX CPU pack path do (1) when the permute is non-trivial; `NativeOptimizedBackend` is the planned home for (2).

### The inner BMM kernel

Once we're at $(B, M, K) \times (B, K, N) \to (B, M, N)$, the workhorse is `linalg.batched_matmul`. It dispatches:

- SM90/Hopper $\to$ `warp_specialize_gemm_with_multicasting` with WGMMA + TMA.
- SM100/Blackwell $\to$ analogous TCGEN05-based kernel.
- CPU + Apple Silicon $\to$ `apple_accelerate.mojo` calls vDSP/AMX through Accelerate.
- CPU + AVX-512 $\to$ BLIS-style micro-kernel with packing.

For non-BMM-shaped contractions—small K, small M, weird strides—we pay the indexing math of the BMM kernel without amortizing it over enough FLOPs (tip: we are going to use GETT here, see below.)

## 2. Contraction-path cost models

For multi-operand einsum, pairwise contraction order moves runtime by orders of magnitude. Three algorithms from `opt_einsum`:

### Reduced-size heuristic (greedy)

A pairwise step's cost has two parts: FLOPs and intermediate size. They couple on BMM-shaped contractions - bigger intermediates means more FLOPs - but the coupling is not perfect.

`greedy` uses a single scalar:

$$\text{cost}_{\text{reduced\_size}}(A, B) = |A| + |B| - |A \otimes B|$$

where $|A|$ is the element count of $A$ and $|A \otimes B|$ is the element count of $A \cdot B$. Bigger reduction wins - it prefers steps that shrink the working set most.

Smith et al. 2018 (`opt_einsum`, JOSS 3:753) reports near-optimal paths on ML-shaped contractions ($n \le 10$) at $O(n^2)$ per step instead of the DP's exponential. The classic failure: several large intermediates of comparable size compete with one small intermediate, and the heuristic picks on absolute reduction instead of ratio - a $1000 \to 100$ step beats a $100 \to 10$ step even when the latter unlocks better downstream choices.

### Optimal DP (Bellman-Held-Karp)

For $n \le 16$ the optimal path is tractable via DP over operand subsets:

$$f(S) = \min_{\emptyset \subsetneq T \subsetneq S} \left[ f(T) + f(S \setminus T) + \text{cost}(T, S \setminus T) \right]$$

with $f(\{i\}) = 0$ for singletons. Answer: $f(\{1, \ldots, n\})$. Recover the first split by recording the minimizing $T$ at each subset.

$O(3^n)$ time - each $(T, S \setminus T)$ pair visited once, $3^n / 2$ total pairs. $O(2^n)$ memory. At $n=16$: 43M states, ~5s on a modern CPU - usable for compile-time, impractical past $n=20$.

Pure FLOPs gives compute-optimal paths; pure memory gives memory-optimal paths; `opt_einsum`'s default mixes them via `reduced_size` per step.

We ship FLOPs (`path.mojo`) as the DP cost - compute-optimal - and use `reduced_size` only inside the greedy.

### Why not `reduced_size` as the DP cost too?

You can - `opt_einsum` offers both.

FLOPs directly determine compute time on the BMM-lowered path (cubic in the inner loop), and memory cost is bounded by FLOPs to within a factor of $|K|$. `reduced_size`'s pitch is peak memory as a hard constraint (OOM), and on GPU peak intermediates often dominate runtime through eviction.

For ML-shaped einsums ($\le 8$ operands, each $\le 6$ dims) the two cost models converge. They diverge for tensor-network contractions ($n > 20$) where slicing - accepting more FLOPs to fit memory - matters. We don't ship cotengra-style slicing yet.

### Cardoso et al. 2024

Cardoso et al. 2024 ([arxiv 2405.09644](https://arxiv.org/abs/2405.09644)) shows pure `reduced_size` undervalues steps with high FLOP/memory divergence. Their cost:

$$\text{cost}_{\text{Cardoso}}(A, B) = \alpha \cdot \text{flops}(A, B) + (1 - \alpha) \cdot \text{reduced\_size}(A, B)$$

with $\alpha$ tuned per-problem.

`path.mojo` keeps FLOP and memory cost as separate functions, so the Cardoso mix can be added without changing the parser or plan IR.

### The branch family

`opt_einsum` ships `branch-all`, `branch-2`, `branch-1` - best-first searches over the contraction tree, pruned by current best total. They sit between DP-optimal and greedy on the time/quality curve. `path.mojo` implements the same family.

## 3. GETT: GEMM-like Tensor-Tensor multiplication

The BMM lowering's weakness is the physical permute cost. When contracted dims are not adjacent in memory, you pay a bandwidth-bound transpose with no FLOPs to amortize. For irregular contractions the transpose dominates.

GETT (Springer et al. 2018, [arxiv 1607.00145](https://arxiv.org/abs/1607.00145); same idea Matthews exploited in TBLIS, [arxiv 1607.00291](https://arxiv.org/abs/1607.00291)) fuses the transpose into the GEMM's tile-loading code.

### BLIS micro-kernel structure

BLIS (Van Zee et al.) decomposes a GEMM kernel into three layers:

1. **Partition loops** - outer loops over M, N, K tiles.
2. **Packing** - copy each tile from its source layout into a tight, register-aligned buffer (Apack for A, Bpack for B).
3. **Micro-kernel** - the inner FLOP loop, register-blocked at the hardware tile shape ($6 \times 8$ for AVX-512 fp64, $64 \times 128 \times 16$ for SM90 WGMMA).

In a standard GEMM, packing is a memcpy with stride math.

### The fusion

For a tensor contraction, M / K / N are _flattenings_ of multiple source dims. The packing routine already walks those dims to gather a tile. So instead of a separate transpose pass, the packer indexes the source tensor through the multi-dim M, K, N mappings directly. No intermediate buffer; no separate transpose kernel.

```
// Standard BMM-lowered path:
permuted_A = transpose(A, M_axes ++ K_axes)   // physical permute
permuted_B = transpose(B, K_axes ++ N_axes)   // physical permute
for each M-tile, K-tile, N-tile:
    Apack = pack(permuted_A[M-tile, K-tile])
    Bpack = pack(permuted_B[K-tile, N-tile])
    micro_kernel(Apack, Bpack, C-tile)

// GETT path:
for each M-tile, K-tile, N-tile:
    Apack = pack_tensor(A, M_tile_mapping, K_tile_mapping)
    Bpack = pack_tensor(B, K_tile_mapping, N_tile_mapping)
    micro_kernel(Apack, Bpack, C-tile)
```

`pack_tensor` does what `pack(permuted_...)` does but reads through the original strides directly. No temporary buffer for transposes; the M / K / N flattening lives in cache-hot packing code rather than a separate bandwidth-bound op.

### Empirical wins

Matthews's TBLIS Table III (Haswell, dgemm): TBLIS within 5-10% of theoretical peak on most synthetic contractions, beating TTGT by 1.3-2.5x when permutes were expensive. Springer & Bientinesi report similar numbers for GETT on Skylake.

On GPU, cuTENSOR ships `CUTENSOR_ALGO_GETT` as a dispatch option. Hopper's WGMMA + TMA fit this path: TMA fetches tile-shaped regions with arbitrary multi-dim strides into shared memory, and WGMMA consumes from shared. moeinsum's `NativeOptimizedBackend` (P12) follows this pattern.

### When GETT loses

GETT needs more code per kernel: every contraction shape technically wants its own permute-aware packing routine. cuTENSOR sidesteps this with JIT per-contraction (cuTENSOR 2.0). Without JIT, you ship either a kernel per shape family (CUTLASS's template explosion) or a generic-but-slower packer.

Mojo's compile-time specialization gives each unique B/K/M/N signature its own packing loop when the equation is a `StringLiteral` or JIT-cache key. That avoids NVRTC and a CUTLASS-style template matrix.

## 4. Low-precision accumulation

A bf16 $\to$ bf16 accumulator GEMM has a real numerical-correctness problem that compounds with $K$.

Accumulate $K$ random products $a_k b_k$ where $a_k, b_k \sim \mathcal{N}(0, 1)$. The true sum is $\sum_{k=1}^K a_k b_k$; the accumulated result is $\sum_{k=1}^K \text{fl}(a_k b_k)$ where $\text{fl}$ rounds at the accumulator's precision.

Each $\text{fl}(\cdot)$ introduces a relative error $\epsilon_k$ bounded by the unit roundoff $u$:

- bf16: $u = 2^{-7} \approx 7.8 \times 10^{-3}$ (7 mantissa bits).
- fp16: $u = 2^{-10} \approx 9.8 \times 10^{-4}$.
- fp32: $u = 2^{-23} \approx 1.2 \times 10^{-7}$.

Accumulated error is $\sum_k a_k b_k \epsilon_k$. Under independence and zero-mean,

$$\sigma(\text{err}) \approx u \cdot \sqrt{K} \cdot \sigma(ab).$$

For bf16 at $K=64$: $u\sqrt{64} \approx 7.8 \times 10^{-3} \cdot 8 = 6.2 \times 10^{-2}$ - _6% relative error_. At $K=1024$: 25%. At $K=4096$ (typical transformer dim): 50%. bf16 accumulation is not usable for these reductions.

fp32 accumulation with bf16 inputs: $u\sqrt{K} \approx 1.2 \times 10^{-7} \cdot \sqrt{4096} \approx 7.7 \times 10^{-6}$ at $K=4096$ - fine.

### Implementation rule

For any einsum with $K > 64$ and low-precision inputs, use a higher-precision accumulator. cuBLAS bf16 GEMMs default to `CUBLAS_COMPUTE_32F`; cuTENSOR bf16 contractions default to `CUTENSOR_COMPUTE_32F`. bf16-accumulating bf16 is sane only when $K$ is statically known to be small (e.g. 16 in some attention heads).

moeinsum's API has an `accum_dtype` parameter. The reference backend ignores it and accumulates in fp64; optimized backends use the backend's matmul accumulation policy until the Mojo TileTensor cutover exposes opcode-level control.

### Pairwise vs serial summation

Even at full precision, serial accumulation of $K$ terms has worst-case error $O(K \cdot u)$. Pairwise summation reduces this to $O(\log K \cdot u)$. cuBLAS and modern GEMMs pairwise-sum inside the tile; moeinsum's reference path is serial and fp64, while optimized backends should pairwise-accumulate in the inner loop.

## 5. The diagonal stride trick

Diagonal extraction (`'ii->i'`) and trace (`'ii->'`) are free - view-only, no copy - when implemented carefully.

For a 2D row-major contiguous matrix of shape $(n, n)$ with element size $s$:

$$A_{ij} \text{ at byte offset } s \cdot (i \cdot n + j).$$

The diagonal $A_{ii}$ has byte offset $s \cdot i \cdot (n + 1)$ - a 1D strided view with element stride $(n+1)$, zero copy.

For higher rank - `'iji->ij'` extracts the $i=k$ slice of a 3D tensor - the same logic generalizes. If input has shape $(s_0, s_1, s_2)$ and strides $(\sigma_0, \sigma_1, \sigma_2)$ in elements, and we want the diagonal along axes 0 and 2 (label `i` repeated), the result has:

- shape $(s_0, s_1)$ (with $s_0 = s_2$)
- strides $(\sigma_0 + \sigma_2, \sigma_1)$

In general, for a diagonal across $k$ repeated occurrences of one label at axes $a_1, \ldots, a_k$, the diagonal-axis stride is $\sigma_{a_1} + \sigma_{a_2} + \cdots + \sigma_{a_k}$; other axes unchanged.

moeinsum's `unary.mojo` implements this by summing the strides of repeated labels into one surviving axis. It is zero-copy whenever the input can be represented as a view.

The historical bug (PyTorch issue #21760 et al.) was forgetting non-contiguous inputs. `A[::2, ::2]` is a 2x downsampled view - row stride `2n`, column stride 2, so the diagonal stride should be `2n + 2`, not `n + 1`. The contiguous-only formula gives the wrong answer.

## 6. Output-permutation choice

Once `(*B, *M, *K) x (*B, *K, *N) -> (*B, *M, *N)` is done, we permute `(*B, *M, *N)` to `out_labels`. The planner can also swap lhs and rhs (compute `(*B, *N, *K) x (*B, *K, *M) -> (*B, *N, *M)`) when that natural axis order already matches `out_labels`.

JAX exploits this. `lax_numpy.py:3288-3300`:

```python
# Try both orders of lhs and rhs, in the hope that one of them means we
# don't need an explicit transpose. opt_einsum likes to contract from
# right to left, so we expect (rhs,lhs) to have the best chance of not
# needing a transpose.
names = batch_names_str + remaining_rhs_names + remaining_lhs_names
if names == result_names:
    dimension_numbers = ((rhs_cont, lhs_cont), (rhs_batch, lhs_batch))
    operand = _dot_general(rhs, lhs, dimension_numbers, precision)
else:
    names = batch_names_str + remaining_lhs_names + remaining_rhs_names
    dimension_numbers = ((lhs_cont, rhs_cont), (lhs_batch, rhs_batch))
    operand = _dot_general(lhs, rhs, dimension_numbers, precision)
```

moeinsum's `classify_pair` currently uses lhs-first, so some contractions take a final permute they could avoid. The fix belongs in pair classification or pairwise lowering.
