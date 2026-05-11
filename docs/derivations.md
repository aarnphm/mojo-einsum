---
title: Derivations
date: 2025/05/10
---

_see also [[notation|Notation]] for more information._


## 1. The BMM lowering

> [!important]
>
> The central claim of practical einsum implementation is that _any two-operand contraction reduces to a batched matrix-matrix multiply, after at most one permutation per operand_.
>
> This is JAX's `einsum` implementation, PyTorch's `sumproduct_pair` does (`aten/src/ATen/native/Linear.cpp`), and what cuTENSOR ultimately lowers to under the hood.

Take a two-operand contraction `lhs,rhs->out` with labels classified into [[notation#four sets]]:

- B—batch (in $\text{lhs} \cap \text{rhs} \cap \text{out}$)
- M—free-left (in $\text{lhs} \cap \text{out}$)
- N—free-right (in $\text{rhs} \cap  \text{out}$)
- K—contract (in $\text{lhs} \cap \text{rhs}$)

Choose any orderings $b_1, \ldots, b_{|B|}$ for the B labels, $m_1, \ldots, m_{|M|}$ for M, $n_1, \ldots, n_{|N|}$ for N, $k_1, \ldots, k_{|K|}$ for K.

Permute lhs to label order `(b, m, k)`, permute rhs to label order `(b, k, n)`:

$$
L_{(b),(m),(k)} = \text{lhs}_{b, m, k}, \quad R_{(b),(k),(n)} = \text{rhs}_{b, k, n}
$$

where $(b)$ is the flattened batch index, $(m)$ flattens all M dims, etc. We notice that each flattened dim is just the product of the corresponding axis sizes. The contraction becomes:

$$C_{(b),(m),(n)} = \sum_{(k)} L_{(b),(m),(k)} \, R_{(b),(k),(n)}$$

which is **exactly a batched matmul** of $|B|$-batched matrices, where each batch element is `(|M_flat|, |K_flat|) × (|K_flat|, |N_flat|) → (|M_flat|, |N_flat|)`. Finally reshape `C` back to the `(b, m, n)` axis-labeled tensor and permute its axes to match `out_labels` order.

There is either a reshape (zero-copy per strides), a permute, or a batched GEMM.

### When is the permute free?

> [!important]
>
> A permutation is free _iff_ it can be expressed as a layout change with no data movement.

For row-major contiguous input, this requires the target permutation to coincide with the natural memory order. In practice:

- $(M, K) \to (M, K)$ is free.
- $(K, M) \to (M, K)$ requires a physical transpose unless $K = 1$ or $M = 1$.
- $(B1, B2, M, K) \to (B1*B2, M, K)$ is free (flattening of contiguous dims).
- $(M, B, K) → (B, M, K)$ requires physical movement.

The backend dispatches to `linalg.batched_matmul`'s `transpose_a` / `transpose_b` flags when the would-be permutation is exactly a 2D transpose of the inner block; this lets cuBLAS / Apple BLAS handle it as a packing variant rather than a separate kernel. For more general permutations, you either:

1. Materialize the permute into a fresh buffer (TTGT—Transpose-Transpose-GEMM-Transpose).
2. Fuse the permute into the GEMM's tile-loading code (see [[#3. GETT GEMM-like Tensor-Tensor multiplication|GETT]]).

The `MaxBackend` does (1) when the permute is non-trivial; `NativeOptimizedBackend` will do (2).

### What about the inner BMM kernel?

Once we're at $(B, M, K) \times (B, K, N) \to (B, M, N)$ shape, the workhorse is `linalg.batched_matmul` from `~/workspace/modular/max/kernels/src/linalg/bmm.mojo`. It dispatches:

- SM90/Hopper $\to$ `warp_specialize_gemm_with_multicasting` with WGMMA + TMA.
- SM100/Blackwell $\to$ analogous TCGEN05-based kernel.
- CPU + Apple Silicon $\to$ `apple_accelerate.mojo` calls vDSP/AMX through the Accelerate framework.
- CPU + AVX-512 $\to$ BLIS-style micro-kernel with packing.

This is the "free win" of the BMM lowering: you inherit the work of every BLAS author since 1979. The price is that for non-BMM-shaped contractions — small K, small M, weird strides — you pay for indexing math the BMM kernel doesn't expect. §3 discusses the workaround.

## 2. Contraction-path cost models

For multi-operand einsum, the order of pairwise contractions matters by orders of magnitude. `path.mojo` implements three algorithms; this section derives their cost models.

### Reduced-size heuristic (greedy)

The cost of one pairwise step is conceptually two things: FLOPs (compute) and intermediate-tensor size (memory + bandwidth). For BMM-shaped contractions these are coupled — bigger intermediates mean more FLOPs — but the coupling is not perfect.

opt_einsum's `greedy` algorithm uses a single scalar cost:

$$\text{cost}_{\text{reduced\_size}}(A, B) = |A| + |B| - |A \otimes B|$$

where $|A|$ is the element count of tensor $A$ and $|A \otimes B|$ is the element count of the result of contracting $A$ and $B$. Bigger reduction = better choice — the heuristic prefers steps that _shrink_ the working-set memory most aggressively.

The Smith & Gray 2018 paper (`opt_einsum`, JOSS 3:753) reports this heuristic finds near-optimal paths on ML-shaped contractions (n ≤ 10) while being O(n²) per step rather than the DP's exponential. The classic failure case is when several large intermediates of comparable size compete with one very small intermediate — the heuristic picks based on absolute reduction, not ratio, and can prefer a 1000→100 step over a 100→10 step even when the latter unlocks much better downstream choices.

### Optimal DP (Bellman-Held-Karp)

For n ≤ 16 the optimal path is tractable via DP over operand subsets:

$$f(S) = \min_{\emptyset \subsetneq T \subsetneq S} \left[ f(T) + f(S \setminus T) + \text{cost}(T, S \setminus T) \right]$$

with base case $f(\{i\}) = 0$ for singletons. The answer is $f(\{1, \ldots, n\})$, and the optimal first split is recovered by recording the minimizing $T$ at each subset.

Complexity: O($3^n$) time (each pair $(T, S \setminus T)$ visited once, total pairs is $3^n / 2$), O($2^n$) memory for the table. At n=16 that's 43M states, ~5 seconds on a modern CPU — usable for compile-time evaluation; impractical at n=20+.

The cost function we plug in matters. Pure FLOPs gives compute-optimal paths; pure memory gives memory-optimal paths; opt_einsum's default mixes them via the reduced_size heuristic applied to each step's cost. moeinsum's `path.mojo` ships FLOPs as the DP cost — fully compute-optimal — and uses reduced_size only inside the greedy.

### Why use reduced_size as the DP cost too?

You can — opt_einsum offers both. The argument for FLOPs in the DP: FLOPs directly determine compute time on the BMM-lowered path (cubical in the inner loop bound), and memory cost is bounded by FLOPs to within a factor of $|K|$ where $K$ is the contracted dim. The argument for reduced_size: peak memory is a hard constraint (OOM), and on GPU peak intermediates often dominate runtime through eviction.

For ML-shaped einsums (≤ 8 operands, each ≤ 6 dims) the two cost models almost always agree. They diverge for tensor-network contractions (n > 20) where slicing — accepting more FLOPs to fit memory — becomes important. moeinsum doesn't ship cotengra-style slicing in v0.1; n > 30 contractions should use `cotengra` directly via Python and pass the explicit path.

### Cardoso et al. 2024

The Cardoso et al. paper _Optimizing Tensor Contraction Paths: A Greedy Algorithm Approach With Improved Cost Functions_ ([arxiv 2405.09644](https://arxiv.org/abs/2405.09644)) shows that pure `reduced_size` undervalues steps with high FLOP/memory divergence. Their proposed cost:

$$\text{cost}_{\text{Cardoso}}(A, B) = \alpha \cdot \text{flops}(A, B) + (1 - \alpha) \cdot \text{reduced\_size}(A, B)$$

with $\alpha$ tuned per-problem. moeinsum's `path.mojo` has FLOP and memory cost as separate functions; wiring Cardoso's mixed cost in is a one-line edit. Future work.

### The branch family

opt_einsum also ships `branch-all`, `branch-2`, `branch-1` — best-first searches over the contraction tree, pruned by current best total cost. These sit between DP optimal and greedy on the time/quality curve. For n=5–7 they're often the sweet spot. moeinsum's v0.1 omits them; the implementation is mechanical and lands when a real workload demands it.

## 3. GETT: GEMM-like Tensor-Tensor multiplication

The BMM lowering's weakness is the cost of physical permutation. When contracted dimensions aren't naturally adjacent in memory, you pay a bandwidth-bound transpose op with no FLOPs to amortize. For irregular contractions this transpose can dominate runtime.

The GETT approach (Springer & Bientinesi 2018, [arxiv 1607.00145](https://arxiv.org/abs/1607.00145); the same idea Devin Matthews exploited in TBLIS, [arxiv 1607.00291](https://arxiv.org/abs/1607.00291)) avoids the transpose by fusing it into the GEMM's tile-loading code.

### The BLIS micro-kernel structure (background)

BLIS (Van Zee et al.) decomposes a GEMM kernel into three layers:

1. **Partition loops** — outer loops over M-tile, N-tile, K-tile.
2. **Packing** — copy each tile from its source layout into a tight, register-aligned buffer (Apack for A, Bpack for B).
3. **Micro-kernel** — the inner FLOP loop, register-blocked at the hardware's tile shape (e.g. 6×8 for AVX-512 fp64, 64×128×16 for SM90 WGMMA).

In a standard GEMM, packing is a "boring" memcpy with stride math. The micro-kernel doesn't care where its tile came from — it just reads from Apack.

### The GETT insight

For a tensor contraction, the M / K / N axes are flattenings of multiple source dims. The packing routine _already_ has to walk those source dims to gather a tile. So: instead of a separate transpose pass, the packing routine indexes the source tensor through the (multi-dim) M, K, N axis mappings directly. No intermediate buffer; no separate transpose kernel. The micro-kernel is unchanged.

In pseudocode:

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

`pack_tensor` does what `pack(permuted_...)` does but reads through the original strides directly. The savings: no temporary buffer for the transposes, and the M / K / N flattening logic is in cache-hot packing code rather than a separate bandwidth-bound op.

### Empirical wins

Matthews's TBLIS Table III (Haswell, dgemm): TBLIS within 5–10% of theoretical peak on most synthetic contractions, beating TTGT by 1.3–2.5× when the permutations would have been expensive. Springer & Bientinesi report similar numbers for GETT on Skylake.

On GPU, cuTENSOR ships `CUTENSOR_ALGO_GETT` as one of its dispatch options. Hopper's WGMMA + TMA make GETT natural — TMA asynchronously fetches tile-shaped regions with arbitrary multi-dim strides directly into shared memory, and WGMMA consumes from shared without caring about the source layout. This is essentially GETT's "fuse the permute into packing" idea, implemented at the hardware level. moeinsum's `NativeOptimizedBackend` (P12) follows this pattern.

### When GETT loses

GETT is more complex code per kernel: every contraction shape technically wants its own permute-aware packing routine. cuTENSOR sidesteps this with JIT compilation per-contraction (cuTENSOR 2.0). Without JIT, you either ship a kernel per shape family (CUTLASS's template explosion) or use a generic-but-slower packer.

Mojo's compile-time specialization is the natural answer here. If the equation is a `StringLiteral` (or a JIT-cache key in our P7 model), each unique B/K/M/N shape signature compiles to its own specialized packing loop. No NVRTC tax, no template explosion. This is the architectural argument for a fresh Mojo implementation over wrapping cuTENSOR.

## 4. Low-precision accumulation: √K rounding error

A bf16 → bf16 accumulator GEMM (or any low-precision accumulator) has a real numerical-correctness problem that compounds with the contracted dimension K. This isn't a quality-of-implementation knob — it's mathematics.

### The derivation

Suppose we accumulate $K$ random products $a_k b_k$ where $a_k, b_k \sim \mathcal{N}(0, 1)$. The true sum is $\sum_{k=1}^K a_k b_k$, and the accumulated result is $\sum_{k=1}^K \text{fl}(a_k b_k)$ where $\text{fl}$ is the rounding operator at the accumulator's precision.

Each $\text{fl}(\cdot)$ introduces a relative error $\epsilon_k$ bounded by the unit roundoff $u$ of the accumulator's format. For bf16, $u = 2^{-7} \approx 7.8 \times 10^{-3}$ (only 7 mantissa bits). For fp16, $u = 2^{-10} \approx 9.8 \times 10^{-4}$. For fp32, $u = 2^{-23} \approx 1.2 \times 10^{-7}$.

The accumulated error is $\sum_k a_k b_k \epsilon_k$. Under independence and zero-mean, the standard deviation of the error scales as:

$$\sigma(\text{err}) \approx u \cdot \sqrt{K} \cdot \sigma(a b)$$

For bf16 accumulation with $K = 64$, that's $u \sqrt{64} \approx 7.8 \times 10^{-3} \cdot 8 = 6.2 \times 10^{-2}$ — _6% relative error_. At $K = 1024$, it's 25%. For $K = 4096$ (a typical transformer dim), it's 50%. **The results are garbage.**

fp32 accumulation with bf16 inputs: $u \sqrt{K} \approx 1.2 \times 10^{-7} \cdot \sqrt{4096} \approx 7.7 \times 10^{-6}$ at $K = 4096$ — perfectly fine.

### Implementation rule

For any einsum with $K > 64$ and low-precision inputs, use a higher-precision accumulator. cuBLAS bf16 GEMMs default to `CUBLAS_COMPUTE_32F` accumulation; cuTENSOR bf16 contractions default to `CUTENSOR_COMPUTE_32F`. The only place bf16-accumulating bf16 is sane is when $K$ is statically known to be small (e.g. 16 in some attention heads).

moeinsum's API has an `accum_dtype` parameter (default fp32 when inputs are fp16 or bf16). The `MaxBackend` forwards this to `linalg.batched_matmul`'s compute-type parameter; the reference backend ignores it (always fp64 internally for v0.1) and is the source of truth for numerical regression testing.

### Pairwise vs serial summation

A second-order effect: even at full precision, serial accumulation of $K$ terms has worst-case error $O(K \cdot u)$. Pairwise summation (recurse-and-sum) reduces this to $O(\log K \cdot u)$ — much better for large $K$. cuBLAS and modern GEMMs do pairwise summation inside the tile; a naive einsum kernel might not. The reference backend in moeinsum is serial-accumulating; the optimized backends (P11+) should pairwise-accumulate inside the inner loop. This isn't a correctness issue for the reference (fp64 has enough mantissa to absorb the worst case at any practical $K$) but is a quality-of-implementation knob worth surfacing.

## 5. The diagonal stride trick

Diagonal extraction (`'ii->i'`) and trace (`'ii->'`) are the canonical examples of operations that can be free — view-only, no copy — when implemented carefully.

For a 2D row-major contiguous matrix of shape $(n, n)$ with element size $s$, the data layout is:

$$A_{ij} \text{ at byte offset } s \cdot (i \cdot n + j)$$

The diagonal $A_{ii}$ has byte offset $s \cdot (i \cdot n + i) = s \cdot i \cdot (n + 1)$. That's a 1D strided view with element stride $(n + 1)$, zero copy.

For higher rank — `'iji->ij'` extracts the i=k slice of a 3D tensor — the same logic generalizes. If the input has shape $(s_0, s_1, s_2)$ and strides $(\sigma_0, \sigma_1, \sigma_2)$ (in elements), and we want the "diagonal" along axes 0 and 2 (label `i` repeated), the result has:

- shape $(s_0, s_1)$ (with `s_0 == s_2` required)
- strides $(\sigma_0 + \sigma_2, \sigma_1)$

In general, for a diagonal across k repeated occurrences of one label at axes $a_1, \ldots, a_k$, the diagonal-axis stride is $\sigma_{a_1} + \sigma_{a_2} + \cdots + \sigma_{a_k}$. The other axes are unchanged.

This generalization is what moeinsum's `unary.mojo` (Phase 3) implements: take the per-axis strides from the input's `Layout`, sum the strides of repeated labels into a new axis, build a new `Layout` with the result. Zero copy in all cases where the input is contiguous-enough to be viewed.

The historical implementation bug here (PyTorch issue #21760 et al.) was forgetting to handle non-contiguous inputs. `A[::2, ::2]` is a 2× downsampled view — its row stride is `2n`, column stride is 2, so the diagonal stride should be `2n + 2`, not `n + 1`. The contiguous-only formula gives the wrong answer.

## 6. Output-permutation choice

Once the contraction `(*B, *M, *K) × (*B, *K, *N) → (*B, *M, *N)` is done, we need to permute `(*B, *M, *N)` to match `out_labels`. There's a choice nobody documents well: we can also swap the role of lhs and rhs (compute `(*B, *N, *K) × (*B, *K, *M) → (*B, *N, *M)`) when that produces a result whose natural axis order already matches `out_labels`.

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

moeinsum's `classify_pair` currently always uses the lhs-first order, which means about half of contractions get a final permute they could have avoided. Improving this is a one-line conditional in the plan builder; future work for the perf phase.

---

These six derivations cover the load-bearing math of moeinsum. The notation primer (`notation.md`) establishes the vocabulary; this document establishes the algorithms; `perf.md` covers the empirics. The plan, the kernels, and the backends are then implementation of these ideas.
