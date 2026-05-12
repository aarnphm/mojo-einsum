---
title: Einsum notation
date: 2025/05/10
---

_Einsum is a grammar for tensor contractions where the indices carry the operation._

Take a matrix product $C = A B$. Elementwise:

$$C_{ik} = \sum_j A_{ij} B_{jk}$$

The $\sum$ is redundant: any index that appears twice on the right but never on the left is summed over by convention. The equation collapses to:

$$C_{ik} = A_{ij} B_{jk}$$

The same operation in NumPy:

```python
np.einsum("ij,jk->ik", A, B)
```

Read left to right: operand label sequences separated by commas, an arrow, the output's label sequence.

## Single-operand operations

The convention applies to one operand as naturally as to two.

**Sum.** `"ij->"` contracts both `i` and `j` away - the result is a scalar:

$$s = \sum_{ij} A_{ij}$$

For a $3 \times 4$ matrix that's a sum over twelve entries.

**Transpose.** `"ij->ji"` is a permutation. NumPy and moeinsum both return a view with permuted strides; no data moves.

**Diagonal.** `"ii->i"` is the trickiest single-operand case. The repeated `i` means _the same index value is used for both axes_. The output keeps one `i`:

$$d_i = A_{ii}$$

For a $4 \times 4$ matrix this extracts the four diagonal entries. The stride trick: the diagonal is a 1D view over the same buffer with stride `row_stride + col_stride`. For a contiguous row-major $n \times n$ matrix that's `n + 1`.

**Trace.** `"ii->"` is diagonal-then-sum:

$$t = \sum_i A_{ii}$$

Collapse the diagonal first (the repeated-`i` constraint), then sum out the surviving `i`. Result is a scalar.

**Implicit output.** `"ij"` with no `->` means "infer the output." NumPy's rule: every label appearing exactly once across all inputs, sorted alphabetically, is the output. So `"ij"` equals `"ij->ij"`, and `"ii"` equals `"ii->"` (every label appears more than once, so the output has no labels - a scalar).

## Two-operand operations

Two operands cover inner and outer products, matvec, matmul, batched matmul, and arbitrary contractions.

**Inner product.** `"i,i->"`:

$$s = \sum_i x_i y_i$$

**Outer product.** `"i,j->ij"`. Result is a matrix of pairwise products:

$$M_{ij} = x_i y_j$$

`i` and `j` are independent, so no implicit sum.

**Matrix-vector.** `"ij,j->i"`. `j` contracts away; `i` survives:

$$y_i = \sum_j A_{ij} x_j$$

**Batched matmul.** `"bij,bjk->bik"`. The `b` axis broadcasts; for each `b` the contraction `ij,jk->ik` runs independently. Total: $B$ independent matrix multiplications.

**Double contraction (Frobenius inner product).** `"ij,ij->"`:

$$s = \sum_{ij} A_{ij} B_{ij}$$

**Trace of a product.** `"ij,ji->"` - both `i` and `j` contract:

$$t = \sum_{ij} A_{ij} B_{ji} = \mathrm{tr}(AB)$$

## The four label categories

For any two-operand contraction `lhs,rhs->out`, every label falls into one of four categories - the B/K/M/N classification JAX and PyTorch both use internally:

- **B** (batch): in `lhs`, `rhs`, and `out`. Broadcast across it.
- **K** (contract): in `lhs` and `rhs`, not in `out`. Summed out.
- **M** (free-left): in `lhs` and `out`, not in `rhs`. Survives from the left operand.
- **N** (free-right): in `rhs` and `out`, not in `lhs`. Survives from the right operand.

| Equation       | B   | M   | K    | N   |
| -------------- | --- | --- | ---- | --- |
| `ij,jk->ik`    | -   | i   | j    | k   |
| `bij,bjk->bik` | b   | i   | j    | k   |
| `ij,ij->`      | -   | -   | i, j | -   |
| `ij,ji->`      | -   | -   | i, j | -   |
| `bij,bkj->bik` | b   | i   | j    | k   |

`ij,ji->` and `ij,ij->` share a B/K/M/N table but have different stride math - _the second operand is transposed in one but not the other_. The classifier doesn't capture this; the planner has to.

## Multi-operand and the contraction path

Einsum scales to any number of operands. `"ij,jk,kl->il"` contracts three matrices in sequence. There's no built-in associativity rule - the implementation picks a _path_, a binary tree of pairwise contractions. For three matrices there are two:

- `(ij, jk) -> ik`, then `(ik, kl) -> il`
- `(jk, kl) -> jl`, then `(ij, jl) -> il`

Bellman's matrix-chain example shows why the choice matters by orders of magnitude. Let $A$ be $100 \times 1$, $B$ be $1 \times 10^5$, $C$ be $10^5 \times 1$ - the final result is a single scalar.

| Path    | First step                      | Intermediate      | Second step                     | Total ops           |
| ------- | ------------------------------- | ----------------- | ------------------------------- | ------------------- |
| $(AB)C$ | $100 \cdot 1 \cdot 10^5 = 10^7$ | $100 \times 10^5$ | $100 \cdot 10^5 \cdot 1 = 10^7$ | $\sim 2 \cdot 10^7$ |
| $A(BC)$ | $1 \cdot 10^5 \cdot 1 = 10^5$   | $1 \times 1$      | $100 \cdot 1 \cdot 1 = 100$     | $\sim 10^5$         |

200x in FLOPs, $10^7$x in peak intermediate memory.[^moe] Path selection is its own optimization problem; moeinsum implements `opt_einsum`'s family natively (`greedy`, `optimal`, `random-greedy`, `branch`, `auto`). Algorithms in `derivations.md`.

[^moe]: This shape pattern shows up in MoE routing - a wide ephemeral activation between two narrow projections - which is why the chain matters in practice and not just in textbooks.

## Ellipsis: broadcasting across unknown ranks

The `...` token stands for "any number of leading dimensions."

`"...ij,jk->...ik"` is a batched matmul where the batch shape is whatever the inputs supply. For a 4-D `(B1, B2, M, K)` lhs and 2-D `(K, N)` rhs, the ellipsis expands to two labels covering `B1` and `B2`. The rhs has no ellipsis, so it broadcasts: it carries no batch dims and is reused for every `(B1, B2)` position.

moeinsum's parser represents ellipsis with a sentinel label (`-1`) on the first pass. `expand_ellipsis(eq, operand_ranks)` substitutes fresh label IDs once operand ranks are known. Deferred expansion keeps the parser independent of operand shapes.

## Size-1 broadcast

A label can take size 1 in one operand and size $N$ in another. The size-1 axis is replicated $N$ times along that label.

```python
a.shape == (1, 3, 4)  # cij
b.shape == (5, 4, 6)  # cjk
np.einsum("cij,cjk->cik", a, b)  # → (5, 3, 6); `c` resolves to 5
```

The reference backend strides the size-1 axis with stride 0; it contributes nothing to the flat offset. The MAX backend emits `ops.broadcast_to` after permutation, before the matmul reshape.

Two checks the validator enforces:

- The mismatch must involve a literal 1. `(3, 4) ij` vs `(5, 4) ij` raises - $M \neq N$ is a real conflict, not a broadcast.
- Cross-operand only. Repeated labels inside one operand (`ii->` on a non-square matrix) are diagonal extraction over mismatched extents, and numpy rejects it.

## Attention as einsum

Multi-head attention is two einsums and a softmax. Given:

- Q of shape `(batch, heads, seq_q, dim_head)` - labels `bhqd`
- K of shape `(batch, heads, seq_k, dim_head)` - labels `bhkd`
- V of shape `(batch, heads, seq_k, dim_v)` - labels `bhkv`

Pre-softmax scores:

```python
scores = np.einsum("bhqd,bhkd->bhqk", Q, K)
```

`b` and `h` are batch (B), `d` contracts (K), `q` and `k` are free (M and N).

Attended values, after softmax and scaling:

```python
out = np.einsum("bhqk,bhkv->bhqv", weights, V)
```

`b` and `h` batch, `k` contracts, `q` free-left, `v` free-right.

Two einsums and a softmax. The PyTorch source for the same block runs forty lines.

## IR

moeinsum carries a small IR for path optimization.

`parse("ij,jk->ik")` produces an `EinsumEquation`:

```
inputs: [[0, 1], [1, 2]]    # interned labels: i=0, j=1, k=2
output: [0, 2]
n_labels: 3
label_chars: ["i", "j", "k"]
has_explicit_output: True
```

Labels intern as ints. NumPy and PyTorch use chars and inherit a 52-label limit (a-zA-Z); moeinsum pays one int per label and lifts the cap.[^tn-tradeoff]

[^tn-tradeoff]: This trades cache locality (one byte per char) for label-space (one int per label). For ML workloads the cap matters more than the byte; for tensor-network workloads with millions of labels per equation the calculus inverts.

The `EinsumEquation` feeds the path optimizer (`path.mojo`), which produces a `ContractionPath` - a sequence of pairwise contraction steps. Each step becomes a `PlanStep` with B/K/M/N classification baked in (`classify_pair` in `plan.mojo` mirrors JAX's classifier).

The `ContractionPlan` is the IR backends consume. Backends decide how to execute each step: pairwise steps lower to batched matmul on appropriate reshapes; unary steps lower to sum, diagonal, transpose, or trace.

The engineering work is: pick a path that doesn't blow the intermediate, make the reshape free, fold the permutation into the kernel's tile loader. Those three live in `derivations.md` and `perf.md`.
