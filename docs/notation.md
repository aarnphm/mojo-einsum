---
title: Einsum notation
date: 2025/05/10
---

Take a matrix product $C = A B$. Written elementwise:

$$C_{ik} = \sum_j A_{ij} B_{jk}$$

the $\sum$ symbol here is redundant, because any index that appears twice on the right but never on the left is, by convention, summed over. The equation then becomes:

$$C_{ik} = A_{ij} B_{jk}$$

here is the same operations but in numpy:

```python
np.einsum("ij,jk->ik", A, B)
```

Read left-to-right: operand label sequences separated by commas, an arrow, the output's label sequence.

## Single-operand operations

The convention applies to one operand just as naturally as to two.

**Sum.** `"ij->"` means "take a matrix, contract both `i` and `j` away"—the result is a scalar equal to the sum of all elements:

$$s = \sum_{ij} A_{ij}$$

**Transpose.** `"ij->ji"` is a permutation. In both NumPy and Mojo implementations of this we never copy data and just return a view with permuted strides.

**Diagonal.** `"ii->i"` is the trickiest single-operand case. The repeated `i` on the input means _the same index value is used for both axes_. The output has one `i`, so we keep that varying:

$$d_i = A_{ii}$$

For a $4 \times 4$ matrix this extracts the four diagonal entries. Implementations exploit the stride trick where the diagonal is a 1D view over the same data with stride equal to `row_stride + col_stride` (for a contiguous row-major $n \times n$ matrix that's `n + 1`).

**Trace.** `"ii->"` is "diagonal then sum":

$$t = \sum_i A_{ii}$$

The output is a scalar. Internally we read it as: collapse the diagonal first (the repeated `i` constraint), then sum out the surviving `i`.

**Implicit output.** `"ij"` with no `->` means "infer the output." NumPy's convention is to take every label that appears exactly once across all inputs, sort alphabetically, that's your output. So `"ij"` is the same as `"ij->ij"`, and `"ii"` is the same as `"ii->"` (every label appears more than once, so the output has no labels—a scalar).

## Two-operand operations

We represents inner and outer products, matvec, matmul, batched matmul, arbitrary contractions via two-operand operation

**Inner product.** `"i,i->"`:

$$s = \sum_i x_i y_i$$

**Outer product.** `"i,j->ij"`, result is a matrix of all pairwise products:

$$M_{ij} = x_i y_j$$

The labels `i` and `j` are independent, so no implicit sum.

**Matrix-vector.** `"ij,j->i"`, the result is the contracted axis:

$$y_i = \sum_j A_{ij} x_j$$

**Batched matmul.** `"bij,bjk->bik"`. For the _batch_ axis, the operation is broadcast across `b`. For each `b`, the contraction `ij,jk->ik` is performed independently. The total operation is $B$ independent matrix multiplications.

**Double contraction (Frobenius inner product).** `"ij,ij->"`:

$$s = \sum_{ij} A_{ij} B_{ij}$$

**Trace of a product.** `"ij,ji->"`—`i` and `j` are both contracted:

$$t = \sum_{ij} A_{ij} B_{ji} = \mathrm{tr}(AB)$$

## four sets

For any two-operand contraction `lhs,rhs->out`, every label falls into one of four category:

- **B** (batch): in `lhs`, `rhs`, and `out`. Broadcast across it.
- **K** (contract): in `lhs` and `rhs`, not in `out`. Summed out.
- **M** (free-left): in `lhs` and `out`, not in `rhs`. Survives from the left operand.
- **N** (free-right): in `rhs` and `out`, not in `lhs`. Survives from the right operand.

This is well-known with both JAX/PyTorch implementation.

| Equation       | B   | M   | K    | N   |
| -------------- | --- | --- | ---- | --- |
| `ij,jk->ik`    | —   | i   | j    | k   |
| `bij,bjk->bik` | b   | i   | j    | k   |
| `ij,ij->`      | —   | —   | i, j | —   |
| `ij,ji->`      | —   | —   | i, j | —   |
| `bij,bkj->bik` | b   | i   | j    | k   |

Note `ij,ji->` and `ij,ij->` have the same B/K/M/N table but different stride math—_the second operand is transposed in one but not the other_.

## Multi-operand and the contraction path

Einsum scales to any number of operands. `"ij,jk,kl->il"` contracts three matrices in sequence. There's no built-in associativity rule—the implementation chooses a _path_, a binary tree of pairwise contractions. For three matrices there are two:

- `(ij, jk) -> ik`, then `(ik, kl) -> il`
- `(jk, kl) -> jl`, then `(ij, jl) -> il`

Bellman's matrix-chain example shows why the choice can matter by orders of magnitude. Let $A$ be $100 \times 1$, $B$ be $1 \times 10^5$, $C$ be $10^5 \times 1$ — the final result is a single scalar.

| Path    | First step                      | Intermediate      | Second step                     | Total ops           |
| ------- | ------------------------------- | ----------------- | ------------------------------- | ------------------- |
| $(AB)C$ | $100 \cdot 1 \cdot 10^5 = 10^7$ | $100 \times 10^5$ | $100 \cdot 10^5 \cdot 1 = 10^7$ | $\sim 2 \cdot 10^7$ |
| $A(BC)$ | $1 \cdot 10^5 \cdot 1 = 10^5$   | $1 \times 1$      | $100 \cdot 1 \cdot 1 = 100$     | $\sim 10^5$         |

The ratio is ~200x in FLOPs and $10^7$x in peak intermediate memory. (example is MoE routing)

This is a path optimization problem.

I implemented opt_einsum's family of algorithms (`greedy`, `optimal`, `random-greedy`, `branch`, `auto`) natively. The algorithms themselves are in `derivations.md`.

## Ellipsis: broadcasting across unknown ranks

The `...` token stands for "any number of leading dimensions."

`"...ij,jk->...ik"` is a batched matmul where the batch shape is whatever the inputs supply. For a 4-D `(B1, B2, M, K)` lhs and 2-D `(K, N)` rhs, the ellipsis expands to two labels representing `B1` and `B2`. The rhs has no ellipsis, so it broadcasts: it carries no batch dims and is reused for every `(B1, B2)` position.

In moeinsum's parser, ellipsis is initially represented by a sentinel label (`-1`). A second pass—`expand_ellipsis(eq, operand_ranks)` — substitutes fresh label IDs once we know the operand ranks. This deferred expansion keeps the parser independent of operand shapes.

## Attention as einsum

We can use einsum to express multi-head attention. Given:

- Q of shape `(batch, heads, seq_q, dim_head)` — labels `bhqd`
- K of shape `(batch, heads, seq_k, dim_head)` — labels `bhkd`
- V of shape `(batch, heads, seq_k, dim_v)` — labels `bhkv`

The pre-softmax scores are:

```python
scores = np.einsum("bhqd,bhkd->bhqk", Q, K)
```

Here `b` and `h` are batch dims (B), `d` is the contracted dim (K), `q` and `k` are the free dims (M and N respectively).

After softmax and scaling, the attended values:

```python
out = np.einsum("bhqk,bhkv->bhqv", weights, V)
```

`b` and `h` again batch, `k` contracted, `q` free-left, `v` free-right.

Two einsums and a softmax. The PyTorch source for the same block runs forty lines.

## IR

moeinsum has a small IR to represent path optimization.

When `parse("ij,jk->ik")` runs, it produces an `EinsumEquation`:

```
inputs: [[0, 1], [1, 2]]    # interned labels: i=0, j=1, k=2
output: [0, 2]
n_labels: 3
label_chars: ["i", "j", "k"]
has_explicit_output: True
```

Labels are interned as ints. NumPy and PyTorch use chars and inherit a 52-label limit (a-zA-Z); we pay one int per label and lift the cap. This is a trade off with tensor-networks, but for most ML workload, it should be ok.

The `EinsumEquation` is consumed by the path optimizer (`path.mojo`), which produces a `ContractionPath`—a sequence of pairwise contraction steps. Each step is then turned into a `PlanStep` with the B/K/M/N classification baked in (this is `classify_pair` in `plan.mojo`, which mirrors JAX's classifier).

The `ContractionPlan` is the IR backends consume; they decide how to execute each step.

Once you have the plan, each pairwise step is a batched matmul on appropriate reshapes; each unary step is a sum, diagonal, transpose, or trace. The interesting engineering is to pick a path that doesn't blow the intermediate, then making the reshape free, then folding the permutation into the kernel's tile loader. Those are in `derivations.md` and `perf.md`.
