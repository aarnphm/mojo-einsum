# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions track the phase ladder in `progress.md`.

## [Unreleased] - v0.1 in progress

### Public API

- `einsum(eq, *operands, backend, optimize, accum_dtype, dtype, return_type, deterministic)` - DLPack-first input, framework-native return (torch in -> torch out), `deterministic=True` as the bit-equality contract.
- `einsum_path(eq, *shapes, optimize)` - every named optimizer plus caller-supplied paths `[(i, j), ...]`.
- `parse_equation(eq)` - IR introspection.
- `path_cost(eq, shapes, path)` - FLOP + peak-intermediate accounting.
- `moeinsum-bench` console script - single-equation timing + `--sweep-optimizers` ratios table.

### Shipped

- **P0** Scaffolding. mohaus + uv layout, dependency declarations, dir structure.
- **P1** Parser + plan IR + reference backend + flat-list FFI + numpy bridge. Mojo 1.0 syntax.
- **P2** Parser: basic, implicit output, ellipsis, trace, diagonal, multi-char labels (int interning).
- **P3** Unary kernels: layout-only transpose + diagonal views, tiled scalar `reduce_sum`.
- **P4** Path optimizers: greedy, optimal-DP, `random-greedy(-N)`, `branch-{all,2,1}`, `auto` (opt_einsum threshold table), explicit caller-supplied paths. All wired through FFI.
- **P6** Multi-operand orchestration: working-set semantics in `build_naive_plan` verified on 4/5-chain tests. Peak-sized `ContractionContext` arena deferred to P5.
- **P7** Plan cache: Python-side LRU keyed by `(eq, shape_sig, dtype, backend, optimize, accum_dtype, target)`. Stores immutable snapshots - caller mutation doesn't pollute hits.
- **P8** DLPack interop: `_interop.to_numpy` preserves dtype; `_interop.from_numpy(arr, kind)` round-trips to torch / jax / mlx / cupy / tensorflow. fp64 cast lives at the FFI boundary, not the API.
- **P9\*** Precision parameters wired (`accum_dtype`, `dtype`, `deterministic`). Real low-precision accumulation lands with `MaxBackend`; the reference backend always accumulates in fp64.
- **P13** Bench CLI: JSON output, per-platform metadata, median/min/max timing, optional path introspection, optimizer-sweep with ratios.
- **P15** Docs (5 files): `notation.md`, `derivations.md` (BMM lowering, path-cost models, GETT, $\sqrt{K}$ accumulation rule), `perf.md`, `comparisons.md` (vs opt_einsum / JAX / PyTorch / cuTENSOR / TBLIS), `ffi.md` (P5 design-spike).

### Skeleton (kernel/codegen pending)

- **P5** `MaxBackend`: skeleton + four-step lowering pseudocode in `src/einsum/backends/max.mojo`. Full cutover needs `mojo-include-paths` re-enable + TileTensor / RuntimeLayout plumbing - see `docs/ffi.md`.
- **P11** Native CPU GETT: `src/einsum/backends/native.mojo` skeleton; TBLIS-style pack-with-permute design in the module docstring.
- **P12** Native GPU SM90 GETT: same module; WGMMA + TMA + permute-fused-pack design notes.
- **P14** `MaxGraphBackend`: Python-side `classify_pair(lhs, rhs, out)` + `plan_to_graph_spec(eq, shapes, path)` emit `(matmul | transpose | reduce_sum | diagonal, payload)` ops. `max.graph.ops` codegen pass remains; gated on `max` Python package install.

### Pending

- **P10** GPU dispatch validation: blocked on P5.

### Tests

- 416 Python tests + 9 Mojo smoke checks.
- 4 framework-interop tests skip when torch / jax / mlx aren't installed.
- JAX corpus parity (59 cases): mirrors `lax_numpy_einsum_test.py` - one/two/three/four/five-operand, multi-axis diagonal, ellipsis (leading / trailing / middle / full-axis), rank-6 dense, integer bit-exact subset.
- opt_einsum path parity (60 cases on 30 tensor-network shapes): greedy FLOPs <= opt_einsum greedy; DP optimal matches exactly for n <= 8.
- Hypothesis property suite (25 cases): parser determinism, working-set semantics across all optimizers, optimal/branch-all FLOP <= greedy ordering, cache idempotency, transpose involution, outer-product, full-reduction, trace, identity, dtype preservation, mixed-dtype promotion, ellipsis matmul parity over prefix_rank in [0, 3], `branch-2 <= greedy`, `auto <= naive`, deterministic-flag bit-equality, size-1 dim handling, integer matmul bit-exact equivalence.
- LRU eviction + documented "seven gotchas" (diagonal on non-contiguous input, ellipsis mismatched-rank prefix, broadcast-against-singleton, integer bit-exact at K=256).

### Architecture decisions

- **Equation IR uses int-interned labels**, not chars - supports more than 52 distinct labels.
- **Runtime parsing only**, with the JIT plan cache providing the specialization story. Compile-time `StringLiteral` parser deferred until Mojo exposes parameter-time byte indexing.
- **MAX is a backend, not the lowering target.** The library defines its own `ContractionPlan` IR; backends (`reference`, `max`, `native`, `max_graph`) consume the plan.
- **`linalg.batched_matmul` is the fast-path target** inside `MaxBackend`. Rank-2 `linalg.matmul` `comptime asserts` rank-2 and is unusable as the unified path.
- **`TileTensor` is the canonical operand type** across MAX-using backends from P5 onward.
- **DLPack-first interop** with `__array_interface__` / `np.asarray` fallback. One adapter covers numpy / jax / torch / mlx / cupy / tensorflow.

### Out of scope for v0.1

- Compile-time `StringLiteral` parser (revisit when Mojo exposes parameter-time byte indexing).
- Sparse einsum (TACO / structured-sparsity).
- Autodiff hooks.
- Distributed contractions (mesh-aware / SPMD).
- Cotengra-class hypergraph-partitioning paths.
- ROCm-specific GETT.
