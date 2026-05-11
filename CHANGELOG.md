# Changelog

All notable changes land here. Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions track the project's phase ladder (see `progress.md` and the approved plan at `~/.claude/plans/let-s-plan-and-propse-iridescent-hare.md`).

## [Unreleased] — v0.1 in progress

### Public API

- `einsum(eq, *operands, backend, optimize, accum_dtype, dtype, return_type, deterministic)` — DLPack-first input adapter, framework-native return (torch in → torch out), `deterministic=True` as the documented bit-equality contract.
- `einsum_path(eq, *shapes, optimize)` — every named optimizer plus caller-supplied explicit paths `[(i, j), ...]`.
- `parse_equation(eq)` — IR introspection.
- `path_cost(eq, shapes, path)` — FLOP + peak-intermediate accounting helper.
- `moeinsum-bench` console script — single-equation timing + `--sweep-optimizers` ratios table.

### Phases shipped

- **P0** Scaffolding. mohaus + uv layout, dependency declarations, dir structure.
- **P1** Stupid einsum: parser + plan IR + reference backend + flat-list FFI + numpy bridge. Mojo 1.0 syntax.
- **P2** Parser polish: basic / implicit output / ellipsis / trace / diagonal / multi-char labels via int interning.
- **P3** Unary kernels: layout-only transpose + diagonal views, tiled scalar reduce_sum.
- **P4** Path optimizer: greedy + optimal-DP + `random-greedy(-N)` + `branch-{all,2,1}` + `auto` (opt_einsum threshold table) + explicit caller-supplied paths. All wired through the FFI.
- **P6** Multi-operand orchestration: working-set semantics in `build_naive_plan` verified on 4/5-chain tests. Peak-sized `ContractionContext` arena deferred until `MaxBackend` lands.
- **P7** JIT plan cache: Python-side LRU keyed by `(eq, shape_sig, dtype, backend, optimize, accum_dtype, target)`. Cache stores immutable snapshots — caller mutation can't pollute hits.
- **P8** DLPack interop: `_interop.to_numpy` preserves dtype by default; `_interop.from_numpy(arr, kind)` round-trips back to torch / jax / mlx / cupy / tensorflow. The fp64 cast is at the FFI boundary, not the API.
- **P9*** Precision parameters wired (`accum_dtype`, `dtype`, `deterministic`). Real low-precision accumulation lands with `MaxBackend`; today the reference backend always accumulates in fp64.
- **P13** Bench CLI: JSON output, per-platform metadata, median/min/max timing, optional path introspection, optimizer-sweep mode with ratios.
- **P15** Docs (5 markdown files): `notation.md` (einsum primer), `derivations.md` (BMM lowering, path cost models, GETT, √K accumulation rule), `perf.md` (tuning guide), `comparisons.md` (scorecard vs opt_einsum / JAX / PyTorch / cuTENSOR / TBLIS), `ffi.md` (P5 design-spike). All editor-reviewed.

### Phases with skeletons (kernel/codegen pending)

- **P5** `MaxBackend`: structural skeleton + four-step lowering pseudocode in `src/einsum/backends/max.mojo`. The full FFI cutover requires `mojo-include-paths` re-enable + TileTensor / RuntimeLayout plumbing — see `docs/ffi.md`.
- **P11** Native CPU GETT: `src/einsum/backends/native.mojo` skeleton; TBLIS-style pack-with-permute design in the module docstring.
- **P12** Native GPU SM90 GETT: same module; WGMMA + TMA + permute-fused-pack design notes.
- **P14** `MaxGraphBackend`: Python-side `classify_pair(lhs, rhs, out)` + `plan_to_graph_spec(eq, shapes, path)` shipped — emits `(matmul | transpose | reduce_sum | diagonal, payload)` ops. The `max.graph.ops` codegen pass is the remaining seam; gated on the `max` Python package install.

### Phases pending

- **P10** GPU dispatch validation: pending P5.

### Tests

- 289 active Python tests + 9 Mojo smoke checks.
- 4 framework-interop tests skip cleanly when torch / jax / mlx aren't installed.
- Hypothesis property suite (25 cases) covers parser determinism, working-set semantics across all optimizers, optimal/branch-all FLOP ≤ greedy ordering, cache idempotency, transpose involution, outer-product, full-reduction, trace, identity, dtype preservation, mixed-dtype promotion, ellipsis matmul parity over prefix_rank ∈ [0, 3], `branch-2 ≤ greedy`, `auto ≤ naive`, deterministic-flag bit-equality, size-1 dim handling, integer matmul bit-exact equivalence.

### Architecture decisions

- **Equation IR uses int-interned labels**, not chars — supports more than 52 distinct labels.
- **Runtime parsing only**, with a JIT plan cache providing the specialization story. Compile-time `StringLiteral` parser deferred (Mojo doesn't yet expose parameter-time byte indexing).
- **MAX is a backend, not the lowering target.** Library defines its own `ContractionPlan` IR; backends (`reference`, `max`, `native`, `max_graph`) consume the plan.
- **`linalg.batched_matmul` is the practical fast-path target** inside `MaxBackend` — not rank-2 `linalg.matmul` (the latter `comptime asserts` rank-2).
- **`TileTensor` will be the canonical operand type** across MAX-using backends from P5 onward.
- **DLPack-first interop** with `__array_interface__` / `np.asarray` fallback. One adapter covers numpy / jax / torch / mlx / cupy / tensorflow.

### Out of scope for v0.1

- Compile-time `StringLiteral` parser (revisit once Mojo exposes parameter-time byte indexing).
- Sparse einsum (TACO / structured-sparsity).
- Automatic differentiation hooks.
- Distributed contractions (mesh-aware / SPMD).
- Cotengra-class hypergraph-partitioning paths.
- ROCm-specific GETT.
