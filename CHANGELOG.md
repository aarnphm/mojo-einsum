# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions track the phase ladder in `progress.md`.

## [Unreleased] - v0.1 in progress

### Public API

- `einsum(eq, *operands, backend, optimize, accum_dtype, dtype, return_type, deterministic)` — DLPack-first input, framework-native return (torch in → torch out), `deterministic=True` as the reference-backend bit-equality contract.
- `einsum_path(eq, *shapes, optimize)` — every named optimizer plus caller-supplied paths `[(i, j), ...]`.
- `parse_equation(eq)` — IR introspection.
- `path_cost(eq, shapes, path)` — FLOP + peak-intermediate accounting.
- `moeinsum-bench` console script — single-equation timing + `--sweep-optimizers` ratios table.

### Shipped

- **P0** Scaffolding. mohaus + uv layout, dependency declarations, dir structure.
- **P1** Parser + plan IR + reference backend + flat-list FFI + numpy bridge.
- **P2** Parser: basic, implicit output, ellipsis, trace, diagonal, multi-char labels (int interning), size-1 per-label broadcast (`(1, N) -> N` cross-operand merge; within-operand strict pass preserves `ii->` on `(1,3)` rejection).
- **P3** Unary kernels: layout-only transpose + diagonal views, tiled scalar `reduce_sum`.
- **P4** Path optimizers: greedy, optimal-DP, `random-greedy(-N)`, `branch-{all,2,1}`, `auto` (opt_einsum threshold table), explicit caller-supplied paths. All wired through FFI.
- **P5** Default MAX backend: `python/moeinsum/_max_backend.py` is the public executable MAX Graph bridge for `backend="max[:cpu|gpu]"`; Modular/MAX is a default dependency. The Mojo backend seam in `src/einsum/backends/max.mojo` now packs the flat-buffer ABI into BMM-shaped `TileTensor` buffers and lowers pairwise steps through `linalg.bmm.batched_matmul`.
- **P6** Multi-operand orchestration: working-set semantics in `build_naive_plan` verified on 4/5-chain tests. Peak-sized `ContractionContext` arena deferred to the post-v0.1 kernel cutover.
- **P7** Plan cache: Python-side LRU keyed by `(eq, shape_sig, dtype, backend, optimize, accum_dtype, target)`. Stores immutable snapshots so caller mutation doesn't pollute hits. `_MODEL_CACHE` (executable MAX path) is independently LRU-bounded with eviction + MRU promotion pinned.
- **P8** DLPack interop: `_interop.to_numpy` preserves dtype; `_interop.from_numpy(arr, kind)` round-trips to torch / jax / mlx / cupy / tensorflow. fp64 cast lives at the FFI boundary, not the API. `jaxlib` → `jax` alias in `source_kind` keeps jax round-trips from collapsing to numpy.
- **P9** Precision parameters wired (`accum_dtype`, `dtype`, `deterministic`); MAX casts pairwise inputs and reductions to fp32/fp64 accumulator dtype, rejects low-precision accumulators, and preserves bf16 output dtype. CPU MAX rejects bf16 graph inputs, so the CPU bf16 route compiles fp32 graphs and casts the public result back to bf16. Reference backend always accumulates in fp64.
- **P10** GPU dispatch validation, Python-side: `backend="max:gpu"` flows through `max.graph` and is exercised in `tests/python/test_max_backend.py`. B200 smoke passes on the supported BMM / ellipsis subset; broader hardware sweeps remain task #32.
- **P11/P12** Native backend: `backend="native"` executes the Mojo flat-buffer `ContractionPlan` engine across pairwise and unary steps. SIMD CPU GETT and SM90 WGMMA are now explicit post-v0.1 perf work, not a separate API.
- **P13** Bench CLI: JSON output, per-platform metadata, median/min/max timing, optional path introspection, optimizer-sweep with ratios.
- **P13** `--cache-bench` cold/hot ratio (§3 audit); `--compare` / `--compare-engines` against numpy / opt_einsum / mlx; `--dtype bfloat16` via `ml_dtypes` with numpy-skip handshake.
- **P14** `MaxGraphBackend`: Python-side lowering spec and execution now live in `_max_backend.py`. The old `_max_graph.py` alias/module is gone; `classify_pair(...)`, `lowering_spec(...)`, and `MaxGraphBackend.execute(...)` share the executable code path. Repeated labels lower through `gather_nd`, reductions cast to the accumulator dtype, and pairwise steps lower through MAX Graph matmul.
- **P14** Executable MAX ellipsis expansion: `backend="max[:cpu|gpu]"` rewrites `...` to synthetic internal labels before graph lowering, including right-aligned broadcast ellipses and implicit-output ellipsis-first ordering.
- **P15** Docs (6 files): `notation.md` (size-1 broadcast pinned), `derivations.md` (BMM lowering, path-cost models, GETT, $\sqrt{K}$ accumulation rule), `perf.md`, `comparisons.md` (vs opt_einsum / JAX / PyTorch / cuTENSOR / TBLIS), `plan-verification.md` (8 claim rows backed), `ffi.md` (optional TileTensor cutover notes).

### Kernel Perf Follow-Up

- **Mojo TileTensor MAX cutover**: `src/einsum/backends/max.mojo` now owns the packed `TileTensor` pairwise path. Remaining kernel work is zero-copy `RuntimeLayout` / DLPack FFI, output-order avoidance, and GPU target selection.
- **Native CPU GETT**: flat-buffer executor shipped; TBLIS-style pack-with-permute kernel work remains. Design in `docs/derivations.md` §3.
- **Native GPU SM90 GETT**: flat-buffer executor shipped; WGMMA + TMA + permute-fused-pack kernel work remains.

### Hardware-backed follow-up

- **Task #32** Broader B200 benchmark sweep against `backend="max:gpu"` (`coder ssh vpham/aaron`).
- **Task #33** §8 cross-platform bench matrix — commit JSON fixtures across `{fp32, bf16, fp16} × {square-BMM, irregular, rank-3}` on M3/M4-Max + B200.

### Tests

- 449 passed / 6 skipped Python tests + 3 Mojo smoke files.
- Skips break down as: torch / mlx / cupy / tf framework-interop tests when the framework isn't importable, plus `opt_einsum` / `tqdm` optional-extra gates. Each surfaces a clear reason via `pytest -rs`.
- JAX corpus parity (59 cases): mirrors `lax_numpy_einsum_test.py` — one/two/three/four/five-operand, multi-axis diagonal, ellipsis (leading / trailing / middle / full-axis), rank-6 dense, integer bit-exact subset.
- opt_einsum path parity (60 cases on 30 tensor-network shapes): greedy FLOPs ≤ opt_einsum greedy; DP optimal matches exactly for $n \le 8$.
- Hypothesis property suite (26 test functions): parser determinism, working-set semantics across all optimizers, optimal/branch-all FLOP ≤ greedy ordering, cache idempotency, transpose involution, outer-product, full-reduction, trace, identity, dtype preservation, mixed-dtype promotion, ellipsis matmul parity over `prefix_rank` in [0, 3], `branch-2 ≤ greedy`, `auto ≤ naive`, deterministic-flag bit-equality, size-1 dim handling, integer matmul bit-exact equivalence.
- LRU eviction (`_PLAN_CACHE` + `_MODEL_CACHE` both bounded, MRU promotion + eviction pinned) and documented sharp edges: diagonal on non-contiguous input, ellipsis mismatched-rank prefix, broadcast-against-singleton, integer bit-exact at $K=256$, $K > 64$ bf16 $\sqrt{K}$ drift.
- Plan §2 random-greedy band: parametric on $n \in \{12, 16, 20, 25, 30\}$ against opt_einsum DP, asserts ≤ 5% FLOP-cost band.
- Plan §3 cache-bench fixture: `docs/fixtures/cache_bench_example.json` + subprocess coverage in `test_bench_cli.py::test_module_entry_cache_bench`.
- Mojo `compute_path` smoke: full algorithm family (greedy / optimal / branch-{all,2,1} / random-greedy / explicit) across $n \in \{12, 16, 20\}$.

### Architecture decisions

- **Equation IR uses int-interned labels** rather than chars — supports more than 52 distinct labels.
- **Runtime parsing only**, with the JIT plan cache providing the specialization story. Compile-time `StringLiteral` parser deferred until Mojo exposes parameter-time byte indexing.
- **MAX is a backend, not the lowering target.** The library defines its own `ContractionPlan` IR; backends (`reference`, `max`, `native`) consume the plan.
- **`max.graph.ops.matmul` is the public fast-path target** for v0.1. The Mojo backend seam uses `linalg.bmm.batched_matmul` internally without adding a second Python surface.
- **`TileTensor` stays behind the backend seam** rather than leaking into the v0.1 public backend contract.
- **DLPack-first interop** with `__array_interface__` / `np.asarray` fallback. One adapter covers numpy / jax / torch / mlx / cupy / tensorflow.

### Out of scope for v0.1

- Compile-time `StringLiteral` parser (revisit when Mojo exposes parameter-time byte indexing).
- Sparse einsum (TACO / structured-sparsity).
- Autodiff hooks.
- Distributed contractions (mesh-aware / SPMD).
- Cotengra-class hypergraph-partitioning paths.
- ROCm-specific GETT.
