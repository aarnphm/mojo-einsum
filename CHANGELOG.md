# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions track the phase ladder in `progress.md`.

## [Unreleased] - v0.1 in progress

### Public API

- `einsum(eq, *operands, backend, optimize, accum_dtype, dtype, return_type, deterministic)` — DLPack-first input, framework-native return (torch in → torch out), `deterministic=True` as the bit-equality contract.
- `einsum_path(eq, *shapes, optimize)` — every named optimizer plus caller-supplied paths `[(i, j), ...]`.
- `parse_equation(eq)` — IR introspection.
- `path_cost(eq, shapes, path)` — FLOP + peak-intermediate accounting.
- `moeinsum-bench` console script — single-equation timing + `--sweep-optimizers` ratios table.

### Shipped

- **P0** Scaffolding. mohaus + uv layout, dependency declarations, dir structure.
- **P1** Parser + plan IR + reference backend + flat-list FFI + numpy bridge. Mojo 1.0 syntax.
- **P2** Parser: basic, implicit output, ellipsis, trace, diagonal, multi-char labels (int interning), size-1 per-label broadcast (`(1, N) -> N` cross-operand merge; within-operand strict pass preserves `ii->` on `(1,3)` rejection).
- **P3** Unary kernels: layout-only transpose + diagonal views, tiled scalar `reduce_sum`.
- **P4** Path optimizers: greedy, optimal-DP, `random-greedy(-N)`, `branch-{all,2,1}`, `auto` (opt_einsum threshold table), explicit caller-supplied paths. All wired through FFI.
- **P6** Multi-operand orchestration: working-set semantics in `build_naive_plan` verified on 4/5-chain tests. Peak-sized `ContractionContext` arena deferred to P5.
- **P7** Plan cache: Python-side LRU keyed by `(eq, shape_sig, dtype, backend, optimize, accum_dtype, target)`. Stores immutable snapshots so caller mutation doesn't pollute hits. `_MODEL_CACHE` (executable MAX path) is independently LRU-bounded with eviction + MRU promotion pinned.
- **P8** DLPack interop: `_interop.to_numpy` preserves dtype; `_interop.from_numpy(arr, kind)` round-trips to torch / jax / mlx / cupy / tensorflow. fp64 cast lives at the FFI boundary, not the API. `jaxlib` → `jax` alias in `source_kind` keeps jax round-trips from collapsing to numpy.
- **P9** Precision parameters wired (`accum_dtype`, `dtype`, `deterministic`); bf16 inputs route through MAX with an fp32 accumulator. K > 64 drift held under 1% with $\sqrt{K}$ growth (rejects linear-K bf16-accumulator regressions). Reference backend always accumulates in fp64.
- **P10** GPU dispatch validation, Python-side: `backend="max:gpu"` flows through `max.graph` and is exercised in `tests/python/test_max_backend.py`. Hardware sweep on B200 is task #32; Mojo-side `target="gpu"` waits on P5.
- **P13** Bench CLI: JSON output, per-platform metadata, median/min/max timing, optional path introspection, optimizer-sweep with ratios.
- **P13** `--cache-bench` cold/hot ratio (§3 audit); `--compare` / `--compare-engines` against numpy / opt_einsum / mlx; `--dtype bfloat16` via `ml_dtypes` with numpy-skip handshake.
- **P13** `is_loadable()` argparse gate on `--backend max:*` so MAX ABI mismatches surface as a clean error rather than a 30-line dlopen stacktrace.
- **P14** `MaxGraphBackend`: Python-side plan-to-graph translation — `classify_pair(lhs, rhs, out)`, `plan_to_graph_spec(eq, shapes, path)`, `DimClassification`, `GraphSpec`. `MaxGraphBackend.execute(...)` bridges through `_max_backend.execute_max`; the spec is end-to-end executable on the BMM-lowerable subset.
- **P15** Docs (6 files): `notation.md` (size-1 broadcast pinned), `derivations.md` (BMM lowering, path-cost models, GETT, $\sqrt{K}$ accumulation rule), `perf.md`, `comparisons.md` (vs opt_einsum / JAX / PyTorch / cuTENSOR / TBLIS), `plan-verification.md` (8 claim rows backed), `ffi.md` (P5 design-spike).

### Skeleton (kernel/codegen pending)

- **P5** Mojo `MaxBackend`: skeleton + four-step lowering pseudocode in `src/einsum/backends/max.mojo`. Full cutover needs `mojo-include-paths` re-enable + TileTensor / RuntimeLayout plumbing — see `docs/ffi.md`. The Python-side MAX path (`_max_backend.py`, reached via `backend="max[:cpu|gpu]"`) ships and covers the same lowering surface today.
- **P11** Native CPU GETT: `src/einsum/backends/native.mojo` skeleton raises a phase-aware error; TBLIS-style pack-with-permute design in `docs/derivations.md` §3.
- **P12** Native GPU SM90 GETT: same module; WGMMA + TMA + permute-fused-pack design notes.

### Hardware-blocked

- **Task #32** B200 GPU sweep against `backend="max:gpu"` (`coder ssh vpham/aaron`).
- **Task #33** §8 cross-platform bench matrix — commit JSON fixtures across `{fp32, bf16, fp16} × {square-BMM, irregular, rank-3}` on M3/M4-Max + B200.

### Tests

- 478 passed / 33 skipped Python tests + 3 Mojo smoke files.
- Skips break down as: torch / mlx / cupy / tf framework-interop tests when the framework isn't importable, `max.graph` tests when `is_loadable()` reports a bazel-vs-PyPI ABI conflict, and `opt_einsum` / `tqdm` / `ml_dtypes` optional-extra gates. Each surfaces a clear reason via `pytest -rs`.
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
- **MAX is a backend, not the lowering target.** The library defines its own `ContractionPlan` IR; backends (`reference`, `max`, `native`, `max_graph`) consume the plan.
- **`linalg.batched_matmul` is the fast-path target** inside `MaxBackend`. Rank-2 `linalg.matmul` `comptime asserts` rank-2, which rules it out as the unified path.
- **`TileTensor` is the canonical operand type** across MAX-using backends from P5 onward.
- **DLPack-first interop** with `__array_interface__` / `np.asarray` fallback. One adapter covers numpy / jax / torch / mlx / cupy / tensorflow.

### Out of scope for v0.1

- Compile-time `StringLiteral` parser (revisit when Mojo exposes parameter-time byte indexing).
- Sparse einsum (TACO / structured-sparsity).
- Autodiff hooks.
- Distributed contractions (mesh-aware / SPMD).
- Cotengra-class hypergraph-partitioning paths.
- ROCm-specific GETT.
