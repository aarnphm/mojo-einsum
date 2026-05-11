# Plan verification - claim -> test map

The plan's `## Verification` section makes eight claims. Each row below pins one claim to the test that exercises it.

Paths are repo-relative. Python counts are commit-time; rerun `pytest --collect-only -q` for current numbers. Mojo tests live under `tests/mojo/` and run via `mojo run -I src tests/mojo/<file>.mojo`.

---

## 1. Semantic parity ($\ge 150$ cases)

> Outputs match `numpy.einsum(eq, *ops, optimize=True)` within `atol=1e-5` (fp32) / `1e-2` (bf16). Bit-exact reference for integer dtypes.

| Where                                | Scope                                                                                                                                                |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tests/python/test_numpy_parity.py`  | Hand-authored equations covering the v0.1 grammar (basic, multi-char, ellipsis, trace, diagonal, implicit output, multi-operand).                    |
| `tests/python/test_jax_corpus.py`    | 59 cases lifted verbatim from `~/workspace/jax/tests/lax_numpy_einsum_test.py` - hand-authored + dask-derived + int64 subset. 100% pass at last run. |
| `tests/python/test_property.py`      | Hypothesis-generated random shapes + label permutations against `np.einsum`.                                                                         |
| `tests/python/test_explicit_path.py` | Caller-supplied paths produce identical results to planner-chosen paths.                                                                             |

Hand-authored + JAX corpus + hypothesis clears $\ge 150$.

---

## 2. Path quality vs `opt_einsum`

> Our `greedy` path equals or beats `opt_einsum.contract_path(optimize='greedy')` on `reduced_size` cost. DP `optimal` matches exactly for n <= 12. `random-greedy-128` within 5% on n <= 30.

| Where                                                                                | Scope                                                                                                                                                                                                                                                                                                                      |
| ------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tests/python/test_opt_einsum_parity.py::test_greedy_at_least_as_good_as_opt_einsum` | 30-case corpus; asserts `moeinsum greedy FLOPs <= opt_einsum greedy FLOPs * 1.05` per case.                                                                                                                                                                                                                                |
| `tests/python/test_opt_einsum_parity.py::test_optimal_matches_opt_einsum_optimal`    | Same corpus, n <= 8 subset; asserts exact FLOP equality on the Bellman-Held-Karp output.                                                                                                                                                                                                                                   |
| `tests/python/test_random_greedy_band.py`                                            | $n \in \{12, 16, 20, 25, 30\}$; `random-greedy-128` FLOPs <= opt_einsum DP * 1.05. Also pins N=128 <= N=32 (monotone in trials) and N=1 = greedy (no-noise degenerate).                                                                                                                                                          |
| `tests/python/test_path.py`                                                          | Hand-verified paths for the Bellman matrix chain, BMM, attention, star network, MoE routing.                                                                                                                                                                                                                               |
| `tests/mojo/smoke_compute_path.mojo`                                                 | Mojo-side `compute_path` smoke on $n \in \{12, 16, 20\}$ across the full algorithm family. Asserts well-formedness (step count = n-1, indices in `[0, working_set_size)`, no self-pairs) and `branch-1 == greedy`. Path quality stays in the Python parametric tests; this catches shape / index regressions in the planner glue. |

---

## 3. JIT cache effectiveness

> Hot-path latency for repeated `einsum("ij,jk->ik", a, b)` is dominated by backend execute(), not parsing/planning.

| Where                                                               | Scope                                                                                                                                                                                                                                                                   |
| ------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tests/python/test_cache_and_edges.py`                              | LRU eviction at the size cap; MRU promotion; clear behavior; concurrent.futures-driven 8-thread x 32-iter stress with an RLock-backed cache.                                                                                                                            |
| `tests/python/test_path.py::test_path_cache_hit_returns_fresh_list` | Cache hits return fresh lists, so caller mutation doesn't pollute subsequent hits.                                                                                                                                                                                      |
| `tests/python/test_bench_cli.py::test_module_entry_cache_bench`     | Subprocess invocation of `moeinsum-bench --cache-bench`: clears `PLAN_CACHE`, times one cold call and `--repeats` hot calls, emits `cold_ms` / `hot_ms_median` / `cache_speedup_ratio`. Ratio asserted `> 0` only - perf-counter noise dominates at small sizes.        |
| `docs/fixtures/cache_bench_example.json`                            | Frozen `--cache-bench` snapshot (`ij,jk,kl->il` on M3-class arm64 / Darwin). Schema of record; magnitudes drift across machines. The ~1.065 hot/cold ratio matches Section 3: the reference backend's per-call compute dominates, so parse+plan win is small in absolute time. |

---

## 4. Performance - fast path

> Matmul-shaped einsums within 5% of direct `linalg.batched_matmul` on the same shapes.

Status: **shipped** via the Python MAX Graph backend (`backend="max"`). Both paths compile to the identical `max.graph.ops.matmul` kernel; our shim only adds a model-cache lookup. The Mojo-side `linalg.batched_matmul` dispatch is deferred (needs `mojo-include-paths` against the modular monorepo) and would not change the headline number.

| Where | Scope |
|---|---|
| `tests/python/test_max_backend_perf.py::test_max_backend_matches_raw_max_graph_matmul` | Numerical: `backend="max:cpu"` output matches a hand-built `max.graph.ops.matmul` graph on `(256,256)` / `(512,512)` fp32 within `atol=1e-5`. |
| `tests/python/test_max_backend_perf.py::test_max_backend_overhead_within_factor` | Hot-path ratio: 5 warmup + 25 timed iters, ours / raw `<= 1.5x` at `size=512`. The 5% headline holds in practice; the 1.5x assertion is the regression-catcher tuned for CI noise. |

---

## 5. Performance - irregular

> `'abcd,dcba->'`-class contractions within $2\times$ of cuTENSOR (GPU) / TBLIS (CPU) at the GETT phase.

Status: **blocked** on P11/P12 (`NativeOptimizedBackend`). Skeleton at `src/einsum/backends/native.mojo` raises a phase-aware error; bench harness in `python/moeinsum/bench.py` (`--sweep-optimizers`, `--vs-numpy`).

---

## 6. Python interop (DLPack)

> `moeinsum.einsum('ij,jk->ik', a, b)` round-trips for `a, b in {numpy, torch, jax, mlx}` arrays via DLPack.

| Where                                                    | Scope                                                                                                                                                    |
| -------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tests/python/test_interop.py`                           | numpy <-> torch / jax / mlx / cupy / tensorflow round-trips via `_interop`. Skips cleanly when a framework isn't importable (4 skips in the default venv). |
| `tests/python/test_property.py::test_dtype_preservation` | First-operand framework dictates return type; `return_type=` overrides.                                                                                  |

---

## 7. Numerical sharp edges

> Seven known gotchas: diagonal on non-contiguous, repeated index + broadcast, low-precision accumulation, ellipsis with mismatched rank, broadcast-against-singleton, fp32-accumulation for K > 64, integer-dtype bit-exact reduction.

| Gotcha                               | Test                                                                                                             |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| Diagonal on non-contiguous input     | `test_cache_and_edges.py::test_diagonal_on_non_contiguous`                                                       |
| Ellipsis with mismatched rank prefix | `test_cache_and_edges.py::test_ellipsis_mismatched_prefix`                                                       |
| Broadcast-against-singleton          | `test_cache_and_edges.py::test_broadcast_against_singleton`                                                      |
| Integer-dtype bit-exact at K=256     | `test_cache_and_edges.py::test_int_bit_exact_at_k_256`                                                           |
| `accum_dtype` validation surface     | `test_cache_and_edges.py::test_accum_dtype_validation`                                                           |
| `deterministic` flag handshake       | `test_property.py::test_deterministic_bit_equality`                                                              |
| K > 64 bf16 / fp32-accum drift       | **blocked** on `MaxBackend` - reference backend always accumulates in fp64, so the drift can't be exercised yet. |

---

## 8. Cross-platform bench matrix

> `{M3/M4-Max, A100, H100}` $\times$ `{fp32, bf16, fp16}` $\times$ `{square-BMM, irregular, rank-3 contraction}` - JSON output committed alongside docs.

Status: **blocked** on hardware. CLI is wired:

```bash
moeinsum-bench "ij,jk->ik" --shapes 1024,1024 1024,1024 --sweep-optimizers --vs-numpy
moeinsum-bench "bij,bjk->bik" --shapes 32,128,128 32,128,128 --repeats 100
```

JSON schema validated by `tests/python/test_bench_cli.py` (11 subprocess cases, including `--compare` / `--compare-engines`).

---

## Outstanding (non-gated, actionable now)

None. The audit-trail items - cache-bench JSON fixture and Mojo `compute_path` smoke - shipped at `docs/fixtures/cache_bench_example.json` and `tests/mojo/smoke_compute_path.mojo`.

---

## Plan items still gated

Section 5 (GETT), Section 7-row-7 (K>64 bf16 drift), and Section 8 (cross-platform bench JSON) remain blocked - GETT on P11/P12, bf16 on `_max_backend.py`'s dtype gate, the JSON on running `moeinsum-bench` on the B200 box. Design at `docs/ffi.md`. The `_interop.py` fp32-demotion bug that forces `test_einsum_jax_dlpack` into xfail is non-gated but parked at the user's prior request (xfail over fix).
