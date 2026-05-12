# Plan verification - claim -> test map

The plan's `## Verification` section makes eight claims. Each row below pins one claim to the test that exercises it.

Paths are repo-relative. Python counts are a snapshot; rerun `pytest --collect-only -q` for current numbers. Mojo tests live under `tests/mojo/` and run via `mojo run -I src tests/mojo/<file>.mojo`.

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

| Where                                                                                | Scope                                                                                                                                                                                                                                                                                                                             |
| ------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tests/python/test_opt_einsum_parity.py::test_greedy_at_least_as_good_as_opt_einsum` | 30-case corpus; asserts `moeinsum greedy FLOPs <= opt_einsum greedy FLOPs * 1.05` per case.                                                                                                                                                                                                                                       |
| `tests/python/test_opt_einsum_parity.py::test_optimal_matches_opt_einsum_optimal`    | Same corpus, n <= 8 subset; asserts exact FLOP equality on the Bellman-Held-Karp output.                                                                                                                                                                                                                                          |
| `tests/python/test_random_greedy_band.py`                                            | $n \in \{12, 16, 20, 25, 30\}$; `random-greedy-128` FLOPs <= opt_einsum DP \* 1.05. Also pins N=128 <= N=32 (monotone in trials) and N=1 = greedy (no-noise degenerate).                                                                                                                                                          |
| `tests/python/test_path.py`                                                          | Hand-verified paths for the Bellman matrix chain, BMM, attention, star network, MoE routing.                                                                                                                                                                                                                                      |
| `tests/mojo/smoke_compute_path.mojo`                                                 | Mojo-side `compute_path` smoke on $n \in \{12, 16, 20\}$ across the full algorithm family. Asserts well-formedness (step count = n-1, indices in `[0, working_set_size)`, no self-pairs) and `branch-1 == greedy`. Path quality stays in the Python parametric tests; this catches shape / index regressions in the planner glue. |

---

## 3. JIT cache effectiveness

> Hot-path latency for repeated `einsum("ij,jk->ik", a, b)` is dominated by backend execute(), not parsing/planning.

| Where                                                                 | Scope                                                                                                                                                                                                                                                                                |
| --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `tests/python/test_cache_and_edges.py`                                | `PLAN_CACHE` LRU eviction at the size cap; MRU promotion; clear behavior; concurrent.futures-driven 8-thread x 32-iter stress with an RLock-backed cache.                                                                                                                            |
| `tests/python/test_max_backend.py` (`test_max_backend_model_cache_*`) | `_MODEL_CACHE` invariants on the explicit MAX Graph interop path: identical signature hits the cache (cache grows by 1 not 2); dtype + shape are part of the key; LRU eviction caps at `_MODEL_CACHE_MAX`; hit promotes to MRU so freshly-touched entries survive the next eviction. |
| `tests/python/test_path.py::test_path_cache_hit_returns_fresh_list`   | Cache hits return fresh lists, so caller mutation doesn't pollute subsequent hits.                                                                                                                                                                                                   |
| `tests/python/test_bench_cli.py::test_module_entry_cache_bench`       | Subprocess invocation of `moeinsum-bench --cache-bench`: clears `PLAN_CACHE`, times one cold call and `--repeats` hot calls, emits `cold_ms` / `hot_ms_median` / `cache_speedup_ratio`. Ratio asserted `> 0` only - perf-counter noise dominates at small sizes.                     |
| `docs/fixtures/cache_bench_example.json`                              | Frozen `--cache-bench` snapshot (`ij,jk,kl->il` on M3-class arm64 / Darwin). Schema of record; magnitudes drift across machines. The ~1.065 hot/cold ratio matches Section 3: the reference backend's per-call compute dominates, so parse+plan win is small in absolute time.       |

---

## 4. Performance - fast path

> Matmul-shaped einsums within 5% of direct `linalg.batched_matmul` on the same shapes.

Status: **shipped** across native CPU and graph executable paths, with a lazy native GPU export added for accelerator hosts. Public `backend="max:cpu"` now calls the Mojo MAX TileTensor backend through typed pointer payload entries (`_native.einsum_max_f32_cpu_ptrs` / `_native.einsum_max_f64_cpu_ptrs`). `backend="max"` tries `_native_gpu` on accelerator hosts when that extension is built, then falls through to MAX Graph. The graph interop perf tests keep the raw `max.graph.ops.matmul` comparison pinned, while the public CPU parity tests pin the TileTensor cutover.

| Where                                                                                        | Scope                                                                                                                                     |
| -------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `tests/mojo/smoke_parse.mojo::check_max_backend_matmul`                                      | Mojo seam: `src/einsum/backends/max.mojo::execute_max` lowers `ij,jk->ik` and `ji,jk->ki` pairwise steps through packed `TileTensor` BMM. |
| `tests/python/test_max_backend.py::test_max_cpu_matches_numpy`                               | Public `backend="max:cpu"` parity across matmul, BMM, outer, chains, broadcast, ellipsis, reductions, transpose, and diagonal cases.      |
| `tests/python/test_max_backend.py::test_public_max_cpu_bypasses_python_graph_cache`          | Public `max:cpu` does not populate the Python MAX Graph model cache, proving the native CPU route is used.                                |
| `tests/python/test_max_backend_perf.py::test_max_graph_interop_matches_raw_max_graph_matmul` | Numerical: graph interop output matches a hand-built `max.graph.ops.matmul` graph on `(256,256)` / `(512,512)` fp32 within `atol=1e-5`.   |
| `tests/python/test_max_backend_perf.py::test_max_graph_interop_overhead_within_factor`       | Hot-path graph ratio: 5 warmup + 25 timed iters, interop / raw `<= 1.5x` at `size=512`; the wider bound accounts for CI noise.            |

---

## 5. Performance - irregular

> `'abcd,dcba->'`-class contractions within $2\times$ of cuTENSOR (GPU) / TBLIS (CPU) at the GETT phase.

Status: **semantic backend shipped, optimized kernels pending**. `backend="native"` executes the Mojo flat-buffer plan engine and is covered by parity tests; optimized GETT CPU/GPU kernels remain post-v0.1 perf work. Bench harness lives in `python/moeinsum/_cli/bench.py` (`--sweep-optimizers`, `--vs-numpy`).

---

## 6. Python interop (DLPack)

> `moeinsum.einsum('ij,jk->ik', a, b)` round-trips for `a, b in {numpy, torch, jax, mlx}` arrays via DLPack.

| Where                                                    | Scope                                                                                                                                                      |
| -------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tests/python/test_interop.py`                           | numpy <-> torch / jax / mlx / cupy / tensorflow round-trips via `_interop`. Skips cleanly when a framework isn't importable (4 skips in the default venv). |
| `tests/python/test_property.py::test_dtype_preservation` | First-operand framework dictates return type; `return_type=` overrides.                                                                                    |

---

## 7. Numerical sharp edges

> Known numerical gotchas: diagonal on non-contiguous, ellipsis with mismatched rank, broadcast-against-singleton, repeated-index + broadcast, integer-dtype bit-exact reduction, `accum_dtype` validation, `deterministic` flag handshake, K > 64 bf16 / fp32-accum drift.

| Gotcha                               | Test                                                                                                                                                                                                                                                                                                                                                                                              |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Diagonal on non-contiguous input     | `test_cache_and_edges.py::test_diagonal_on_non_contiguous`                                                                                                                                                                                                                                                                                                                                        |
| Ellipsis with mismatched rank prefix | `test_cache_and_edges.py::test_ellipsis_mismatched_prefix`                                                                                                                                                                                                                                                                                                                                        |
| Broadcast-against-singleton          | `test_cache_and_edges.py::test_broadcast_against_singleton_batch_axis` (numpy-parity on a batch label) + `::test_broadcast_against_singleton_contract_axis` (broadcast on the contracted label) + `::test_broadcast_real_size_mismatch_still_rejected` (3≠5 still raises) + `::test_within_operand_size_mismatch_still_rejected` (`ii->` on (1,3) still raises — broadcast is cross-operand only) |
| Repeated index + broadcast           | `test_cache_and_edges.py::test_repeated_label_with_cross_operand_broadcast` (`ii,ij->j` with `(1,1)`/`(3,4)` - diagonal-extract collapses to size 1 on `i`, then cross-operand broadcast lifts it to 3)                                                                                                                                                                                           |
| Integer-dtype bit-exact at K=256     | `test_cache_and_edges.py::test_int_bit_exact_at_k_256`                                                                                                                                                                                                                                                                                                                                            |
| `accum_dtype` validation surface     | `test_cache_and_edges.py::test_accum_dtype_validation`                                                                                                                                                                                                                                                                                                                                            |
| `deterministic` flag handshake       | `test_property.py::test_deterministic_bit_equality`                                                                                                                                                                                                                                                                                                                                               |
| K > 64 bf16 / fp32-accum drift       | `test_max_backend_bf16.py::test_bf16_matmul_drift_within_one_percent` (K $\in \{128, 256\}$ rel-drift $< 1\%$) + `test_bf16_drift_grows_sublinearly_in_k` (sqrt-K growth ratio $< 3\times$ from K=64 to K=256, rejecting linear-K bf16-accum regression).                                                                                                                                         |

---

## 8. Cross-platform bench matrix

> `{M3/M4-Max, A100, H100}` $\times$ `{fp32, bf16, fp16}` $\times$ `{square-BMM, irregular, rank-3 contraction}` - JSON output committed alongside docs.

Status: **blocked** on hardware. CLI is wired:

```bash
moeinsum-bench "ij,jk->ik" --shapes 1024,1024 1024,1024 --sweep-optimizers --vs-numpy
moeinsum-bench "bij,bjk->bik" --shapes 32,128,128 32,128,128 --repeats 100
```

JSON schema validated by `tests/python/test_bench_cli.py` (subprocess coverage for `--include-path`, `--sweep-optimizers`, `--vs-numpy`, `--compare-engines`, `--cache-bench`, `--dtype bfloat16`, `random-greedy-N`, invalid-equation and explicit-path rejection, `[project.scripts]` console entry).

---

## Outstanding (non-gated, actionable now)

No current non-gated items are listed. The cache-bench JSON fixture and Mojo `compute_path` smoke live at `docs/fixtures/cache_bench_example.json` and `tests/mojo/smoke_compute_path.mojo`.

---

## Plan items still gated

Section 5's optimized GETT target and Section 8's cross-platform bench JSON remain blocked: GETT is post-v0.1 perf work, and the JSON waits on running `moeinsum-bench` on the B200 box. Design lives in `docs/ffi.md`.
