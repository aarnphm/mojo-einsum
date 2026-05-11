# Plan verification — claim → test map

The plan's `## Verification` section makes eight claims. This doc pins each one to the test that exercises it, so reviewers can audit coverage without grepping.

Paths are repo-relative. Counts are commit-time; rerun `pytest --collect-only -q` for current numbers.

---

## 1. Semantic parity (≥150 cases)

> Outputs match `numpy.einsum(eq, *ops, optimize=True)` within `atol=1e-5` (fp32) / `1e-2` (bf16). Bit-exact reference for integer dtypes.

| Where | Scope |
|---|---|
| `tests/python/test_numpy_parity.py` | Hand-authored equations covering the v0.1 grammar (basic, multi-char, ellipsis, trace, diagonal, implicit output, multi-operand). |
| `tests/python/test_jax_corpus.py` | 59 cases lifted verbatim from `~/workspace/jax/tests/lax_numpy_einsum_test.py` — hand-authored + dask-derived + int64 subset. 100% pass at last run. |
| `tests/python/test_property.py` | Hypothesis-generated random shapes + label permutations against `np.einsum`. |
| `tests/python/test_explicit_path.py` | Caller-supplied paths produce identical results to planner-chosen paths. |

Plan asks for ≥150 cases; hand-authored + JAX corpus + hypothesis clears it.

---

## 2. Path quality vs `opt_einsum`

> Our `greedy` path equals or beats `opt_einsum.contract_path(optimize='greedy')` on `reduced_size` cost. DP `optimal` matches exactly for n ≤ 12. `random-greedy-128` within 5% on n ≤ 30.

| Where | Scope |
|---|---|
| `tests/python/test_opt_einsum_parity.py::test_greedy_at_least_as_good_as_opt_einsum` | 30-case corpus; asserts `moeinsum greedy FLOPs ≤ opt_einsum greedy FLOPs × 1.05` per case. |
| `tests/python/test_opt_einsum_parity.py::test_optimal_matches_opt_einsum_optimal` | Same corpus, n ≤ 8 subset; asserts exact FLOP equality on the Bellman-Held-Karp output. |
| `tests/python/test_path.py` | Hand-verified paths for the Bellman matrix chain, BMM, attention, star network, MoE routing. |

Gap: `random-greedy-N` is wired structurally (parser, dispatcher) but the 5%-band claim has no parametric test over n ≤ 30. See Outstanding §1.

---

## 3. JIT cache effectiveness

> Hot-path latency for repeated `einsum("ij,jk->ik", a, b)` is dominated by backend execute(), not parsing/planning.

| Where | Scope |
|---|---|
| `tests/python/test_cache_and_edges.py` | LRU eviction at the size cap; MRU promotion; clear behavior; concurrent.futures-driven 8-thread × 32-iter stress with an RLock-backed cache. |
| `tests/python/test_path.py::test_path_cache_hit_returns_fresh_list` | Cache hits return fresh lists, so caller mutation doesn't pollute subsequent hits. |

Hot-vs-cold microbenchmark lives in `python/moeinsum/bench.py` (call twice, compare medians); not CI-gated. See Outstanding §3.

---

## 4. Performance — fast path

> Matmul-shaped einsums within 5% of direct `linalg.batched_matmul` on the same shapes.

Status: **blocked**. `MaxBackend` dispatches to `linalg.batched_matmul` but is gated on `mojo-include-paths`, currently removed from `pyproject.toml` after a build-time MLIR error. Design at `docs/ffi.md`; once unblocked, the test belongs at `tests/python/test_max_backend_perf.py`.

---

## 5. Performance — irregular

> `'abcd,dcba->'`-class contractions within 2× of cuTENSOR (GPU) / TBLIS (CPU) at the GETT phase.

Status: **blocked** on P11/P12 (`NativeOptimizedBackend`). Skeleton at `src/einsum/backends/native.mojo` raises a phase-aware error; bench harness in `python/moeinsum/bench.py` (`--sweep-optimizers`, `--vs-numpy`).

---

## 6. Python interop (DLPack)

> `moeinsum.einsum('ij,jk->ik', a, b)` round-trips for `a, b ∈ {numpy, torch, jax, mlx}` arrays via DLPack.

| Where | Scope |
|---|---|
| `tests/python/test_interop.py` | numpy ↔ torch / jax / mlx / cupy / tensorflow round-trips via `_interop`. Skips cleanly when a framework isn't importable (4 skips in the default venv). |
| `tests/python/test_property.py::test_dtype_preservation` | First-operand framework dictates return type; `return_type=` overrides. |

---

## 7. Numerical sharp edges

> Seven known gotchas: diagonal on non-contiguous, repeated index + broadcast, low-precision accumulation, ellipsis with mismatched rank, broadcast-against-singleton, fp32-accumulation for K > 64, integer-dtype bit-exact reduction.

| Gotcha | Test |
|---|---|
| Diagonal on non-contiguous input | `test_cache_and_edges.py::test_diagonal_on_non_contiguous` |
| Ellipsis with mismatched rank prefix | `test_cache_and_edges.py::test_ellipsis_mismatched_prefix` |
| Broadcast-against-singleton | `test_cache_and_edges.py::test_broadcast_against_singleton` |
| Integer-dtype bit-exact at K=256 | `test_cache_and_edges.py::test_int_bit_exact_at_k_256` |
| `accum_dtype` validation surface | `test_cache_and_edges.py::test_accum_dtype_validation` |
| `deterministic` flag handshake | `test_property.py::test_deterministic_bit_equality` |
| K > 64 bf16 / fp32-accum drift | **blocked** on `MaxBackend` — reference backend always accumulates in fp64, so the drift can't be exercised yet. |

---

## 8. Cross-platform bench matrix

> `{M3/M4-Max, A100, H100} × {fp32, bf16, fp16} × {square-BMM, irregular, rank-3 contraction}` — JSON output committed alongside docs.

Status: **blocked** on hardware. CLI is wired:

```bash
moeinsum-bench "ij,jk->ik" --shapes 1024,1024 1024,1024 --sweep-optimizers --vs-numpy
moeinsum-bench "bij,bjk->bik" --shapes 32,128,128 32,128,128 --repeats 100
```

JSON schema validated by `tests/python/test_bench_cli.py` (8 subprocess cases).

---

## Outstanding (non-gated, actionable now)

Don't require FFI; would tighten verification:

1. **§2 `random-greedy-N` 5%-band** — parametric test over n ∈ {12, 20, 30} vs opt_einsum DP. ~30 lines.
2. **Mojo unit tests** for `_flop_cost` / `_reduced_size_cost` in `src/einsum/path.mojo`. Exercised end-to-end today; direct tests catch regressions earlier.
3. **§3 cache microbenchmark** — `bench.py --cache-bench` emitting hot/cold ratio. Structurally verified, not numerically pinned.

---

## Plan items still gated

§4, §5, §7-row-7, and §8 above are blocked on hardware or the FFI seam. Design at `docs/ffi.md`.
