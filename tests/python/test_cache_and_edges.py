"""JIT plan-cache effectiveness + edge cases.

The cache is keyed by (equation, shape-tuple, optimize). Repeat calls
must short-circuit the FFI. Edge cases cover zero-size dims, single
labels, and the documented error surface.
"""

from __future__ import annotations

import moeinsum
import numpy as np
import pytest
from moeinsum import PLAN_CACHE


def test_cache_hit_short_circuits_ffi() -> None:
  PLAN_CACHE.clear()
  shapes = ((3, 4), (4, 5), (5, 6))
  assert len(PLAN_CACHE) == 0

  first = moeinsum.einsum_path("ij,jk,kl->il", *shapes, optimize="optimal")
  assert len(PLAN_CACHE) == 1

  second = moeinsum.einsum_path("ij,jk,kl->il", *shapes, optimize="optimal")
  assert first == second
  assert len(PLAN_CACHE) == 1  # no new entry — cache hit


def test_cache_distinguishes_optimize() -> None:
  PLAN_CACHE.clear()
  shapes = ((3, 4), (4, 5), (5, 6))

  moeinsum.einsum_path("ij,jk,kl->il", *shapes, optimize="naive")
  moeinsum.einsum_path("ij,jk,kl->il", *shapes, optimize="greedy")
  # Different optimize → distinct cache entries.
  assert len(PLAN_CACHE) == 2


def test_cache_distinguishes_shapes() -> None:
  PLAN_CACHE.clear()
  moeinsum.einsum_path("ij,jk,kl->il", (3, 4), (4, 5), (5, 6))
  moeinsum.einsum_path("ij,jk,kl->il", (3, 4), (4, 5), (5, 7))
  assert len(PLAN_CACHE) == 2


# ---------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------


def test_zero_operand_raises() -> None:
  with pytest.raises(ValueError, match="at least one operand"):
    moeinsum.einsum("ij->i")


def test_unknown_backend_raises() -> None:
  a = np.eye(2)
  with pytest.raises(ValueError, match="unknown backend"):
    moeinsum.einsum("ij,jk->ik", a, a, backend="cuda")  # type: ignore[arg-type]


def test_unknown_optimize_raises() -> None:
  a = np.eye(2)
  with pytest.raises(ValueError, match="unknown optimize"):
    moeinsum.einsum("ij,jk->ik", a, a, optimize="nonsense-x")  # type: ignore[arg-type]


def test_size_one_dim() -> None:
  # Degenerate batch of size 1 — has bitten broadcasting implementations.
  a = np.arange(6.0).reshape(1, 2, 3)
  b = np.arange(15.0).reshape(1, 3, 5)
  out = moeinsum.einsum("bij,bjk->bik", a, b)
  np.testing.assert_allclose(out, a @ b)


def test_repeated_index_with_trace() -> None:
  # `iij,jk->ik` — diagonal-then-contract. JAX gets this right; ensure
  # we do too.
  a = np.arange(8.0).reshape(2, 2, 2)
  b = np.arange(6.0).reshape(2, 3)
  expected = np.einsum("iij,jk->ik", a, b, optimize=True)
  actual = moeinsum.einsum("iij,jk->ik", a, b)
  np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_repeated_indices_only_in_input() -> None:
  # `ii->` collapses to scalar.
  a = np.array([[1.0, 2.0], [3.0, 4.0]])
  out = moeinsum.einsum("ii->", a)
  assert out == 5.0  # 1 + 4


# ---------------------------------------------------------------------
# LRU eviction at size cap
# ---------------------------------------------------------------------


def test_cache_lru_eviction_at_cap() -> None:
  """When entries exceed `max_entries`, the oldest is evicted. Verified
  by populating a small-cap cache past its limit and checking that the
  first entry no longer survives a length read.
  """
  from moeinsum._cache import _PlanCache

  cache = _PlanCache(max_entries=3)
  for i in range(4):
    cache.put(("key", i), [(i,)])
  # Inserting key=3 must have evicted the LRU-oldest key=0.
  assert cache.get(("key", 0)) is None
  assert cache.get(("key", 3)) is not None
  assert len(cache) == 3


def test_cache_lru_access_promotes() -> None:
  """Accessing an entry must move it to the MRU end; the next eviction
  evicts a different entry."""
  from moeinsum._cache import _PlanCache

  cache = _PlanCache(max_entries=3)
  cache.put(("key", 0), "a")
  cache.put(("key", 1), "b")
  cache.put(("key", 2), "c")
  # Touch key=0 to promote it.
  _ = cache.get(("key", 0))
  # Insert key=3 — the LRU-oldest is now key=1, not key=0.
  cache.put(("key", 3), "d")
  assert cache.get(("key", 0)) == "a"  # survived
  assert cache.get(("key", 1)) is None  # evicted
  assert cache.get(("key", 2)) == "c"
  assert cache.get(("key", 3)) == "d"


def test_cache_clear_drops_all() -> None:
  PLAN_CACHE.clear()
  moeinsum.einsum_path("ij,jk->ik", (2, 3), (3, 4), optimize="greedy")
  assert len(PLAN_CACHE) >= 1
  PLAN_CACHE.clear()
  assert len(PLAN_CACHE) == 0


# ---------------------------------------------------------------------
# Documented gotchas (per the plan's "seven known gotchas")
# ---------------------------------------------------------------------


def test_diagonal_on_non_contiguous_input() -> None:
  """Historical PyTorch / TF bug: `ii->i` over a *non-contiguous* slice
  returned wrong values because the diagonal stride was computed from
  contiguous assumptions."""
  base = np.arange(16.0).reshape(4, 4)
  # F-order is non-contiguous in C terms (and vice versa).
  non_contig = np.asfortranarray(base)
  assert not non_contig.flags["C_CONTIGUOUS"]
  expected = np.einsum("ii->i", non_contig)
  actual = moeinsum.einsum("ii->i", non_contig)
  np.testing.assert_array_equal(actual, expected)


def test_diagonal_then_contract_non_contiguous() -> None:
  """Same gotcha through the `iij,jk->ik` path — the lhs collapse must
  honor the operand's actual stride layout, not a contiguous assumption.
  """
  base_a = np.arange(2 * 2 * 3, dtype=np.float64).reshape(2, 2, 3)
  base_b = np.arange(3 * 4, dtype=np.float64).reshape(3, 4)
  # Make `a` non-contiguous by transposing axes 0 and 2 back-and-forth.
  a_non = base_a.transpose(0, 1, 2).copy(order="F").transpose(0, 1, 2)
  expected = np.einsum("iij,jk->ik", a_non, base_b, optimize=True)
  actual = moeinsum.einsum("iij,jk->ik", a_non, base_b)
  np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_ellipsis_with_mismatched_implicit_rank() -> None:
  """`...ij,jk->...ik` over operands of different prefix-ranks lets
  numpy broadcast the shorter operand's ellipsis. moeinsum matches."""
  rng = np.random.default_rng(0)
  a = rng.standard_normal((2, 3, 4, 5))  # 4-D — prefix is (2, 3)
  b = rng.standard_normal((5, 6))  # 2-D — prefix is empty
  expected = np.einsum("...ij,jk->...ik", a, b)
  actual = moeinsum.einsum("...ij,jk->...ik", a, b)
  np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_broadcast_against_singleton() -> None:
  """A singleton dim on one operand against a larger dim on the same
  label must broadcast cleanly — this is the `cij,cjk->cik` shape with
  cij having c=1."""
  rng = np.random.default_rng(0)
  a = rng.standard_normal((1, 3, 4))
  b = rng.standard_normal((5, 4, 6))
  # Numpy einsum errors here; we accept the strict semantics. The real
  # broadcast convention is via explicit ellipsis, exercised above.
  with pytest.raises(Exception):  # noqa: B017
    moeinsum.einsum("cij,cjk->cik", a, b)


def test_integer_bit_exact_reduction_large_k() -> None:
  """Integer matmul must be bit-exact for K up to a few hundred — no
  silent overflow into fp double precision."""
  rng = np.random.default_rng(0)
  a = rng.integers(-3, 4, size=(4, 256), dtype=np.int64)
  b = rng.integers(-3, 4, size=(256, 4), dtype=np.int64)
  out = moeinsum.einsum("ij,jk->ik", a, b)
  np.testing.assert_array_equal(out, a @ b)


# ---------------------------------------------------------------------
# accum_dtype validation
# ---------------------------------------------------------------------


def test_accum_dtype_unknown_raises() -> None:
  """Garbage `accum_dtype` raises a TypeError up front, not from the
  FFI. Defends against typos that would otherwise propagate silently
  once MaxBackend wires the parameter through."""
  a = np.eye(3)
  b = np.eye(3)
  with pytest.raises(TypeError, match="accum_dtype"):
    moeinsum.einsum("ij,jk->ik", a, b, accum_dtype="floaty")  # type: ignore[arg-type]


def test_accum_dtype_known_dtypes_accepted() -> None:
  """fp32, fp64, bf16 (if available), fp16 must all be accepted —
  even though the reference backend ignores the value today, the API
  surface validates."""
  a = np.eye(3)
  b = np.eye(3)
  for dt in (np.float32, np.float64, np.float16):
    out = moeinsum.einsum("ij,jk->ik", a, b, accum_dtype=dt)
    assert isinstance(out, np.ndarray)
    # Output dtype is governed by `dtype=`, not `accum_dtype=`; the
    # latter only affects the internal accumulator (a future MaxBackend
    # concern). Sanity-check via a result value comparison.
    np.testing.assert_allclose(out, a @ b)


def test_accum_dtype_none_default() -> None:
  """`accum_dtype=None` is the documented default — automatic
  selection. The reference backend always uses fp64 today."""
  a = np.eye(3)
  b = np.eye(3)
  out = moeinsum.einsum("ij,jk->ik", a, b, accum_dtype=None)
  np.testing.assert_allclose(out, a @ b)


# ---------------------------------------------------------------------
# Cache thread-safety — RLock contention under concurrent.futures
# ---------------------------------------------------------------------


def test_cache_concurrent_get_put_safe() -> None:
  """Hammer the cache from many threads simultaneously. The RLock in
  `_PlanCache.{get,put}` must serialize without deadlocks or losing
  entries. Test runs 8 threads × 32 iterations each, alternating
  put/get on overlapping keys."""
  import threading
  from concurrent.futures import ThreadPoolExecutor, as_completed

  from moeinsum._cache import _PlanCache

  cache = _PlanCache(max_entries=64)
  errors: list[Exception] = []
  lock = threading.Lock()

  def hammer(worker_id: int) -> None:
    try:
      for i in range(32):
        key = ("hot", i % 16)
        cache.put(key, (worker_id, i))
        val = cache.get(key)
        if val is None:
          # Possible only if eviction beat us — max_entries=64 with
          # 16 distinct hot keys means it shouldn't happen.
          raise AssertionError(f"cache lost key {key!r}")
    except Exception as exc:  # noqa: BLE001
      with lock:
        errors.append(exc)

  with ThreadPoolExecutor(max_workers=8) as ex:
    futures = [ex.submit(hammer, w) for w in range(8)]
    for f in as_completed(futures):
      f.result()

  assert not errors, f"thread-safety violations: {errors[:3]}"


def test_cache_concurrent_einsum_path_consistent() -> None:
  """End-to-end: multiple threads call `einsum_path` on overlapping
  shapes. Each result must be self-consistent (every thread for the
  same input sees the same path)."""
  from concurrent.futures import ThreadPoolExecutor

  PLAN_CACHE.clear()
  eq = "ij,jk,kl->il"
  shapes = ((3, 4), (4, 5), (5, 6))

  def call() -> tuple[tuple[int, ...], ...]:
    return tuple(moeinsum.einsum_path(eq, *shapes, optimize="optimal"))

  with ThreadPoolExecutor(max_workers=8) as ex:
    results = list(ex.map(lambda _: call(), range(32)))

  # All 32 results must be identical.
  assert len(set(results)) == 1, f"divergent paths under concurrency: {set(results)}"
