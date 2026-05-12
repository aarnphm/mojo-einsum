"""Executable MAX backend parity tests.

These are intentionally small. The Python MAX Graph interop path pays
compile time per shape, so cache coverage stays explicit while the public
`backend="max:cpu"` path exercises the native Mojo MAX backend.
"""

from __future__ import annotations

import moeinsum
import numpy as np
import pytest
from moeinsum import _interop_max
from moeinsum._interop_max import MaxGraphBackend


def _execute_cached_graph(
  eq: str,
  operands: list[np.ndarray],
  *,
  accum_dtype: np.dtype | None = None,
) -> np.ndarray:
  path = moeinsum.einsum_path(eq, *[operand.shape for operand in operands])
  return _interop_max._execute_max_graph(eq, operands, path, "max:cpu", accum_dtype=accum_dtype)


@pytest.mark.parametrize(
  ("eq", "shapes"),
  [
    ("ij,jk->ik", [(3, 4), (4, 5)]),
    ("bij,bjk->bik", [(2, 3, 4), (2, 4, 5)]),
    ("ij,kj->ik", [(3, 4), (5, 4)]),
    ("ij,jk,kl->il", [(3, 4), (4, 5), (5, 6)]),
    ("ij,ij->", [(3, 4), (3, 4)]),
    ("ij->i", [(3, 4)]),
    # Empty K (no contracted dims) -> outer product via degenerate matmul.
    # `_lower_pair` collapses k=1 via `_product([]) == 1`, reshape gives
    # (M, 1) x (1, N) -> (M, N). If the special-case path ever regresses,
    # this is the first thing that breaks.
    ("i,j->ij", [(3,), (4,)]),
    # Pure unary transpose: no reduce, no permute axis dropped, just a
    # permutation. Goes through `_reduce_out_labels` -> noop, then
    # `_permute_if_needed` flips axes. Untested before this row.
    ("ij->ji", [(3, 4)]),
    # Size-1 broadcast on a batch label: (1, 3, 4) cij vs (5, 4, 6) cjk
    # resolves `c` to 5. `_lower_pair` emits `ops.broadcast_to` before
    # the matmul reshape; without it the reshape to `[5, m, k]` would
    # fail because the permuted lhs still has a batch extent of 1.
    ("cij,cjk->cik", [(1, 3, 4), (5, 4, 6)]),
    # Size-1 broadcast on the contracted label: a[3,1] vs b[4,5] -> j
    # resolves to 4. The contraction sum collapses to a[i,0]*sum_j b[j,k].
    ("ij,jk->ik", [(3, 1), (4, 5)]),
    # Ellipsis expansion is shape-dependent, so both public MAX CPU and
    # graph spec lowering route through the native expanded parser.
    ("...ij,jk->...ik", [(2, 3, 4), (4, 5)]),
    ("...ij,...jk->...ik", [(2, 3, 4, 5), (3, 5, 6)]),
    ("ij...,jk...->ik...", [(2, 3, 5, 1), (3, 4, 5, 1)]),
    ("...ij,...jk", [(7, 2, 3), (7, 3, 4)]),
  ],
)
def test_max_cpu_matches_numpy(eq: str, shapes: list[tuple[int, ...]]) -> None:
  rng = np.random.default_rng(0)
  operands = [rng.standard_normal(shape).astype(np.float32) for shape in shapes]

  actual = moeinsum.einsum(eq, *operands, backend="max:cpu")
  expected = np.einsum(eq, *operands, optimize=True)

  np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_max_graph_backend_execute_matches_numpy() -> None:
  a = np.arange(12, dtype=np.float32).reshape(3, 4)
  b = np.arange(20, dtype=np.float32).reshape(4, 5)
  path = moeinsum.einsum_path("ij,jk->ik", a.shape, b.shape)

  actual = MaxGraphBackend().execute("ij,jk->ik", [a.shape, b.shape], path, [a, b])

  np.testing.assert_allclose(actual, a @ b, atol=1e-5, rtol=1e-5)


def test_public_max_cpu_bypasses_python_graph_cache() -> None:
  _interop_max._MODEL_CACHE.clear()
  a = np.arange(12, dtype=np.float32).reshape(3, 4)
  b = np.arange(20, dtype=np.float32).reshape(4, 5)

  actual = moeinsum.einsum("ij,jk->ik", a, b, backend="max:cpu")

  np.testing.assert_allclose(actual, a @ b, atol=1e-5, rtol=1e-5)
  assert len(_interop_max._MODEL_CACHE) == 0


def test_max_backend_model_cache_reuses_compiled_graph() -> None:
  """Identical (eq, shapes, dtype, path, backend) must reuse one compiled graph.

  The §4 perf ratio (`ours / raw <= 1.5x`) depends entirely on this cache hitting
  on call 2+. If a careless edit drops a hashable field from the cache key, or
  worse rebuilds the key as non-hashable, the ratio degrades silently and you
  only notice when a benchmark reviewer asks why the headline number moved.
  Asserting on cache length here catches the breakage the day the key drifts.
  """
  _interop_max._MODEL_CACHE.clear()
  a = np.arange(12, dtype=np.float32).reshape(3, 4)
  b = np.arange(20, dtype=np.float32).reshape(4, 5)

  before = len(_interop_max._MODEL_CACHE)
  _execute_cached_graph("ij,jk->ik", [a, b])
  after_first = len(_interop_max._MODEL_CACHE)
  _execute_cached_graph("ij,jk->ik", [a, b])
  after_second = len(_interop_max._MODEL_CACHE)

  assert after_first == before + 1, f"first call should add exactly one cache entry, grew by {after_first - before}"
  assert after_second == after_first, (
    f"second call with identical signature should hit the cache; "
    f"grew from {after_first} to {after_second} — key has drifted"
  )


def test_max_backend_model_cache_keys_on_dtype() -> None:
  """dtype must be part of the cache key: an fp32-compiled graph won't
  accept fp64 inputs, so the key has to discriminate. If it doesn't, the
  second call either reuses the wrong model (wrong results) or crashes at
  execute time — both worse than a clean miss."""
  _interop_max._MODEL_CACHE.clear()
  a32 = np.arange(12, dtype=np.float32).reshape(3, 4)
  b32 = np.arange(20, dtype=np.float32).reshape(4, 5)
  a64 = a32.astype(np.float64)
  b64 = b32.astype(np.float64)

  before = len(_interop_max._MODEL_CACHE)
  _execute_cached_graph("ij,jk->ik", [a32, b32])
  _execute_cached_graph("ij,jk->ik", [a64, b64])
  after = len(_interop_max._MODEL_CACHE)
  assert after - before == 2, (
    f"dtype change should produce a fresh compile (cache grows by 2); grew by {after - before}"
  )


def test_max_backend_model_cache_keys_on_accum_dtype() -> None:
  _interop_max._MODEL_CACHE.clear()
  a = np.arange(12, dtype=np.float32).reshape(3, 4)
  b = np.arange(20, dtype=np.float32).reshape(4, 5)

  before = len(_interop_max._MODEL_CACHE)
  _execute_cached_graph("ij,jk->ik", [a, b], accum_dtype=np.float32)
  _execute_cached_graph("ij,jk->ik", [a, b], accum_dtype=np.float64)
  after = len(_interop_max._MODEL_CACHE)
  assert after - before == 2, (
    f"accum_dtype change should produce a fresh compile (cache grows by 2); grew by {after - before}"
  )


def test_max_backend_rejects_low_precision_accum_dtype() -> None:
  a = np.arange(12, dtype=np.float32).reshape(3, 4)
  b = np.arange(20, dtype=np.float32).reshape(4, 5)

  with pytest.raises(NotImplementedError, match="accum_dtype"):
    moeinsum.einsum("ij,jk->ik", a, b, backend="max:cpu", accum_dtype=np.float16)


def test_max_backend_model_cache_keys_on_shape() -> None:
  """Same equation, different shapes -> different compiled graphs.
  MAX graphs are shape-static (TensorType pins concrete dims), so reusing
  the (3,4)x(4,5) compile against (3,5)x(5,7) inputs would fail at execute.
  """
  _interop_max._MODEL_CACHE.clear()
  rng = np.random.default_rng(0)
  a1 = rng.standard_normal((3, 4)).astype(np.float32)
  b1 = rng.standard_normal((4, 5)).astype(np.float32)
  a2 = rng.standard_normal((3, 5)).astype(np.float32)
  b2 = rng.standard_normal((5, 7)).astype(np.float32)

  before = len(_interop_max._MODEL_CACHE)
  _execute_cached_graph("ij,jk->ik", [a1, b1])
  _execute_cached_graph("ij,jk->ik", [a2, b2])
  after = len(_interop_max._MODEL_CACHE)
  assert after - before == 2, (
    f"shape change should produce a fresh compile (cache grows by 2); grew by {after - before}"
  )


def test_max_backend_model_cache_lru_evicts_at_max() -> None:
  """`_MODEL_CACHE` is LRU-bounded - a long-running server compiling new
  signatures every call must not leak. Squeeze the cap, fill past it,
  and check the oldest entry is the one that got dropped (MRU survives).
  Restore the cap afterwards so the rest of the suite uses the real bound.
  """
  _interop_max._MODEL_CACHE.clear()
  saved_cap = _interop_max._MODEL_CACHE_MAX
  _interop_max._MODEL_CACHE_MAX = 3
  try:
    rng = np.random.default_rng(0)
    shapes = [(3, 4), (3, 5), (3, 6), (3, 7)]  # 4 distinct compiles, cap=3
    for cols in (s[1] for s in shapes):
      a = rng.standard_normal((3, cols)).astype(np.float32)
      b = rng.standard_normal((cols, 2)).astype(np.float32)
      _execute_cached_graph("ij,jk->ik", [a, b])

    assert len(_interop_max._MODEL_CACHE) == 3, f"LRU should cap at 3, got {len(_interop_max._MODEL_CACHE)}"
    # Oldest key was the (3,4)x(4,2) compile - it should have been
    # evicted; the remaining three are (3,5), (3,6), (3,7).
    surviving_shapes = {key[1] for key in _interop_max._MODEL_CACHE}
    assert ((3, 4), (4, 2)) not in surviving_shapes, "oldest entry should have been evicted"
  finally:
    _interop_max._MODEL_CACHE_MAX = saved_cap
    _interop_max._MODEL_CACHE.clear()


def test_max_backend_model_cache_lru_promotes_on_hit() -> None:
  """A cache hit moves the entry to MRU. After hitting an old entry, a
  subsequent eviction should drop the previously second-oldest, not the
  freshly-promoted one."""
  _interop_max._MODEL_CACHE.clear()
  saved_cap = _interop_max._MODEL_CACHE_MAX
  _interop_max._MODEL_CACHE_MAX = 3
  try:
    rng = np.random.default_rng(0)
    # Fill at cap.
    pairs = [(3, 4), (3, 5), (3, 6)]
    for cols in pairs:
      a = rng.standard_normal((3, cols[1])).astype(np.float32)
      b = rng.standard_normal((cols[1], 2)).astype(np.float32)
      _execute_cached_graph("ij,jk->ik", [a, b])
    assert len(_interop_max._MODEL_CACHE) == 3

    # Hit the oldest entry - it should move to MRU.
    a = rng.standard_normal((3, 4)).astype(np.float32)
    b = rng.standard_normal((4, 2)).astype(np.float32)
    _execute_cached_graph("ij,jk->ik", [a, b])
    assert len(_interop_max._MODEL_CACHE) == 3, "hit should not grow cache"

    # Insert a 4th distinct signature - the (3,5) entry should evict,
    # not (3,4) which we just promoted.
    a = rng.standard_normal((3, 7)).astype(np.float32)
    b = rng.standard_normal((7, 2)).astype(np.float32)
    _execute_cached_graph("ij,jk->ik", [a, b])
    surviving = {key[1] for key in _interop_max._MODEL_CACHE}
    assert ((3, 4), (4, 2)) in surviving, "freshly-promoted entry must survive"
    assert ((3, 5), (5, 2)) not in surviving, "second-oldest should have evicted"
  finally:
    _interop_max._MODEL_CACHE_MAX = saved_cap
    _interop_max._MODEL_CACHE.clear()
