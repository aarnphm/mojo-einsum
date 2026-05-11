"""Executable MAX Graph backend parity tests.

These are intentionally small. The first call per shape pays MAX graph
compile time, so this file pins coverage breadth without turning the
default test run into a coffee break.
"""

from __future__ import annotations

import moeinsum
import numpy as np
import pytest
from moeinsum._max_graph import MaxGraphBackend, is_loadable

pytestmark = pytest.mark.skipif(
  not is_loadable(),
  reason="max.graph not installed or ABI-incompatible with moeinsum._native in this env",
)


@pytest.mark.parametrize(
  ("eq", "shapes"),
  [
    ("ij,jk->ik", [(3, 4), (4, 5)]),
    ("bij,bjk->bik", [(2, 3, 4), (2, 4, 5)]),
    ("ij,kj->ik", [(3, 4), (5, 4)]),
    ("ij,jk,kl->il", [(3, 4), (4, 5), (5, 6)]),
    ("ij,ij->", [(3, 4), (3, 4)]),
    ("ij->i", [(3, 4)]),
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


def test_max_backend_model_cache_reuses_compiled_graph() -> None:
  """Identical (eq, shapes, dtype, path, backend) must reuse one compiled graph.

  The §4 perf ratio (`ours / raw <= 1.5x`) depends entirely on this cache hitting
  on call 2+. If a careless edit drops a hashable field from the cache key, or
  worse rebuilds the key as non-hashable, the ratio degrades silently and you
  only notice when a benchmark reviewer asks why the headline number moved.
  Asserting on cache length here catches the breakage the day the key drifts.
  """
  from moeinsum import _max_backend  # noqa: PLC0415

  a = np.arange(12, dtype=np.float32).reshape(3, 4)
  b = np.arange(20, dtype=np.float32).reshape(4, 5)

  before = len(_max_backend._MODEL_CACHE)
  moeinsum.einsum("ij,jk->ik", a, b, backend="max:cpu")
  after_first = len(_max_backend._MODEL_CACHE)
  moeinsum.einsum("ij,jk->ik", a, b, backend="max:cpu")
  after_second = len(_max_backend._MODEL_CACHE)

  assert after_first == before + 1, (
    f"first call should add exactly one cache entry, grew by {after_first - before}"
  )
  assert after_second == after_first, (
    f"second call with identical signature should hit the cache; "
    f"grew from {after_first} to {after_second} — key has drifted"
  )


def test_max_backend_model_cache_keys_on_dtype() -> None:
  """dtype must be part of the cache key: an fp32-compiled graph won't
  accept fp64 inputs, so the key has to discriminate. If it doesn't, the
  second call either reuses the wrong model (wrong results) or crashes at
  execute time — both worse than a clean miss."""
  from moeinsum import _max_backend  # noqa: PLC0415

  a32 = np.arange(12, dtype=np.float32).reshape(3, 4)
  b32 = np.arange(20, dtype=np.float32).reshape(4, 5)
  a64 = a32.astype(np.float64)
  b64 = b32.astype(np.float64)

  before = len(_max_backend._MODEL_CACHE)
  moeinsum.einsum("ij,jk->ik", a32, b32, backend="max:cpu")
  moeinsum.einsum("ij,jk->ik", a64, b64, backend="max:cpu")
  after = len(_max_backend._MODEL_CACHE)
  assert after - before == 2, (
    f"dtype change should produce a fresh compile (cache grows by 2); grew by {after - before}"
  )


def test_max_backend_model_cache_keys_on_shape() -> None:
  """Same equation, different shapes -> different compiled graphs.
  MAX graphs are shape-static (TensorType pins concrete dims), so reusing
  the (3,4)x(4,5) compile against (3,5)x(5,7) inputs would fail at execute.
  """
  from moeinsum import _max_backend  # noqa: PLC0415

  rng = np.random.default_rng(0)
  a1 = rng.standard_normal((3, 4)).astype(np.float32)
  b1 = rng.standard_normal((4, 5)).astype(np.float32)
  a2 = rng.standard_normal((3, 5)).astype(np.float32)
  b2 = rng.standard_normal((5, 7)).astype(np.float32)

  before = len(_max_backend._MODEL_CACHE)
  moeinsum.einsum("ij,jk->ik", a1, b1, backend="max:cpu")
  moeinsum.einsum("ij,jk->ik", a2, b2, backend="max:cpu")
  after = len(_max_backend._MODEL_CACHE)
  assert after - before == 2, (
    f"shape change should produce a fresh compile (cache grows by 2); grew by {after - before}"
  )
