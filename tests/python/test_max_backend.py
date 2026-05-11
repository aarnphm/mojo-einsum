"""Executable MAX Graph backend parity tests.

These are intentionally small. The first call per shape pays MAX graph
compile time, so this file pins coverage breadth without turning the
default test run into a coffee break.
"""

from __future__ import annotations

import moeinsum
import numpy as np
import pytest
from moeinsum._max_graph import MaxGraphBackend, is_available

pytestmark = pytest.mark.skipif(not is_available(), reason="max.graph not installed")


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
