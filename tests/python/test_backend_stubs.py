"""Backend seam tests for the shipped MAX and native paths."""

from __future__ import annotations

import moeinsum
import numpy as np
import pytest
from moeinsum._max_backend import MaxGraphBackend, classify_pair, lowering_spec


def _kinds(spec: dict[str, object]) -> list[str]:
  ops = spec["ops"]
  assert isinstance(ops, list)
  return [str(op["kind"]) for op in ops]


def test_classify_pair_matmul() -> None:
  c = classify_pair("ij", "jk", "ik")
  assert c.batch == ()
  assert c.contract == ("j",)
  assert c.free_lhs == ("i",)
  assert c.free_rhs == ("k",)


def test_classify_pair_batched_matmul() -> None:
  c = classify_pair("bij", "bjk", "bik")
  assert c.batch == ("b",)
  assert c.contract == ("j",)
  assert c.free_lhs == ("i",)
  assert c.free_rhs == ("k",)


def test_classify_pair_frobenius() -> None:
  c = classify_pair("ij", "ij", "")
  assert c.batch == ()
  assert c.contract == ("i", "j")
  assert c.free_lhs == ()
  assert c.free_rhs == ()


def test_classify_pair_outer_product() -> None:
  c = classify_pair("i", "j", "ij")
  assert c.batch == ()
  assert c.contract == ()
  assert c.free_lhs == ("i",)
  assert c.free_rhs == ("j",)


def test_classify_pair_label_order_follows_lhs() -> None:
  c = classify_pair("ab", "ba", "ab")
  assert c.batch == ("a", "b")
  assert c.contract == ()
  assert c.free_lhs == ()
  assert c.free_rhs == ()


def test_lowering_spec_matmul_one_step() -> None:
  spec = lowering_spec("ij,jk->ik", [(2, 3), (3, 4)], [(0, 1)])
  ops = spec["ops"]
  assert isinstance(ops, list)
  assert len(ops) == 1
  op = ops[0]
  assert op["kind"] == "matmul"
  assert op["lhs_labels"] == "ij"
  assert op["rhs_labels"] == "jk"
  assert op["out_labels"] == "ik"
  assert op["batch"] == []
  assert op["contract"] == ["j"]


def test_lowering_spec_three_operand_chain() -> None:
  spec = lowering_spec(
    "ij,jk,kl->il",
    [(100, 1), (1, 100_000), (100_000, 1)],
    [(1, 2), (0, 1)],
  )
  assert _kinds(spec) == ["matmul", "matmul"]
  ops = spec["ops"]
  assert isinstance(ops, list)
  assert ops[0]["out_labels"] == "jl"
  assert ops[1]["out_labels"] == "il"


def test_lowering_spec_unary_trace() -> None:
  spec = lowering_spec("ii->", [(4, 4)], [(0,)])
  assert _kinds(spec) == ["diagonal", "reduce_sum"]


def test_lowering_spec_unary_full_sum() -> None:
  spec = lowering_spec("ij->", [(3, 5)], [(0,)])
  assert _kinds(spec) == ["reduce_sum"]
  ops = spec["ops"]
  assert isinstance(ops, list)
  assert ops[0]["dst_labels"] == ""


def test_lowering_spec_trailing_transpose_when_axis_order_differs() -> None:
  spec = lowering_spec("ji,jk->ki", [(3, 2), (3, 5)], [(0, 1)])
  assert _kinds(spec) == ["matmul"]
  ops = spec["ops"]
  assert isinstance(ops, list)
  assert ops[0]["out_labels"] == "ki"
  assert ops[0]["swapped_operands"] is True
  assert ops[0]["needs_output_transpose"] is False


def test_lowering_spec_expands_ellipsis() -> None:
  spec = lowering_spec("...ij,...jk->...ik", [(2, 3, 4, 5), (3, 5, 6)], [(0, 1)])
  assert _kinds(spec) == ["matmul"]
  assert spec["result_shape"] == [2, 3, 4, 6]


def test_lowering_spec_unary_identity_emits_no_ops() -> None:
  spec = lowering_spec("ij->ij", [(3, 5)], [(0,)])
  assert spec["ops"] == []


@pytest.mark.parametrize(
  ("eq", "shapes"),
  [
    ("ij,jk->ik", [(3, 4), (4, 5)]),
    ("ij->ji", [(3, 4)]),
    ("ij->", [(3, 4)]),
    ("ii->", [(4, 4)]),
    ("iij,jk->ik", [(3, 3, 4), (4, 5)]),
    ("ij,jk,kl->il", [(3, 4), (4, 5), (5, 6)]),
    ("cij,cjk->cik", [(1, 3, 4), (5, 4, 6)]),
  ],
)
def test_einsum_native_backend_matches_numpy(eq: str, shapes: list[tuple[int, ...]]) -> None:
  rng = np.random.default_rng(0)
  arrays = [rng.standard_normal(shape).astype(np.float64) for shape in shapes]

  expected = np.einsum(eq, *arrays, optimize=True)
  actual = moeinsum.einsum(eq, *arrays, backend="native")

  np.testing.assert_allclose(np.asarray(actual), expected, atol=1e-10, rtol=1e-10)


def test_einsum_native_backend_respects_explicit_path() -> None:
  rng = np.random.default_rng(1)
  arrays = [
    rng.standard_normal((2, 3)),
    rng.standard_normal((3, 4)),
    rng.standard_normal((4, 5)),
  ]

  expected = np.einsum("ij,jk,kl->il", *arrays, optimize=True)
  actual = moeinsum.einsum(
    "ij,jk,kl->il",
    *arrays,
    backend="native",
    optimize=[(1, 2), (0, 1)],
  )

  np.testing.assert_allclose(np.asarray(actual), expected, atol=1e-10, rtol=1e-10)


def test_einsum_max_backend_matmul() -> None:
  a = np.arange(12, dtype=np.float32).reshape(3, 4)
  b = np.arange(20, dtype=np.float32).reshape(4, 5)
  actual = moeinsum.einsum("ij,jk->ik", a, b, backend="max:cpu")
  np.testing.assert_allclose(np.asarray(actual), a @ b, atol=1e-5, rtol=1e-5)


def test_einsum_max_backend_repeated_labels() -> None:
  a = np.arange(9, dtype=np.float32).reshape(3, 3)
  np.testing.assert_allclose(
    np.asarray(moeinsum.einsum("ii->", a, backend="max:cpu")),
    np.asarray(np.einsum("ii->", a)),
    atol=1e-5,
    rtol=1e-5,
  )

  b = np.arange(18, dtype=np.float32).reshape(3, 3, 2)
  c = np.arange(10, dtype=np.float32).reshape(2, 5)
  np.testing.assert_allclose(
    np.asarray(moeinsum.einsum("iij,jk->ik", b, c, backend="max:cpu")),
    np.einsum("iij,jk->ik", b, c),
    atol=1e-5,
    rtol=1e-5,
  )


def test_max_graph_backend_execute_matches_numpy() -> None:
  a = np.arange(12, dtype=np.float32).reshape(3, 4)
  b = np.arange(20, dtype=np.float32).reshape(4, 5)
  path = moeinsum.einsum_path("ij,jk->ik", a.shape, b.shape)

  actual = MaxGraphBackend().execute("ij,jk->ik", [a.shape, b.shape], path, [a, b])

  np.testing.assert_allclose(np.asarray(actual), a @ b, atol=1e-5, rtol=1e-5)


def test_einsum_unknown_backend_value_error() -> None:
  with pytest.raises(ValueError, match="unknown backend"):
    moeinsum.einsum("ij,jk->ik", np.eye(3), np.eye(3), backend="quantum")

  with pytest.raises(ValueError, match="unknown backend"):
    moeinsum.einsum("ij,jk->ik", np.eye(3), np.eye(3), backend="max_graph")
