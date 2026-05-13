"""Public API IR-printing tests."""

from __future__ import annotations

import json

import moeinsum
import numpy as np
import pytest


def test_einsum_ir_true_prints_selected_backend_record(capsys: pytest.CaptureFixture[str]) -> None:
  a = np.arange(6, dtype=np.float64).reshape(2, 3)
  b = np.arange(12, dtype=np.float64).reshape(3, 4)

  out = moeinsum.einsum("ij,jk->ik", a, b, backend="native", ir=True)
  rec = json.loads(capsys.readouterr().out)

  np.testing.assert_allclose(out, np.einsum("ij,jk->ik", a, b))
  assert rec["equation"] == "ij,jk->ik"
  assert rec["shapes"] == [[2, 3], [3, 4]]
  assert rec["optimize"] == "auto"
  assert rec["path"] == [[0, 1]]
  assert rec["parser_ir"]["has_explicit_output"] is True
  assert rec["path_cost"]["total_flops"] == 24
  assert rec["plan_graph_spec"]["ops"][0]["kind"] == "matmul"
  assert set(rec["backends"]) == {"native"}


def test_einsum_ir_true_preserves_explicit_path(capsys: pytest.CaptureFixture[str]) -> None:
  a = np.arange(6, dtype=np.float64).reshape(2, 3)
  b = np.arange(12, dtype=np.float64).reshape(3, 4)
  path = [(0, 1)]

  moeinsum.einsum("ij,jk->ik", a, b, optimize=path, ir=True)
  rec = json.loads(capsys.readouterr().out)

  assert rec["optimize"] == [[0, 1]]
  assert rec["path"] == [[0, 1]]
  assert set(rec["backends"]) == {"reference"}


def test_einsum_ir_rejects_non_bool() -> None:
  a = np.eye(2)

  with np.testing.assert_raises_regex(TypeError, "ir must be bool"):
    moeinsum.einsum("ij,jk->ik", a, a, ir="yes")
