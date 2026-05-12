"""Lowering-inspection CLI tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys


def _python() -> str:
  return sys.executable


def _run_lowering(*args: str) -> subprocess.CompletedProcess[str]:
  env = os.environ.copy()
  env.pop("__PYVENV_LAUNCHER__", None)
  return subprocess.run(
    [_python(), "-m", "moeinsum.lowering", *args],
    capture_output=True,
    text=True,
    env=env,
    timeout=30,
    check=False,
  )


def test_lowering_cli_reports_max_matmul_shape() -> None:
  proc = _run_lowering("ij,jk->ik", "--shapes", "3,4", "4,5", "--backend", "max:cpu")
  assert proc.returncode == 0, proc.stderr
  rec = json.loads(proc.stdout)

  assert rec["path"] == [[0, 1]]
  max_rec = rec["backends"]["max:cpu"]
  assert max_rec["status"] == "ok"
  assert max_rec["device_policy"] == "CPU()"
  op = max_rec["ops"][0]
  assert op["kind"] == "matmul"
  assert op["target_op"] == "max.graph.ops.matmul"
  assert op["bmm_shape"]["lhs"] == [3, 4]
  assert op["bmm_shape"]["rhs"] == [4, 5]
  assert op["bmm_shape"]["out"] == [3, 5]


def test_lowering_cli_reports_max_diagonal_lowering() -> None:
  proc = _run_lowering("ii->", "--shapes", "3,3", "--backend", "max")
  assert proc.returncode == 0, proc.stderr
  rec = json.loads(proc.stdout)

  max_rec = rec["backends"]["max"]
  assert max_rec["status"] == "ok"
  assert max_rec["supported"] is True
  assert [op["kind"] for op in max_rec["ops"]] == ["diagonal", "reduce_sum"]
  assert max_rec["ops"][0]["target_op"] == "max.graph.ops.gather_nd"


def test_lowering_cli_reports_max_operand_swap_to_avoid_transpose() -> None:
  proc = _run_lowering("ji,jk->ki", "--shapes", "3,2", "3,5", "--backend", "max:cpu")
  assert proc.returncode == 0, proc.stderr
  rec = json.loads(proc.stdout)

  op = rec["backends"]["max:cpu"]["ops"][0]
  assert op["swapped_operands"] is True
  assert op["natural_labels"] == "ki"
  assert op["needs_output_transpose"] is False
  assert len(rec["backends"]["max:cpu"]["ops"]) == 1
