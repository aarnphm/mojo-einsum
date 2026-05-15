"""Bench CLI subprocess + JSON-schema tests.

Runs the `moeinsum-bench` entry point as a subprocess and validates
the JSON output shape. Catches schema drift that integration tests
inside the same process miss (e.g. broken `argparse` choices, broken
script entry, broken JSON dump).

Two invocation paths are validated:
  - `python -m moeinsum._cli.bench ...`
  - `moeinsum-bench ...` (the `[project.scripts]` entry installed by
    `pip install -e .`)

These tests do real subprocess work and add ~3-5s to the suite. If
runtime matters, gate them with `-k 'not bench_cli'`.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def _python() -> str:
  return sys.executable


def _bench_args(equation: str, shapes: list[str], **extra: object) -> list[str]:
  out = [equation, "--shapes", *shapes, "--repeats", "3", "--warmup", "1"]
  for k, v in extra.items():
    flag = "--" + k.replace("_", "-")
    if isinstance(v, bool):
      if v:
        out.append(flag)
    else:
      out.extend([flag, str(v)])
  return out


def _run(*argv: str, timeout: float = 30.0) -> subprocess.CompletedProcess:
  env = os.environ.copy()
  # Carry the repo-local packaged Modular runtime through. The extension's
  # rpath already points here; putting it first defends against a stale
  # shell-level DYLD path into an in-tree Modular build.
  modular_lib = (
    Path(__file__).parent.parent.parent / ".venv" / "lib" / "python3.11" / "site-packages" / "modular" / "lib"
  )
  if modular_lib.is_dir():
    existing = env.get("DYLD_LIBRARY_PATH", "")
    env["DYLD_LIBRARY_PATH"] = f"{modular_lib}:{existing}" if existing else str(modular_lib)
  return subprocess.run(
    list(argv),
    capture_output=True,
    text=True,
    timeout=timeout,
    env=env,
    check=False,
  )


def _parse_json(stdout: str) -> dict[str, object]:
  return json.loads(stdout)


# ---------------------------------------------------------------------
# `python -m moeinsum._cli.bench` entry
# ---------------------------------------------------------------------


def test_module_entry_single_optimizer() -> None:
  args = _bench_args("ij,jk->ik", ["3,4", "4,5"])
  proc = _run(_python(), "-m", "moeinsum._cli.bench", *args)
  assert proc.returncode == 0, f"stderr: {proc.stderr}"
  rec = _parse_json(proc.stdout)
  for key in (
    "equation",
    "shapes",
    "backend",
    "optimize",
    "dtype",
    "input_framework",
    "input_device",
    "repeats",
    "warmup",
    "total_ms_median",
    "total_ms_min",
    "total_ms_max",
    "platform",
    "timestamp",
  ):
    assert key in rec, f"missing schema key {key!r}"
  assert rec["equation"] == "ij,jk->ik"
  assert rec["shapes"] == [[3, 4], [4, 5]]
  assert rec["backend"] == "reference"
  assert rec["input_framework"] == "numpy"
  assert rec["input_device"] == "cpu"
  assert isinstance(rec["total_ms_median"], (int, float))
  assert rec["total_ms_median"] > 0


def test_module_entry_with_include_path() -> None:
  args = _bench_args("ij,jk,kl->il", ["3,4", "4,5", "5,6"], include_path=True)
  proc = _run(_python(), "-m", "moeinsum._cli.bench", *args)
  assert proc.returncode == 0
  rec = _parse_json(proc.stdout)
  assert "path" in rec
  # 3 operands -> 2 pairwise steps.
  assert len(rec["path"]) == 2


def test_module_entry_sweep_optimizers() -> None:
  args = _bench_args("ij,jk,kl->il", ["3,4", "4,5", "5,6"], sweep_optimizers=True)
  proc = _run(_python(), "-m", "moeinsum._cli.bench", *args)
  assert proc.returncode == 0
  rec = _parse_json(proc.stdout)
  assert "results" in rec
  assert "fastest" in rec
  assert "ratios" in rec
  # Every optimizer the planner exposes must have run in sweep mode.
  for opt in (
    "naive",
    "greedy",
    "optimal",
    "auto",
    "random-greedy",
    "branch-all",
    "branch-2",
    "branch-1",
  ):
    assert opt in rec["results"]
    assert rec["ratios"][opt] >= 1.0  # fastest has ratio == 1.0


def test_module_entry_vs_numpy() -> None:
  args = _bench_args("ij,jk->ik", ["8,8", "8,8"], vs_numpy=True)
  proc = _run(_python(), "-m", "moeinsum._cli.bench", *args)
  assert proc.returncode == 0
  rec = _parse_json(proc.stdout)
  for key in ("numpy_ms_median", "numpy_ms_min", "numpy_ms_max", "vs_numpy_ratio"):
    assert key in rec
  assert rec["vs_numpy_ratio"] > 0


def test_module_entry_compare_engines() -> None:
  args = _bench_args("ij,jk->ik", ["4,4", "4,4"], compare_engines="numpy,opt_einsum,mlx")
  proc = _run(_python(), "-m", "moeinsum._cli.bench", *args)
  assert proc.returncode == 0, f"stderr: {proc.stderr}"
  rec = _parse_json(proc.stdout)
  comparisons = rec["comparisons"]
  assert comparisons["moeinsum"]["status"] == "ok"
  assert comparisons["numpy"]["status"] == "ok"
  assert comparisons["numpy"]["ms_median"] > 0
  assert "speedup_vs_moeinsum" in comparisons["numpy"]
  assert comparisons["opt_einsum"]["status"] in {"ok", "skipped", "error"}
  assert comparisons["mlx"]["status"] in {"ok", "skipped", "error"}
  assert rec["comparison_fastest"] in rec["comparison_ratios"]


def test_module_entry_cuda_requires_torch_input_framework() -> None:
  args = _bench_args("ij,jk->ik", ["4,4", "4,4"], input_device="cuda")
  proc = _run(_python(), "-m", "moeinsum._cli.bench", *args)
  assert proc.returncode != 0
  assert "--input-device cuda requires --input-framework torch" in proc.stderr


def test_module_entry_progress_stays_on_stderr() -> None:
  pytest.importorskip("tqdm")
  args = _bench_args("ij,jk->ik", ["4,4", "4,4"], progress=True)
  proc = _run(_python(), "-m", "moeinsum._cli.bench", *args)
  assert proc.returncode == 0, f"stderr: {proc.stderr}"
  rec = _parse_json(proc.stdout)
  assert rec["equation"] == "ij,jk->ik"
  assert "bench moeinsum" in proc.stderr


def test_module_entry_cache_bench() -> None:
  """`--cache-bench` clears PLAN_CACHE, times one cold call, then times
  --repeats hot calls; result must carry cold_ms, hot_ms_median, and
  cache_speedup_ratio (a positive float). The actual ratio is too
  noisy to assert tightly - we just check the schema and that the
  ratio is finite."""
  args = _bench_args("ij,jk,kl->il", ["3,4", "4,5", "5,6"], cache_bench=True)
  proc = _run(_python(), "-m", "moeinsum._cli.bench", *args)
  assert proc.returncode == 0, f"stderr: {proc.stderr}"
  rec = _parse_json(proc.stdout)
  for key in ("cold_ms", "hot_ms_median", "hot_ms_min", "hot_ms_max", "cache_speedup_ratio"):
    assert key in rec, f"missing schema key {key!r}"
  assert rec["cold_ms"] > 0
  assert rec["hot_ms_median"] > 0
  assert rec["cache_speedup_ratio"] > 0


def test_module_entry_records_modular_debug_shortcuts() -> None:
  args = _bench_args(
    "ij,jk->ik",
    ["3,4", "4,5"],
    modular_debug="source-tracebacks",
    max_ir_output_dir="/tmp/moeinsum-ir",
    max_op_log_level="trace",
  )
  proc = _run(_python(), "-m", "moeinsum._cli.bench", *args)
  assert proc.returncode == 0, f"stderr: {proc.stderr}"
  rec = _parse_json(proc.stdout)
  assert "source-tracebacks" in rec["modular_debug"]
  assert "ir-output-dir=/tmp/moeinsum-ir" in rec["modular_debug"]
  assert "op-log-level=trace" in rec["modular_debug"]
  assert rec["max_ir_output_dir"] == "/tmp/moeinsum-ir"


def test_module_entry_random_greedy_n() -> None:
  args = _bench_args("ij,jk,kl->il", ["3,4", "4,5", "5,6"], optimize="random-greedy-16")
  proc = _run(_python(), "-m", "moeinsum._cli.bench", *args)
  assert proc.returncode == 0
  rec = _parse_json(proc.stdout)
  assert rec["optimize"] == "random-greedy-16"


def test_module_entry_explicit_path_rejected_via_cli() -> None:
  """The CLI's `--optimize` takes a string, so explicit-path callers
  must use the Python API. Verify the CLI rejects an obvious
  attempt with an unknown-optimize error."""
  args = _bench_args("ij,jk->ik", ["3,4", "4,5"], optimize="[(0,1)]")
  proc = _run(_python(), "-m", "moeinsum._cli.bench", *args)
  assert proc.returncode != 0
  assert "unknown optimize" in proc.stderr or "Traceback" in proc.stderr


def test_module_entry_invalid_equation() -> None:
  args = _bench_args("i$j,jk->ik", ["3,4", "4,5"])
  proc = _run(_python(), "-m", "moeinsum._cli.bench", *args)
  assert proc.returncode != 0


def test_module_entry_dtype_bfloat16_routes_via_ml_dtypes() -> None:
  """`--dtype bfloat16` should accept ml_dtypes-backed operands and
  record dtype="bfloat16" in the JSON. The reference backend handles
  the bf16 path via upcast-on-accumulate, so the run succeeds even
  without MAX installed."""
  pytest.importorskip("ml_dtypes")
  args = _bench_args("ij,jk->ik", ["4,4", "4,4"], dtype="bfloat16")
  proc = _run(_python(), "-m", "moeinsum._cli.bench", *args)
  assert proc.returncode == 0, f"stderr: {proc.stderr}"
  rec = _parse_json(proc.stdout)
  assert rec["dtype"] == "bfloat16"
  assert rec["total_ms_median"] > 0


def test_module_entry_bfloat16_skips_numpy_comparison() -> None:
  """bf16 + `--compare-engines numpy` must surface a clear skip
  reason instead of letting numpy silently upgrade to fp32 and report
  a misleading ratio."""
  pytest.importorskip("ml_dtypes")
  args = _bench_args("ij,jk->ik", ["4,4", "4,4"], dtype="bfloat16", compare_engines="numpy")
  proc = _run(_python(), "-m", "moeinsum._cli.bench", *args)
  assert proc.returncode == 0, f"stderr: {proc.stderr}"
  rec = _parse_json(proc.stdout)
  numpy_rec = rec["comparisons"]["numpy"]
  assert numpy_rec["status"] == "skipped"
  assert "bf16" in numpy_rec["reason"].lower() or "bfloat16" in numpy_rec["reason"].lower()


# ---------------------------------------------------------------------
# `moeinsum-bench` script entry (from [project.scripts])
# ---------------------------------------------------------------------


@pytest.mark.skipif(
  shutil.which("moeinsum-bench") is None and not (Path(_python()).parent / "moeinsum-bench").is_file(),
  reason="moeinsum-bench console script not on PATH",
)
def test_console_script_entry() -> None:
  """The `[project.scripts] moeinsum-bench = "moeinsum._cli.bench:main"`
  entry must produce the same JSON shape as `python -m`."""
  script = shutil.which("moeinsum-bench") or str(Path(_python()).parent / "moeinsum-bench")
  args = _bench_args("ij,jk->ik", ["3,4", "4,5"])
  proc = _run(script, *args)
  assert proc.returncode == 0
  rec = _parse_json(proc.stdout)
  assert rec["equation"] == "ij,jk->ik"
  assert rec["total_ms_median"] > 0
