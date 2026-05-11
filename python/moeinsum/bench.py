"""Benchmark CLI — `python -m moeinsum.bench`.

Emits per-step timing as JSON. Runs each measurement N times and reports
the median (more robust to GC pauses than mean / min).

Usage:
  python -m moeinsum.bench "ij,jk->ik" --shapes 256,256 256,256
  python -m moeinsum.bench "ij,jk,kl->il" --shapes 64,64 64,64 64,64 \\
      --backend reference --optimize auto --repeats 100

Output format (JSON to stdout):
  {
    "equation": "ij,jk->ik",
    "shapes": [[256, 256], [256, 256]],
    "backend": "reference",
    "optimize": "auto",
    "path": [[0, 1]],
    "total_ms_median": 12.4,
    "total_ms_min": 12.1,
    "total_ms_max": 13.5,
    "repeats": 100,
    "platform": {
      "machine": "arm64",
      "system": "Darwin",
      "release": "25.3.0"
    },
    "timestamp": "2026-05-11T..."
  }
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import sysconfig
import time
from datetime import UTC, datetime
from pathlib import Path

# Set the libpython link before `moeinsum` imports `_native` and triggers
# the mohaus editable rebuild. uv-managed interpreters live outside the
# loader's default search path, so the rebuild fails without this.
if "MOJO_PYTHON_LIBRARY" not in os.environ:
  _libdir = sysconfig.get_config_var("LIBDIR")
  _libname = sysconfig.get_config_var("LDLIBRARY")
  if _libdir and _libname:
    _candidate = Path(_libdir) / _libname
    if _candidate.is_file():
      os.environ["MOJO_PYTHON_LIBRARY"] = str(_candidate)

import numpy as np

from . import einsum, einsum_path


def _parse_shapes(shape_strs: list[str]) -> list[tuple[int, ...]]:
  """`["3,4,5", "5,6"]` → `[(3,4,5), (5,6)]`."""
  out = []
  for s in shape_strs:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
      raise ValueError(f"empty shape: {s!r}")
    out.append(tuple(int(p) for p in parts))
  return out


def _make_operands(shapes: list[tuple[int, ...]], dtype: np.dtype, seed: int) -> list[np.ndarray]:
  rng = np.random.default_rng(seed)
  return [rng.standard_normal(s).astype(dtype) for s in shapes]


def _time_one(
  eq: str,
  operands: list[np.ndarray],
  backend: str,
  optimize: str,
) -> float:
  """Return wall time of one einsum call, in milliseconds."""
  t0 = time.perf_counter()
  einsum(eq, *operands, backend=backend, optimize=optimize)
  t1 = time.perf_counter()
  return (t1 - t0) * 1000.0


def main(argv: list[str] | None = None) -> int:
  p = argparse.ArgumentParser(
    prog="moeinsum-bench",
    description="Benchmark a single einsum equation across N repeats.",
  )
  p.add_argument("equation")
  p.add_argument(
    "--shapes",
    nargs="+",
    required=True,
    help="Per-operand shapes, comma-separated. E.g. --shapes 3,4 4,5",
  )
  p.add_argument(
    "--backend",
    default="reference",
    choices=["reference"],
    help="Execution backend (only 'reference' in v0.1)",
  )
  p.add_argument(
    "--optimize",
    default="auto",
    choices=["naive", "greedy", "optimal", "auto"],
    help="Path optimizer algorithm",
  )
  p.add_argument(
    "--dtype",
    default="float64",
    choices=["float32", "float64"],
    help="Operand dtype",
  )
  p.add_argument("--repeats", type=int, default=11, help="Number of runs")
  p.add_argument("--warmup", type=int, default=2, help="Warmup runs (untimed)")
  p.add_argument("--seed", type=int, default=0, help="Operand-RNG seed")
  p.add_argument(
    "--include-path",
    action="store_true",
    help="Include the planner-chosen contraction path in the output",
  )
  args = p.parse_args(argv)

  shapes = _parse_shapes(args.shapes)
  dtype = np.dtype(args.dtype)
  operands = _make_operands(shapes, dtype, args.seed)

  # Warmup.
  for _ in range(args.warmup):
    _time_one(args.equation, operands, args.backend, args.optimize)

  timings = []
  for _ in range(args.repeats):
    timings.append(_time_one(args.equation, operands, args.backend, args.optimize))

  result = {
    "equation": args.equation,
    "shapes": [list(s) for s in shapes],
    "backend": args.backend,
    "optimize": args.optimize,
    "dtype": args.dtype,
    "repeats": args.repeats,
    "warmup": args.warmup,
    "total_ms_median": statistics.median(timings),
    "total_ms_min": min(timings),
    "total_ms_max": max(timings),
    "platform": {
      "machine": platform.machine(),
      "system": platform.system(),
      "release": platform.release(),
    },
    "timestamp": datetime.now(UTC).isoformat(),
  }
  if args.include_path:
    result["path"] = einsum_path(args.equation, *shapes, optimize=args.optimize)

  json.dump(result, sys.stdout, indent=2)
  sys.stdout.write("\n")
  return 0


if __name__ == "__main__":
  sys.exit(main())
