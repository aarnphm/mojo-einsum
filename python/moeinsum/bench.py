"""Benchmark CLI — `moeinsum-bench` (or `python -m moeinsum.bench`).

Emits per-step timing as JSON. Runs each measurement N times, reports
the median (robust to GC pauses), min, and max.

Single-optimizer mode (default):
  moeinsum-bench "ij,jk->ik" --shapes 256,256 256,256
  moeinsum-bench "ij,jk,kl->il" --shapes 64,64 64,64 64,64 \\
      --optimize auto --repeats 100

Optimizer-sweep mode (`--sweep-optimizers`): runs every named optimizer
in `_OPTIMIZE` against the same operands and emits a ratios table
(median time vs. fastest):
  moeinsum-bench "ij,jk,kl,lm->im" --shapes 8,4 4,16 16,2 2,32 \\
      --sweep-optimizers

Output JSON (single-optimizer):
  {"equation": ..., "shapes": ..., "backend": "reference",
   "optimize": "auto", "path": [[1, 2], [0, 1]],
   "total_ms_median": 12.4, "total_ms_min": 12.1, "total_ms_max": 13.5,
   "repeats": 100, "platform": {...}, "timestamp": "..."}

Output JSON (sweep mode):
  {"equation": ..., "shapes": ..., "backend": "reference",
   "results": {"naive": {"ms_median": ..., "path": ...},
               "greedy": {...}, ...},
   "fastest": "greedy",
   "ratios": {"naive": 2.4, "greedy": 1.0, ...}}
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
    help=("Path optimizer: naive / greedy / optimal / auto / random-greedy / random-greedy-N / branch-{all,2,1}"),
  )
  p.add_argument(
    "--sweep-optimizers",
    action="store_true",
    help=("Run every standard optimizer and report median-time ratios instead of a single measurement"),
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
  p.add_argument(
    "--vs-numpy",
    action="store_true",
    help=(
      "Also time numpy.einsum(optimize=True) and report the moeinsum / "
      "numpy ratio. Lets you spot regressions against the canonical baseline."
    ),
  )
  args = p.parse_args(argv)

  shapes = _parse_shapes(args.shapes)
  dtype = np.dtype(args.dtype)
  operands = _make_operands(shapes, dtype, args.seed)
  platform_record = {
    "machine": platform.machine(),
    "system": platform.system(),
    "release": platform.release(),
  }

  if args.sweep_optimizers:
    # Cover every named optimizer the path planner exposes. `random-greedy`
    # is the default-N variant; callers wanting `random-greedy-N` specific
    # trial counts should run single-optimizer mode with `--optimize`.
    optimizers = (
      "naive",
      "greedy",
      "optimal",
      "auto",
      "random-greedy",
      "branch-all",
      "branch-2",
      "branch-1",
    )
    per_opt: dict[str, dict[str, object]] = {}
    for opt in optimizers:
      for _ in range(args.warmup):
        _time_one(args.equation, operands, args.backend, opt)
      timings = [_time_one(args.equation, operands, args.backend, opt) for _ in range(args.repeats)]
      per_opt[opt] = {
        "ms_median": statistics.median(timings),
        "ms_min": min(timings),
        "ms_max": max(timings),
        "path": einsum_path(args.equation, *shapes, optimize=opt),
      }
    fastest_name = min(per_opt, key=lambda k: per_opt[k]["ms_median"])
    fastest = per_opt[fastest_name]["ms_median"]
    ratios = {name: round(rec["ms_median"] / fastest, 3) for name, rec in per_opt.items()}
    result = {
      "equation": args.equation,
      "shapes": [list(s) for s in shapes],
      "backend": args.backend,
      "dtype": args.dtype,
      "repeats": args.repeats,
      "warmup": args.warmup,
      "results": per_opt,
      "fastest": fastest_name,
      "ratios": ratios,
      "platform": platform_record,
      "timestamp": datetime.now(UTC).isoformat(),
    }
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0

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
    "platform": platform_record,
    "timestamp": datetime.now(UTC).isoformat(),
  }
  if args.include_path:
    result["path"] = einsum_path(args.equation, *shapes, optimize=args.optimize)

  if args.vs_numpy:
    # Warmup numpy too so we compare hot-path to hot-path.
    for _ in range(args.warmup):
      np.einsum(args.equation, *operands, optimize=True)
    np_timings: list[float] = []
    for _ in range(args.repeats):
      t0 = time.perf_counter()
      np.einsum(args.equation, *operands, optimize=True)
      t1 = time.perf_counter()
      np_timings.append((t1 - t0) * 1000.0)
    np_median = statistics.median(np_timings)
    result["numpy_ms_median"] = np_median
    result["numpy_ms_min"] = min(np_timings)
    result["numpy_ms_max"] = max(np_timings)
    result["vs_numpy_ratio"] = round(statistics.median(timings) / np_median, 3)

  json.dump(result, sys.stdout, indent=2)
  sys.stdout.write("\n")
  return 0


if __name__ == "__main__":
  sys.exit(main())
