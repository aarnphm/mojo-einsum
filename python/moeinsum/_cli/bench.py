"""Benchmark CLI - `moeinsum-bench` (or `python -m moeinsum._cli.bench`).

Emits per-step timing as JSON. Runs each measurement N times, reports
the median (robust to GC pauses), min, and max.

Single-optimizer mode (default):

  moeinsum-bench "ij,jk->ik" --shapes 256,256 256,256
  moeinsum-bench "ij,jk,kl->il" --shapes 64,64 64,64 64,64 \\
      --optimize auto --repeats 100

Optimizer-sweep mode (`--sweep-optimizers`): runs every named optimizer
against the same operands and emits a ratios table (median time vs.
fastest):

  moeinsum-bench "ij,jk,kl,lm->im" --shapes 8,4 4,16 16,2 2,32 \\
      --sweep-optimizers

Comparison mode (`--compare`): times moeinsum plus installed comparison
engines (`numpy`, `opt_einsum`, `jax`, `torch`, `mlx`). Missing engines
are recorded as skipped instead of failing the primary bench.

  moeinsum-bench "ij,jk->ik" --shapes 1024,1024 1024,1024 --compare

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

Output JSON (comparison mode):
  {"comparisons": {"moeinsum": {"status": "ok", "ms_median": ...},
                   "numpy": {"status": "ok", "speedup_vs_moeinsum": ...},
                   "mlx": {"status": "skipped", "reason": "..."}},
   "comparison_fastest": "numpy",
   "comparison_ratios": {"moeinsum": 2.1, "numpy": 1.0}}

Progress bars render to stderr when requested with `--progress`, or by
default when stderr is interactive. JSON always stays on stdout.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import statistics
import sys
import sysconfig
import time
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

# Set the libpython link before `moeinsum` imports `_native` and triggers the
# mohaus editable rebuild. uv-managed interpreters live outside the loader's
# default search path, so the rebuild fails without this.
if "MOJO_PYTHON_LIBRARY" not in os.environ:
  _libdir = sysconfig.get_config_var("LIBDIR")
  _libname = sysconfig.get_config_var("LDLIBRARY")
  if _libdir and _libname:
    _candidate = Path(_libdir) / _libname
    if _candidate.is_file():
      os.environ["MOJO_PYTHON_LIBRARY"] = str(_candidate)

import numpy as np

from .. import einsum, einsum_path
from .._cache import PLAN_CACHE

_tqdm: Callable[..., Iterable[int]] | None
try:
  from tqdm import tqdm as _tqdm_import
except ImportError:  # pragma: no cover - exercised by environments without the bench group
  _tqdm = None
else:
  _tqdm = _tqdm_import


_DEFAULT_COMPARE_ENGINES = ("numpy", "opt_einsum", "jax", "torch", "mlx")
_COMPARE_ENGINE_ALIASES = {
  "numpy": "numpy",
  "np": "numpy",
  "opt_einsum": "opt_einsum",
  "opt-einsum": "opt_einsum",
  "oe": "opt_einsum",
  "jax": "jax",
  "torch": "torch",
  "pytorch": "torch",
  "mlx": "mlx",
}
_MAX_OP_LOG_LEVELS = ("notset", "trace", "debug", "info", "warning", "error", "critical")


def _parse_shapes(shape_strs: list[str]) -> list[tuple[int, ...]]:
  """`["3,4,5", "5,6"]` -> `[(3,4,5), (5,6)]`."""
  out = []
  for s in shape_strs:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
      raise ValueError(f"empty shape: {s!r}")
    out.append(tuple(int(p) for p in parts))
  return out


def _make_numpy_operands(shapes: list[tuple[int, ...]], dtype: np.dtype, seed: int) -> list[np.ndarray]:
  rng = np.random.default_rng(seed)
  return [rng.standard_normal(s).astype(dtype) for s in shapes]


def _torch_dtype(dtype: np.dtype) -> object:
  import torch  # noqa: PLC0415

  name = np.dtype(dtype).name
  if name == "bfloat16":
    return torch.bfloat16
  torch_dtype = getattr(torch, name, None)
  if torch_dtype is None:
    raise NotImplementedError(f"torch benchmark operands do not support dtype {dtype}")
  return torch_dtype


def _make_torch_operands(
  arrays: Sequence[np.ndarray],
  *,
  dtype: np.dtype,
  device: str,
) -> list[object]:
  import torch  # noqa: PLC0415

  if device == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("--input-device cuda requested, but torch.cuda.is_available() is false")
  torch_dtype = _torch_dtype(dtype)
  out: list[object] = []
  for array in arrays:
    if dtype.name == "bfloat16":
      tensor = torch.as_tensor(array.astype("float32"), device=device).to(dtype=torch_dtype)
    else:
      tensor = torch.from_numpy(array).to(device=device, dtype=torch_dtype)
    out.append(tensor.contiguous())
  return out


def _make_operands(
  shapes: list[tuple[int, ...]],
  dtype: np.dtype,
  seed: int,
  *,
  input_framework: str,
  input_device: str,
) -> tuple[list[np.ndarray], list[object]]:
  numpy_operands = _make_numpy_operands(shapes, dtype, seed)
  if input_framework == "numpy":
    if input_device != "cpu":
      raise ValueError("numpy benchmark operands only support --input-device cpu")
    return numpy_operands, list(numpy_operands)
  if input_framework == "torch":
    return numpy_operands, _make_torch_operands(numpy_operands, dtype=dtype, device=input_device)
  raise ValueError(f"unknown input framework {input_framework!r}")


def _torch_cuda_sync_for_operands(operands: Sequence[object]) -> Callable[[], None] | None:
  if not operands:
    return None
  module = type(operands[0]).__module__.split(".", 1)[0]
  if module != "torch":
    return None
  import torch  # noqa: PLC0415

  if any(isinstance(operand, torch.Tensor) and operand.is_cuda for operand in operands):
    return torch.cuda.synchronize
  return None


def _all_torch_operands(operands: Sequence[object]) -> bool:
  if not operands:
    return False
  return type(operands[0]).__module__.split(".", 1)[0] == "torch"


def _configure_torch_bench(
  *,
  allow_tf32: bool | None,
  matmul_precision: str | None,
  record: bool,
) -> dict[str, object] | None:
  if allow_tf32 is None and matmul_precision is None and not record:
    return None

  import torch  # noqa: PLC0415

  if allow_tf32 is not None:
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    if hasattr(torch.backends, "cudnn"):
      torch.backends.cudnn.allow_tf32 = allow_tf32
  if matmul_precision is not None:
    torch.set_float32_matmul_precision(matmul_precision)
  return {
    "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
    "float32_matmul_precision": torch.get_float32_matmul_precision(),
  }


def _time_one(
  eq: str,
  operands: Sequence[object],
  backend: str,
  optimize: str,
  sync: Callable[[], None] | None = None,
) -> float:
  """Return wall time of one einsum call, in milliseconds."""
  if sync is not None:
    sync()
  t0 = time.perf_counter()
  einsum(eq, *operands, backend=backend, optimize=optimize)
  if sync is not None:
    sync()
  t1 = time.perf_counter()
  return (t1 - t0) * 1000.0


def _progress_range(count: int, *, desc: str, enabled: bool) -> Iterable[int]:
  values = range(count)
  if not enabled:
    return values
  if _tqdm is None:
    return values
  return _tqdm(
    values,
    total=count,
    desc=desc,
    unit="run",
    file=sys.stderr,
    dynamic_ncols=True,
    leave=False,
  )


def _parse_compare_engines(raw: str) -> list[str]:
  if raw.strip() == "all":
    return list(_DEFAULT_COMPARE_ENGINES)

  out: list[str] = []
  for part in raw.split(","):
    key = part.strip().lower()
    if not key:
      continue
    engine = _COMPARE_ENGINE_ALIASES.get(key)
    if engine is None:
      names = ", ".join(sorted(_COMPARE_ENGINE_ALIASES))
      raise ValueError(f"unknown compare engine {part!r}; available: all, {names}")
    if engine not in out:
      out.append(engine)
  if not out:
    raise ValueError("--compare-engines must name at least one engine")
  return out


def _configure_modular_debug(
  *,
  modular_debug: str | None,
  max_ir_output_dir: str | None,
  max_op_log_level: str | None,
  max_source_tracebacks: bool,
  max_device_sync: bool,
  max_nan_check: bool,
) -> str | None:
  tokens: list[str] = []
  if modular_debug:
    tokens.extend(part.strip() for part in modular_debug.split(",") if part.strip())
  if max_ir_output_dir:
    tokens.append(f"ir-output-dir={max_ir_output_dir}")
  if max_op_log_level:
    tokens.append(f"op-log-level={max_op_log_level}")
  if max_source_tracebacks:
    tokens.append("source-tracebacks")
  if max_device_sync:
    tokens.append("device-sync-mode")
  if max_nan_check:
    tokens.append("nan-check")
  if not tokens:
    return os.environ.get("MODULAR_DEBUG") or None

  existing = os.environ.get("MODULAR_DEBUG", "").strip()
  merged = ",".join([existing, *tokens] if existing else tokens)
  os.environ["MODULAR_DEBUG"] = merged
  return merged


def _apply_max_debug_api(
  *,
  max_ir_output_dir: str | None,
  max_op_log_level: str | None,
  max_source_tracebacks: bool,
  max_device_sync: bool,
  max_nan_check: bool,
) -> None:
  if max_source_tracebacks:
    from max.graph import Graph  # noqa: PLC0415

    graph_debug = getattr(Graph, "debug", None)
    if graph_debug is not None:
      graph_debug.source_tracebacks = True

  if not any((max_ir_output_dir, max_op_log_level, max_device_sync, max_nan_check)):
    return

  from max.engine import InferenceSession  # noqa: PLC0415

  debug = getattr(InferenceSession, "debug", None)
  if debug is None:
    return
  if max_ir_output_dir:
    debug.ir_output_dir = max_ir_output_dir
  if max_op_log_level:
    debug.op_log_level = max_op_log_level
  if max_device_sync:
    debug.device_sync_mode = True
  if max_nan_check:
    debug.nan_check = True


def _time_callable(
  fn: Callable[[], object],
  *,
  repeats: int,
  warmup: int,
  desc: str,
  show_progress: bool,
  sync: Callable[[], None] | None = None,
) -> dict[str, object]:
  for _ in _progress_range(warmup, desc=f"warmup {desc}", enabled=show_progress):
    fn()
    if sync is not None:
      sync()

  timings: list[float] = []
  for _ in _progress_range(repeats, desc=f"bench {desc}", enabled=show_progress):
    if sync is not None:
      sync()
    t0 = time.perf_counter()
    fn()
    if sync is not None:
      sync()
    t1 = time.perf_counter()
    timings.append((t1 - t0) * 1000.0)

  return {
    "status": "ok",
    "ms_median": statistics.median(timings),
    "ms_min": min(timings),
    "ms_max": max(timings),
  }


def _compare_callable(
  engine: str,
  eq: str,
  numpy_operands: list[np.ndarray],
  primary_operands: Sequence[object],
  optimize: str,
) -> tuple[Callable[[], object], Callable[[], None] | None]:
  if engine == "numpy":
    if numpy_operands and numpy_operands[0].dtype.name == "bfloat16":
      raise ImportError("numpy.einsum upgrades bf16 to fp32 silently; skipping to avoid a misleading measurement")
    return lambda: np.einsum(eq, *numpy_operands, optimize=True), None

  if engine == "opt_einsum":
    import opt_einsum  # noqa: PLC0415

    operands = primary_operands if _all_torch_operands(primary_operands) else numpy_operands
    sync = _torch_cuda_sync_for_operands(operands)
    contract = cast("Callable[..., object]", opt_einsum.contract)
    if optimize == "naive":
      return lambda: contract(eq, *operands, optimize=False), sync
    if optimize == "greedy":
      return lambda: contract(eq, *operands, optimize="greedy"), sync
    if optimize == "optimal":
      return lambda: contract(eq, *operands, optimize="optimal"), sync
    if optimize == "branch-all":
      return lambda: contract(eq, *operands, optimize="branch-all"), sync
    if optimize == "branch-2":
      return lambda: contract(eq, *operands, optimize="branch-2"), sync
    if optimize == "branch-1":
      return lambda: contract(eq, *operands, optimize="branch-1"), sync
    if optimize.startswith("random-greedy"):
      return lambda: contract(eq, *operands, optimize="random-greedy"), sync
    return lambda: contract(eq, *operands, optimize="auto"), sync

  if engine == "jax":
    import jax  # noqa: PLC0415

    if any(arr.dtype == np.float64 for arr in numpy_operands):
      jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp  # noqa: PLC0415

    jax_operands = [jnp.asarray(arr) for arr in numpy_operands]

    def run_jax() -> object:
      out = jnp.einsum(eq, *jax_operands, optimize=True)
      return out.block_until_ready()

    return run_jax, None

  if engine == "torch":
    import torch  # noqa: PLC0415

    if _all_torch_operands(primary_operands):
      torch_operands = list(primary_operands)
    else:
      torch_operands = [torch.from_numpy(arr) for arr in numpy_operands]
    sync = _torch_cuda_sync_for_operands(torch_operands)
    return lambda: torch.einsum(eq, *torch_operands), sync

  if engine == "mlx":
    mx = importlib.import_module("mlx.core")

    mlx_operands = [mx.array(arr) for arr in numpy_operands]

    def run_mlx() -> object:
      out = mx.einsum(eq, *mlx_operands)
      mx.eval(out)
      return out

    return run_mlx, None

  raise ValueError(f"unknown compare engine {engine!r}")


def _run_comparisons(
  *,
  engines: list[str],
  eq: str,
  numpy_operands: list[np.ndarray],
  primary_operands: Sequence[object],
  optimize: str,
  repeats: int,
  warmup: int,
  show_progress: bool,
  moeinsum_record: dict[str, object],
) -> dict[str, object]:
  comparisons: dict[str, dict[str, object]] = {
    "moeinsum": {
      "status": "ok",
      "ms_median": moeinsum_record["total_ms_median"],
      "ms_min": moeinsum_record["total_ms_min"],
      "ms_max": moeinsum_record["total_ms_max"],
    }
  }

  for engine in engines:
    try:
      fn, sync = _compare_callable(engine, eq, numpy_operands, primary_operands, optimize)
    except ModuleNotFoundError as exc:
      comparisons[engine] = {"status": "skipped", "reason": f"missing import: {exc.name}"}
      continue
    except ImportError as exc:
      comparisons[engine] = {"status": "skipped", "reason": str(exc)}
      continue

    try:
      comparisons[engine] = _time_callable(
        fn,
        repeats=repeats,
        warmup=warmup,
        desc=engine,
        show_progress=show_progress,
        sync=sync,
      )
    except Exception as exc:  # noqa: BLE001 - comparison engines should not kill the primary bench
      comparisons[engine] = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}

  ok_medians: dict[str, float] = {}
  for name, rec in comparisons.items():
    ms_median = rec.get("ms_median")
    if rec.get("status") == "ok" and isinstance(ms_median, (int, float)):
      ok_medians[name] = float(ms_median)

  fastest = min(ok_medians, key=lambda name: ok_medians[name])
  fastest_ms = ok_medians[fastest]
  ratios = {name: round(ms / fastest_ms, 3) for name, ms in ok_medians.items()}
  moe_ms = ok_medians["moeinsum"]
  for name, ms in ok_medians.items():
    comparisons[name]["time_ratio_vs_moeinsum"] = round(ms / moe_ms, 3)
    comparisons[name]["speedup_vs_moeinsum"] = round(moe_ms / ms, 3) if ms > 0 else 0.0

  return {
    "comparisons": comparisons,
    "comparison_fastest": fastest,
    "comparison_ratios": ratios,
  }


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
    choices=["reference", "max", "max:cpu", "max:gpu", "native"],
    help="Execution backend",
  )
  p.add_argument(
    "--modular-debug",
    help=(
      "Comma-separated MODULAR_DEBUG tokens to set before MAX loads. "
      "Example: ir-output-dir=.max-ir,op-log-level=trace,source-tracebacks"
    ),
  )
  p.add_argument(
    "--max-ir-output-dir",
    help="Shortcut for MODULAR_DEBUG=ir-output-dir=<dir>; dumps MAX compiler IR during graph compile.",
  )
  p.add_argument(
    "--max-op-log-level",
    choices=_MAX_OP_LOG_LEVELS,
    help="Shortcut for MODULAR_DEBUG=op-log-level=<level>; trace logs each MAX op launch/complete.",
  )
  p.add_argument(
    "--max-source-tracebacks",
    action="store_true",
    help="Shortcut for MODULAR_DEBUG=source-tracebacks.",
  )
  p.add_argument(
    "--max-device-sync",
    action="store_true",
    help="Shortcut for MODULAR_DEBUG=device-sync-mode; useful for GPU error localization.",
  )
  p.add_argument(
    "--max-nan-check",
    action="store_true",
    help="Shortcut for MODULAR_DEBUG=nan-check.",
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
    choices=["float32", "float64", "bfloat16"],
    help=(
      "Operand dtype. bfloat16 routes through ml_dtypes and is only meaningful "
      "with backend=max:cpu or max:gpu - numpy/opt_einsum will skip with a clear "
      "reason since they do not support bf16 natively."
    ),
  )
  p.add_argument(
    "--input-framework",
    default="numpy",
    choices=["numpy", "torch"],
    help=(
      "Framework used to allocate primary benchmark operands. Use "
      "`--input-framework torch --input-device cuda` to measure MAX from "
      "CUDA-resident DLPack operands instead of NumPy host arrays."
    ),
  )
  p.add_argument(
    "--input-device",
    default="cpu",
    choices=["cpu", "cuda"],
    help="Device for primary benchmark operands. `cuda` currently requires --input-framework torch.",
  )
  p.add_argument(
    "--torch-allow-tf32",
    action=argparse.BooleanOptionalAction,
    default=None,
    help=(
      "Set torch.backends.cuda.matmul.allow_tf32 and cudnn.allow_tf32 before "
      "benchmarks. Useful because MAX GPU matmul may use TF32 on NVIDIA."
    ),
  )
  p.add_argument(
    "--torch-matmul-precision",
    choices=["highest", "high", "medium"],
    help="Forwarded to torch.set_float32_matmul_precision for torch comparison rows.",
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
      "numpy ratio. Compatibility shorthand for --compare-engines numpy."
    ),
  )
  p.add_argument(
    "--compare",
    action="store_true",
    help="Also time every installed comparison engine: numpy, opt_einsum, jax, torch, mlx.",
  )
  p.add_argument(
    "--compare-engines",
    help=(
      "Comma-separated comparison engines. Values: all, numpy, opt_einsum, "
      "jax, torch, mlx. Supplying this flag implies --compare."
    ),
  )
  p.add_argument(
    "--cache-bench",
    action="store_true",
    help=(
      "Clear PLAN_CACHE, time one cold call, then time --repeats hot calls. "
      "Emits cold_ms, hot_ms_median, and cache_speedup_ratio."
    ),
  )
  progress = p.add_mutually_exclusive_group()
  progress.add_argument(
    "--progress",
    dest="progress",
    action="store_true",
    default=None,
    help="Show progress bars on stderr. JSON output stays on stdout.",
  )
  progress.add_argument(
    "--no-progress",
    dest="progress",
    action="store_false",
    help="Disable progress bars.",
  )
  args = p.parse_args(argv)
  if args.repeats < 1:
    p.error("--repeats must be >= 1")
  if args.warmup < 0:
    p.error("--warmup must be >= 0")

  modular_debug = _configure_modular_debug(
    modular_debug=args.modular_debug,
    max_ir_output_dir=args.max_ir_output_dir,
    max_op_log_level=args.max_op_log_level,
    max_source_tracebacks=args.max_source_tracebacks,
    max_device_sync=args.max_device_sync,
    max_nan_check=args.max_nan_check,
  )
  show_progress = sys.stderr.isatty() if args.progress is None else args.progress
  if show_progress and _tqdm is None:
    p.error("progress output requires tqdm; run with `uv run --group bench moeinsum-bench ...`")
  if args.backend.startswith("max"):
    _apply_max_debug_api(
      max_ir_output_dir=args.max_ir_output_dir,
      max_op_log_level=args.max_op_log_level,
      max_source_tracebacks=args.max_source_tracebacks,
      max_device_sync=args.max_device_sync,
      max_nan_check=args.max_nan_check,
    )
  if args.compare_engines is not None:
    try:
      compare_engines = _parse_compare_engines(args.compare_engines)
    except ValueError as exc:
      p.error(str(exc))
  elif args.compare:
    compare_engines = list(_DEFAULT_COMPARE_ENGINES)
  elif args.vs_numpy:
    compare_engines = ["numpy"]
  else:
    compare_engines = []
  if args.sweep_optimizers and compare_engines:
    p.error("--compare, --compare-engines, and --vs-numpy cannot be combined with --sweep-optimizers")
  if args.input_framework == "numpy" and args.input_device != "cpu":
    p.error("--input-device cuda requires --input-framework torch")

  shapes = _parse_shapes(args.shapes)
  if args.dtype == "bfloat16":
    import ml_dtypes  # noqa: PLC0415

    dtype = np.dtype(ml_dtypes.bfloat16)
  else:
    dtype = np.dtype(args.dtype)
  try:
    torch_config = _configure_torch_bench(
      allow_tf32=args.torch_allow_tf32,
      matmul_precision=args.torch_matmul_precision,
      record=args.input_framework == "torch",
    )
    numpy_operands, operands = _make_operands(
      shapes,
      dtype,
      args.seed,
      input_framework=args.input_framework,
      input_device=args.input_device,
    )
  except (ImportError, RuntimeError, ValueError, NotImplementedError) as exc:
    p.error(str(exc))
  primary_sync = _torch_cuda_sync_for_operands(operands)
  platform_record = {
    "machine": platform.machine(),
    "system": platform.system(),
    "release": platform.release(),
  }

  if args.sweep_optimizers:
    # Cover every named optimizer the path planner exposes. `random-greedy` is
    # the default-N variant; callers wanting `random-greedy-N` specific trial
    # counts should run single-optimizer mode with `--optimize`.
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
    per_opt_medians: dict[str, float] = {}
    for opt in optimizers:
      for _ in _progress_range(args.warmup, desc=f"warmup {opt}", enabled=show_progress):
        _time_one(args.equation, operands, args.backend, opt, sync=primary_sync)
      timings = [
        _time_one(args.equation, operands, args.backend, opt, sync=primary_sync)
        for _ in _progress_range(args.repeats, desc=f"bench {opt}", enabled=show_progress)
      ]
      median = statistics.median(timings)
      per_opt_medians[opt] = median
      per_opt[opt] = {
        "ms_median": median,
        "ms_min": min(timings),
        "ms_max": max(timings),
        "path": einsum_path(args.equation, *shapes, optimize=opt),
      }
    fastest_name = min(per_opt_medians, key=lambda k: per_opt_medians[k])
    fastest = per_opt_medians[fastest_name]
    # Round ratios to 3dp; anything past that is noise on perf_counter timings.
    ratios = {name: round(ms / fastest, 3) for name, ms in per_opt_medians.items()}
    result = {
      "equation": args.equation,
      "shapes": [list(s) for s in shapes],
      "backend": args.backend,
      "dtype": args.dtype,
      "input_framework": args.input_framework,
      "input_device": args.input_device,
      "repeats": args.repeats,
      "warmup": args.warmup,
      "results": per_opt,
      "fastest": fastest_name,
      "ratios": ratios,
      "platform": platform_record,
      "timestamp": datetime.now(UTC).isoformat(),
    }
    if modular_debug is not None:
      result["modular_debug"] = modular_debug
    if args.max_ir_output_dir:
      result["max_ir_output_dir"] = args.max_ir_output_dir
    if torch_config is not None:
      result["torch"] = torch_config
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0

  # Warmup.
  for _ in _progress_range(args.warmup, desc="warmup moeinsum", enabled=show_progress):
    _time_one(args.equation, operands, args.backend, args.optimize, sync=primary_sync)

  timings = []
  for _ in _progress_range(args.repeats, desc="bench moeinsum", enabled=show_progress):
    timings.append(_time_one(args.equation, operands, args.backend, args.optimize, sync=primary_sync))

  result = {
    "equation": args.equation,
    "shapes": [list(s) for s in shapes],
    "backend": args.backend,
    "optimize": args.optimize,
    "dtype": args.dtype,
    "input_framework": args.input_framework,
    "input_device": args.input_device,
    "repeats": args.repeats,
    "warmup": args.warmup,
    "total_ms_median": statistics.median(timings),
    "total_ms_min": min(timings),
    "total_ms_max": max(timings),
    "platform": platform_record,
    "timestamp": datetime.now(UTC).isoformat(),
  }
  if modular_debug is not None:
    result["modular_debug"] = modular_debug
  if args.max_ir_output_dir:
    result["max_ir_output_dir"] = args.max_ir_output_dir
  if torch_config is not None:
    result["torch"] = torch_config
  if args.include_path:
    result["path"] = einsum_path(args.equation, *shapes, optimize=args.optimize)

  if args.cache_bench:
    PLAN_CACHE.clear()
    # Cold call: empty the cache, time one un-cached execution. The cold path
    # pays parse + plan + path optimize + reference loop.
    if primary_sync is not None:
      primary_sync()
    t0 = time.perf_counter()
    einsum(args.equation, *operands, backend=args.backend, optimize=args.optimize)
    if primary_sync is not None:
      primary_sync()
    t1 = time.perf_counter()
    cold_ms = (t1 - t0) * 1000.0

    hot_timings: list[float] = []
    # Hot calls: cache is now populated. Subsequent invocations skip parse +
    # plan and go straight to the kernel.
    for _ in _progress_range(args.repeats, desc="cache-bench hot", enabled=show_progress):
      hot_timings.append(_time_one(args.equation, operands, args.backend, args.optimize, sync=primary_sync))
    hot_median = statistics.median(hot_timings)
    result["cold_ms"] = cold_ms
    result["hot_ms_median"] = hot_median
    result["hot_ms_min"] = min(hot_timings)
    result["hot_ms_max"] = max(hot_timings)
    result["cache_speedup_ratio"] = round(cold_ms / hot_median, 3) if hot_median > 0 else 0.0

  if compare_engines:
    comparison_result = _run_comparisons(
      engines=compare_engines,
      eq=args.equation,
      numpy_operands=numpy_operands,
      primary_operands=operands,
      optimize=args.optimize,
      repeats=args.repeats,
      warmup=args.warmup,
      show_progress=show_progress,
      moeinsum_record=result,
    )
    result.update(comparison_result)

  if args.vs_numpy:
    comparisons_obj = result.get("comparisons")
    numpy_record = comparisons_obj.get("numpy") if isinstance(comparisons_obj, dict) else None
    if isinstance(numpy_record, dict) and numpy_record.get("status") == "ok":
      np_median = numpy_record["ms_median"]
      result["numpy_ms_median"] = np_median
      result["numpy_ms_min"] = numpy_record["ms_min"]
      result["numpy_ms_max"] = numpy_record["ms_max"]
      result["vs_numpy_ratio"] = round(statistics.median(timings) / np_median, 3)

  json.dump(result, sys.stdout, indent=2)
  sys.stdout.write("\n")
  return 0


if __name__ == "__main__":
  sys.exit(main())
