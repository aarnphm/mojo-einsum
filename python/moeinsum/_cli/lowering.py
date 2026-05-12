"""Backend-lowering inspection CLI for moeinsum."""

from __future__ import annotations

import argparse
import json
import sys

from .. import _native, path_cost
from .._interop_max import lowering_spec as _max_lowering_spec

_OPTIMIZE = (
  "naive",
  "greedy",
  "optimal",
  "auto",
  "random-greedy",
  "branch-all",
  "branch-2",
  "branch-1",
)
_BACKENDS = ("reference", "max", "max:cpu", "max:gpu", "native")


def _parse_shapes(shape_strs: list[str]) -> list[tuple[int, ...]]:
  out = []
  for s in shape_strs:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
      raise ValueError(f"empty shape: {s!r}")
    out.append(tuple(int(p) for p in parts))
  return out


def _is_known_optimize(name: str) -> bool:
  if name in _OPTIMIZE:
    return True
  prefix = "random-greedy-"
  if name.startswith(prefix):
    suffix = name[len(prefix) :]
    return suffix.isdigit() and int(suffix) >= 1
  return False


def _compute_path(eq: str, shapes: list[tuple[int, ...]], optimize: str) -> list[tuple[int, ...]]:
  if not _is_known_optimize(optimize):
    raise ValueError(f"unknown optimize {optimize!r}; available: {_OPTIMIZE}")
  shapes_lists = [list(s) for s in shapes]
  if optimize == "naive" or len(shapes) <= 1:
    result = _native.einsum_path(eq, shapes_lists)
  else:
    result = _native.einsum_compute_path(eq, shapes_lists, optimize)
  return [tuple(step) for step in result]


def _path_cost_record(eq: str, shapes: list[tuple[int, ...]], path: list[tuple[int, ...]]) -> dict[str, object]:
  try:
    return path_cost(eq, shapes, path)
  except Exception as exc:  # noqa: BLE001
    return {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}


def _plan_graph_record(eq: str, shapes: list[tuple[int, ...]], path: list[tuple[int, ...]]) -> dict[str, object]:
  try:
    spec = _max_lowering_spec(eq, shapes, path)
  except Exception as exc:  # noqa: BLE001
    return {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
  return {
    "status": "ok",
    "ops": spec["ops"],
    "result_shape": spec["result_shape"],
  }


def _reference_record() -> dict[str, object]:
  return {
    "status": "ok",
    "supported": True,
    "coverage": "full grammar",
    "ir": "EinsumEquation -> ContractionPlan -> global-index reference loop",
    "implementation": "src/einsum/backends/reference.mojo::execute_reference",
    "ops": [
      {
        "kind": "global_index_loop",
        "target": "UnsafePointer[Float64] buffers",
        "notes": "correctness oracle; scalar serial reduction; fp64 accumulator",
      }
    ],
  }


def _max_record(
  eq: str, shapes: list[tuple[int, ...]], path: list[tuple[int, ...]], backend: str
) -> dict[str, object]:
  try:
    spec = _max_lowering_spec(eq, shapes, path)
  except Exception as exc:  # noqa: BLE001
    return {
      "status": "unsupported",
      "supported": False,
      "reason": f"{type(exc).__name__}: {exc}",
      "implementation": "python/moeinsum/_interop_max.py",
    }

  device_policy = {
    "max": "Accelerator() if max.driver.accelerator_count() > 0 else CPU()",
    "max:cpu": "CPU()",
    "max:gpu": "Accelerator(), error if none exists",
  }[backend]
  spec.update({
    "status": "ok",
    "supported": True,
    "device_policy": device_policy,
    "implementation": "python/moeinsum/_interop_max.py::_lower_graph",
    "compiler_target": "MAX Graph",
    "mojo_backend": {
      "implementation": "src/einsum/backends/max.mojo::execute_max",
      "abi": "UnsafePointer[Float64] flat buffers with TTGT-style pack buffers",
      "pairwise_target": "TileTensor + linalg.bmm.batched_matmul",
      "unary_target": "Mojo stride views and reduce_sum_axes",
      "target_policy": "compile-time target parameter; public FFI backend plumbing still pending",
    },
  })
  return spec


def _native_record() -> dict[str, object]:
  return {
    "status": "ok",
    "supported": True,
    "implementation": "src/einsum/backends/native.mojo",
    "ir": "EinsumEquation -> ContractionPlan -> flat-buffer Mojo plan executor",
    "ops": [
      {
        "kind": "plan_working_set",
        "target": "UnsafePointer[Float64] buffers",
        "notes": "correctness backend for native dispatch; deterministic mixed-radix reductions",
      }
    ],
    "kernel_cutover": [
      "TileTensor/RuntimeLayout operand views",
      "GETT packer for permute-heavy contractions",
      "linalg.batched_matmul fallback for already-BMM-shaped contractions",
      "opcode-level accumulator dtype control",
    ],
  }


def inspect_lowering(
  eq: str,
  shapes: list[tuple[int, ...]],
  *,
  optimize: str = "auto",
  backend: str = "all",
) -> dict[str, object]:
  """Return parser, path, cost, and per-backend lowering information."""
  if backend != "all" and backend not in _BACKENDS:
    raise ValueError(f"unknown backend {backend!r}; available: all, {_BACKENDS}")

  path = _compute_path(eq, shapes, optimize)
  selected = _BACKENDS if backend == "all" else (backend,)
  backend_records: dict[str, object] = {}
  for name in selected:
    if name == "reference":
      backend_records[name] = _reference_record()
    elif name in {"max", "max:cpu", "max:gpu"}:
      backend_records[name] = _max_record(eq, shapes, path, name)
    elif name == "native":
      backend_records[name] = _native_record()

  return {
    "equation": eq,
    "shapes": [list(s) for s in shapes],
    "optimize": optimize,
    "path": [list(step) for step in path],
    "parser_ir": _native.parse_equation(eq),
    "path_cost": _path_cost_record(eq, shapes, path),
    "plan_graph_spec": _plan_graph_record(eq, shapes, path),
    "backends": backend_records,
  }


def main(argv: list[str] | None = None) -> int:
  p = argparse.ArgumentParser(
    prog="moeinsum-lowering",
    description="Inspect parser/path/backend lowering as JSON without executing the contraction.",
  )
  p.add_argument("equation")
  p.add_argument(
    "--shapes",
    nargs="+",
    required=True,
    help="Per-operand shapes, comma-separated. E.g. --shapes 3,4 4,5",
  )
  p.add_argument("--optimize", default="auto", help="Path optimizer name")
  p.add_argument(
    "--backend",
    default="all",
    choices=["all", *_BACKENDS],
    help="Backend lowering to show",
  )
  args = p.parse_args(argv)

  try:
    rec = inspect_lowering(args.equation, _parse_shapes(args.shapes), optimize=args.optimize, backend=args.backend)
  except Exception as exc:  # noqa: BLE001
    p.exit(2, f"{p.prog}: error: {exc}\n")

  json.dump(rec, sys.stdout, indent=2)
  sys.stdout.write("\n")
  return 0


if __name__ == "__main__":
  sys.exit(main())
