"""Backend-lowering inspection CLI for moeinsum."""

from __future__ import annotations

import argparse
import sys

from .._lowering import BACKENDS, dump_lowering_ir


def _parse_shapes(shape_strs: list[str]) -> list[tuple[int, ...]]:
  out = []
  for s in shape_strs:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
      raise ValueError(f"empty shape: {s!r}")
    out.append(tuple(int(p) for p in parts))
  return out


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
    choices=["all", *BACKENDS],
    help="Backend lowering to show",
  )
  args = p.parse_args(argv)

  try:
    dump_lowering_ir(args.equation, _parse_shapes(args.shapes), optimize=args.optimize, backend=args.backend)
  except Exception as exc:  # noqa: BLE001
    p.exit(2, f"{p.prog}: error: {exc}\n")

  return 0


if __name__ == "__main__":
  sys.exit(main())
