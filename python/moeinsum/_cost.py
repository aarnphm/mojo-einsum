"""Path-cost introspection.

Given an einsum equation, operand shapes, and a contraction path, compute
the per-step FLOP cost and the peak intermediate tensor size. Useful for
debugging why one `optimize=` setting beats another.

Stays Python-side — duplicates the cost math from `path.mojo`'s
`_flop_cost` / `_reduced_size_cost` so users can compute costs without
crossing the FFI boundary.
"""

from __future__ import annotations

from typing import cast


def _label_set(labels: list[int]) -> list[int]:
  """Return the input list deduplicated preserving order."""
  out: list[int] = []
  seen: set[int] = set()
  for lbl in labels:
    if lbl not in seen:
      seen.add(lbl)
      out.append(lbl)
  return out


def _intern_labels(eq: str) -> tuple[list[list[int]], list[int]]:
  """Cheap re-implementation of parse() that returns operand label-int
  lists + output label-int list.

  We can't call moeinsum.parse_equation here because that'd create a
  circular import; this is a deliberate duplicate. ASCII-only — matches
  what the Mojo parser handles.
  """
  if "->" in eq:
    input_part, output_part = eq.split("->", 1)
  else:
    input_part, output_part = eq, None

  intern: dict[str, int] = {}
  operand_labels: list[list[int]] = []
  for piece in input_part.split(","):
    labels: list[int] = []
    for ch in piece:
      if ch == ".":
        raise ValueError("path_cost does not support ellipsis yet")
      if not ch.isalpha():
        raise ValueError(f"unexpected character {ch!r} in equation")
      if ch not in intern:
        intern[ch] = len(intern)
      labels.append(intern[ch])
    operand_labels.append(labels)

  if output_part is not None:
    output: list[int] = []
    for ch in output_part:
      if ch == ".":
        raise ValueError("path_cost does not support ellipsis yet")
      if not ch.isalpha():
        raise ValueError(f"unexpected character {ch!r} in equation")
      if ch not in intern:
        intern[ch] = len(intern)
      output.append(intern[ch])
  else:
    counts: dict[int, int] = {}
    for op in operand_labels:
      for lbl in op:
        counts[lbl] = counts.get(lbl, 0) + 1
    output = sorted(lbl for lbl, c in counts.items() if c == 1)

  return operand_labels, output


def _label_sizes(
  operand_labels: list[list[int]],
  shapes: list[tuple[int, ...]],
) -> dict[int, int]:
  sizes: dict[int, int] = {}
  for labels, shape in zip(operand_labels, shapes, strict=True):
    if len(labels) != len(shape):
      raise ValueError(f"operand has {len(labels)} labels but shape rank {len(shape)}")
    for lbl, dim in zip(labels, shape, strict=True):
      if sizes.setdefault(lbl, dim) != dim:
        raise ValueError(f"size conflict on label {lbl}: {sizes[lbl]} vs {dim}")
  return sizes


def path_cost(
  eq: str,
  shapes: list[tuple[int, ...]],
  path: list[tuple[int, ...]],
) -> dict[str, object]:
  """Compute FLOPs + peak intermediate size for a given path.

  Args:
      eq:     Einsum equation string (no ellipsis support in this helper).
      shapes: Per-operand shape tuples in the equation's operand order.
      path:   List of `(lhs_idx, rhs_idx)` working-set pairs (the format
              `einsum_path` returns), or `(idx,)` unary singletons.

  Returns:
      ``{"total_flops": int, "peak_intermediate": int, "steps": [...]}``,
      where each step record is
      ``{"lhs": int, "rhs": int, "flops": int, "out_size": int}``.
  """
  operand_labels, final_output = _intern_labels(eq)
  sizes = _label_sizes(operand_labels, shapes)

  working: list[list[int]] = [list(op) for op in operand_labels]
  total_flops = 0
  peak = 0
  step_records: list[dict[str, int]] = []

  for step in path:
    if len(step) == 1:
      # Unary step — collapse repeated labels then drop reduce-out labels.
      idx = step[0]
      labels = working[idx]
      # Survive only labels needed downstream.
      future: set[int] = set(final_output)
      for j, op in enumerate(working):
        if j != idx:
          future.update(op)
      survived = [lbl for lbl in _label_set(labels) if lbl in future]
      step_flops = 1
      for lbl in _label_set(labels):
        step_flops *= sizes[lbl]
      out_size = 1
      for lbl in survived:
        out_size *= sizes[lbl]
      total_flops += step_flops
      peak = max(peak, out_size)
      working[idx] = survived
      step_records.append({"lhs": idx, "rhs": -1, "flops": step_flops, "out_size": out_size})
      continue

    li, ri = cast("tuple[int, int]", step)
    lhs = working[li]
    rhs = working[ri]

    # Future labels: in final_output or in any other operand still
    # in the working set.
    future = set(final_output)
    for j, op in enumerate(working):
      if j != li and j != ri:
        future.update(op)

    union = _label_set(lhs + rhs)
    out_labels = [lbl for lbl in union if lbl in future]

    step_flops = 1
    for lbl in union:
      step_flops *= sizes[lbl]

    out_size = 1
    for lbl in out_labels:
      out_size *= sizes[lbl]

    total_flops += step_flops
    peak = max(peak, out_size)
    step_records.append({"lhs": li, "rhs": ri, "flops": step_flops, "out_size": out_size})

    # Rebuild working: remove li, ri; append the new intermediate.
    new_working: list[list[int]] = [op for j, op in enumerate(working) if j != li and j != ri]
    new_working.append(out_labels)
    working = new_working

  return {
    "total_flops": total_flops,
    "peak_intermediate": peak,
    "steps": step_records,
  }
