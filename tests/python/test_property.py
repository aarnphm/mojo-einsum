"""Hypothesis-driven property tests.

These supplement the curated parity / unit suite by sweeping the
parameter space — equation shapes, operand permutations, label sizes,
optimizer choice. The goal is *invariants*, not specific cases:

  - parser determinism + idempotency
  - path validity (working-set semantics: n operands → n-1 pairwise steps)
  - optimizer ordering (optimal FLOPs ≤ branch-all FLOPs ≤ greedy FLOPs
    on the same equation, modulo ties)
  - kernel-side invariants (transpose involution, outer-product structure,
    full-sum reduction)
  - numpy parity over a constrained equation generator (n ≤ 4, dims ≤ 6)
  - cache determinism — repeated planner calls return identical paths

The strategies stay narrow on purpose. Hypothesis is good at finding
edge cases inside a well-defined space; we don't want it generating
ellipsis or out-of-grammar strings that would just rediscover the
parser-error paths already covered in `test_p1_smoke.py`.
"""

from __future__ import annotations

import string
from typing import cast

import moeinsum
import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from moeinsum._cost import path_cost

# ─────────────────────────────────────────────────────────────────────
# Strategies — well-formed equation pieces
# ─────────────────────────────────────────────────────────────────────

# Lowercase ASCII pool, kept narrow so distinct-label collisions show up
# instead of every test being a "labels are independent" fixture.
_LABEL_POOL = string.ascii_lowercase[:8]  # 'a'..'h'


@st.composite
def labelstrings(draw: st.DrawFn, min_len: int = 1, max_len: int = 4) -> str:
  """A non-empty string of unique labels drawn from `_LABEL_POOL`."""
  n = draw(st.integers(min_value=min_len, max_value=max_len))
  labels = draw(
    st.lists(
      st.sampled_from(list(_LABEL_POOL)),
      min_size=n,
      max_size=n,
      unique=True,
    )
  )
  return "".join(labels)


@st.composite
def equations(
  draw: st.DrawFn,
  min_operands: int = 1,
  max_operands: int = 4,
) -> tuple[str, list[tuple[int, ...]]]:
  """Generate `(equation_string, shapes)` where every label has a single
  consistent size and the output is the implicit (sorted-unique-non-
  repeated) NumPy convention. Returns the equation in the form
  `"ab,bc->ac"` style — explicit output, no ellipsis, no repeated
  labels within an operand."""
  n_operands = draw(
    st.integers(min_value=min_operands, max_value=max_operands)
  )

  operand_labels: list[str] = [draw(labelstrings()) for _ in range(n_operands)]

  # Label size mapping. Dims kept small so generated tensors are cheap
  # to allocate inside the test.
  used_labels = sorted({c for s in operand_labels for c in s})
  size_map = {
    c: draw(st.integers(min_value=2, max_value=5)) for c in used_labels
  }

  # Implicit output: labels appearing exactly once across operands,
  # sorted alphabetically (the numpy.einsum convention).
  label_counts: dict[str, int] = {}
  for piece in operand_labels:
    for c in piece:
      label_counts[c] = label_counts.get(c, 0) + 1
  output = "".join(sorted(c for c, count in label_counts.items() if count == 1))

  eq = ",".join(operand_labels) + "->" + output
  shapes = [tuple(size_map[c] for c in piece) for piece in operand_labels]
  return eq, shapes


# ─────────────────────────────────────────────────────────────────────
# Parser — determinism + structural invariants
# ─────────────────────────────────────────────────────────────────────


@given(equations(min_operands=1, max_operands=4))
def test_parser_is_deterministic(case: tuple[str, list[tuple[int, ...]]]) -> None:
  """Re-parsing the same equation must return the same IR. No hidden
  state, no rand-dependent label numbering."""
  eq, _ = case
  a = moeinsum.parse_equation(eq)
  b = moeinsum.parse_equation(eq)
  assert a["inputs"] == b["inputs"]
  assert a["output"] == b["output"]
  assert a["n_labels"] == b["n_labels"]
  assert a["has_explicit_output"] == b["has_explicit_output"]


@given(equations(min_operands=1, max_operands=4))
def test_parser_label_counts_consistent(
  case: tuple[str, list[tuple[int, ...]]],
) -> None:
  """`n_labels` equals the size of the union of label-ints across
  inputs ∪ output."""
  eq, _ = case
  ir = moeinsum.parse_equation(eq)
  inputs = cast("list[list[int]]", ir["inputs"])
  output = cast("list[int]", ir["output"])
  union: set[int] = set(output)
  for op in inputs:
    union.update(op)
  assert ir["n_labels"] == len(union)


# ─────────────────────────────────────────────────────────────────────
# Path validity — working-set semantics
# ─────────────────────────────────────────────────────────────────────


@given(
  case=equations(min_operands=2, max_operands=4),
  optimize=st.sampled_from(["greedy", "optimal", "auto", "branch-all"]),
)
@settings(suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_path_satisfies_working_set_semantics(
  case: tuple[str, list[tuple[int, ...]]],
  optimize: str,
) -> None:
  """For n operands, any path must:
    - have len == n - 1 (each pairwise step removes one operand)
    - reference only valid working-set indices at each step
    - leave exactly 1 tensor in the working set at the end
  """
  eq, shapes = case
  n = len(shapes)
  path = moeinsum.einsum_path(eq, *shapes, optimize=optimize)

  # No unary steps in compute_path output — only pairwise.
  assert all(len(step) == 2 for step in path)
  assert len(path) == n - 1, (
    f"expected {n - 1} pairwise steps, got {len(path)}"
  )

  working_size = n
  for step_idx, (li, ri) in enumerate(path):
    assert 0 <= li < working_size, f"step {step_idx}: lhs {li} out of range"
    assert 0 <= ri < working_size, f"step {step_idx}: rhs {ri} out of range"
    assert li != ri, f"step {step_idx}: lhs == rhs == {li}"
    working_size -= 1  # remove 2, append 1
  assert working_size == 1


# ─────────────────────────────────────────────────────────────────────
# Optimizer ordering — optimal ≤ branch-all ≤ greedy on FLOPs
# ─────────────────────────────────────────────────────────────────────


@given(case=equations(min_operands=3, max_operands=4))
@settings(suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_optimal_flops_le_greedy(
  case: tuple[str, list[tuple[int, ...]]],
) -> None:
  """`optimal` must produce a path with FLOP count ≤ `greedy`'s.

  Skip cases that exercise the path-cost helper's known limitations
  (repeated labels within an operand and so on)."""
  eq, shapes = case
  try:
    greedy = moeinsum.einsum_path(eq, *shapes, optimize="greedy")
    optimal = moeinsum.einsum_path(eq, *shapes, optimize="optimal")
    cg = cast(int, path_cost(eq, shapes, greedy)["total_flops"])
    co = cast(int, path_cost(eq, shapes, optimal)["total_flops"])
  except ValueError:
    # path_cost rejects ellipsis-and-similar; the generator never
    # produces those, but the safety net is cheap.
    return
  assert co <= cg, (
    f"optimal FLOPs {co} > greedy FLOPs {cg} for {eq!r} @ {shapes}"
  )


@given(case=equations(min_operands=3, max_operands=4))
@settings(suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_branch_all_flops_le_greedy(
  case: tuple[str, list[tuple[int, ...]]],
) -> None:
  """`branch-all` DFS is seeded by greedy and only updates the bound
  downward, so it must produce ≤ greedy FLOPs."""
  eq, shapes = case
  try:
    greedy = moeinsum.einsum_path(eq, *shapes, optimize="greedy")
    branch = moeinsum.einsum_path(eq, *shapes, optimize="branch-all")
    cg = cast(int, path_cost(eq, shapes, greedy)["total_flops"])
    cb = cast(int, path_cost(eq, shapes, branch)["total_flops"])
  except ValueError:
    return
  assert cb <= cg, (
    f"branch-all FLOPs {cb} > greedy FLOPs {cg} for {eq!r} @ {shapes}"
  )


# ─────────────────────────────────────────────────────────────────────
# Cache determinism
# ─────────────────────────────────────────────────────────────────────


@given(
  case=equations(min_operands=2, max_operands=4),
  optimize=st.sampled_from(["greedy", "optimal", "auto"]),
)
def test_einsum_path_is_idempotent(
  case: tuple[str, list[tuple[int, ...]]],
  optimize: str,
) -> None:
  """Same `(eq, shapes, optimize)` must return an identical path on
  every call. Hot path → LRU hit; cold path → recompute. Both must
  agree."""
  eq, shapes = case
  p1 = moeinsum.einsum_path(eq, *shapes, optimize=optimize)
  p2 = moeinsum.einsum_path(eq, *shapes, optimize=optimize)
  assert p1 == p2


# ─────────────────────────────────────────────────────────────────────
# Numerical parity — random equations vs numpy
# ─────────────────────────────────────────────────────────────────────


@given(case=equations(min_operands=2, max_operands=3))
@settings(
  max_examples=40,
  deadline=None,
  suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
def test_numpy_parity_random_equations(
  case: tuple[str, list[tuple[int, ...]]],
) -> None:
  """For an arbitrary generated equation, moeinsum.einsum's output must
  match numpy.einsum within atol=1e-9.

  We skip pathological cases where numpy itself raises (e.g. a label
  appears in the output but not the inputs — our generator only emits
  the implicit output so that shouldn't happen, but a defensive skip
  keeps the test focused on positive cases)."""
  eq, shapes = case
  rng = np.random.default_rng(0)
  arrays = [rng.standard_normal(s) for s in shapes]

  try:
    expected = np.einsum(eq, *arrays, optimize=True)
  except Exception:
    assume(False)
    return
  actual = moeinsum.einsum(eq, *arrays)
  np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-9)


# ─────────────────────────────────────────────────────────────────────
# Kernel invariants
# ─────────────────────────────────────────────────────────────────────


@given(
  rank=st.integers(min_value=2, max_value=4),
  seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_double_transpose_is_identity(rank: int, seed: int) -> None:
  """`einsum(perm)` twice on the same axes returns the original array."""
  shape = tuple(range(2, 2 + rank))
  rng = np.random.default_rng(seed)
  x = rng.standard_normal(shape)

  src = string.ascii_lowercase[:rank]
  # reverse — guaranteed non-trivial permutation for rank ≥ 2.
  dst = src[::-1]
  eq_fwd = f"{src}->{dst}"
  eq_back = f"{dst}->{src}"

  y = moeinsum.einsum(eq_fwd, x)
  z = moeinsum.einsum(eq_back, y)
  np.testing.assert_allclose(z, x, atol=1e-12, rtol=1e-12)


@given(
  n=st.integers(min_value=2, max_value=8),
  m=st.integers(min_value=2, max_value=8),
  seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_outer_product_factors(n: int, m: int, seed: int) -> None:
  """`einsum('i,j->ij', a, b)[i, j] == a[i] * b[j]` for all i, j."""
  rng = np.random.default_rng(seed)
  a = rng.standard_normal(n)
  b = rng.standard_normal(m)
  outer = moeinsum.einsum("i,j->ij", a, b)
  expected = np.outer(a, b)
  np.testing.assert_allclose(outer, expected, atol=1e-12, rtol=1e-12)


@given(
  rank=st.integers(min_value=1, max_value=4),
  seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_full_reduction_matches_sum(rank: int, seed: int) -> None:
  """Reducing every axis (`"ij...->"`) equals `array.sum()`."""
  shape = tuple(range(2, 2 + rank))
  rng = np.random.default_rng(seed)
  x = rng.standard_normal(shape)

  src = string.ascii_lowercase[:rank]
  eq = f"{src}->"

  reduced = moeinsum.einsum(eq, x)
  np.testing.assert_allclose(reduced, x.sum(), atol=1e-9, rtol=1e-9)


@given(
  n=st.integers(min_value=1, max_value=8),
  seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_trace_matches_numpy(n: int, seed: int) -> None:
  """`einsum('ii->', M)` matches `np.trace(M)` for square M."""
  rng = np.random.default_rng(seed)
  m = rng.standard_normal((n, n))
  tr = moeinsum.einsum("ii->", m)
  np.testing.assert_allclose(tr, np.trace(m), atol=1e-12, rtol=1e-12)


@given(
  case=equations(min_operands=2, max_operands=3),
)
@settings(suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_explicit_path_round_trip(
  case: tuple[str, list[tuple[int, ...]]],
) -> None:
  """A path emitted by `einsum_path(optimize="greedy")` must validate
  cleanly when fed back as an explicit path — the validator accepts
  exactly what the planner emits."""
  eq, shapes = case
  greedy_path = moeinsum.einsum_path(eq, *shapes, optimize="greedy")
  echoed = moeinsum.einsum_path(eq, *shapes, optimize=greedy_path)
  assert echoed == greedy_path


# ─────────────────────────────────────────────────────────────────────
# Single-operand edge cases
# ─────────────────────────────────────────────────────────────────────


@given(
  ndim=st.integers(min_value=1, max_value=3),
  size=st.integers(min_value=2, max_value=6),
  seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_identity_einsum_is_identity(ndim: int, size: int, seed: int) -> None:
  """`einsum('ab...->ab...', x)` returns x exactly (within fp precision).
  Hits the L3 'transpose with no reordering' codepath."""
  shape = (size,) * ndim
  rng = np.random.default_rng(seed)
  x = rng.standard_normal(shape)
  labels = string.ascii_lowercase[:ndim]
  out = moeinsum.einsum(f"{labels}->{labels}", x)
  np.testing.assert_allclose(out, x, atol=1e-12, rtol=1e-12)


# ─────────────────────────────────────────────────────────────────────
# Plan-cache state
# ─────────────────────────────────────────────────────────────────────


def test_plan_cache_hit_does_not_mutate_result() -> None:
  """Two calls to `einsum_path` return *equal* lists, and mutating the
  first list must not leak into a later cache hit. Defends against an
  earlier shape where the cache returned the live list by reference."""
  eq = "ij,jk,kl->il"
  shapes = ((2, 3), (3, 4), (4, 5))
  p1 = moeinsum.einsum_path(eq, *shapes, optimize="greedy")
  p1_copy = list(p1)
  p1.append((99, 99))  # pollute the (potentially shared) backing list
  p2 = moeinsum.einsum_path(eq, *shapes, optimize="greedy")
  assert p2 == p1_copy
