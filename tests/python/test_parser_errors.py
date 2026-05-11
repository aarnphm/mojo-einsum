"""Parser-error grammar coverage.

`test_p1_smoke.py` checks two parse-error shapes (invalid chars,
double-dot). This file exhaustively covers the documented error
surface, so a regression in `src/einsum/parse.mojo` lands as a
specific failure instead of an opaque "all parser tests broke".

Each test names the malformed grammar and the expected error fragment.
The Mojo parser raises typed errors that the FFI normalizes to Python
`Exception`s with `"einsum parse error"` in the message.
"""

from __future__ import annotations

import moeinsum
import numpy as np
import pytest


def _expect_parse_error(eq: str, *, fragment: str = "einsum parse error") -> None:
  with pytest.raises(Exception, match=fragment):
    moeinsum.parse_equation(eq)


# ─────────────────────────────────────────────────────────────────────
# Character-class violations
# ─────────────────────────────────────────────────────────────────────


def test_dollar_sign_rejected() -> None:
  _expect_parse_error("i$j,jk->ik")


def test_digit_rejected() -> None:
  _expect_parse_error("i1j,jk->ik")


def test_underscore_rejected() -> None:
  _expect_parse_error("i_j,jk->ik")


def test_punctuation_rejected() -> None:
  _expect_parse_error("i!j,jk->ik")


def test_whitespace_inside_operand_is_lenient() -> None:
  """Current behaviour: whitespace inside an operand is *not* rejected
  — the parser silently ignores it. Documented here so a future
  strict-mode change shows up as a test deltas instead of a regression."""
  ir = moeinsum.parse_equation("i j,jk->ik")
  # i + j + k = 3 labels even with the embedded space.
  assert ir["n_labels"] == 3


# ─────────────────────────────────────────────────────────────────────
# Ellipsis grammar violations
# ─────────────────────────────────────────────────────────────────────


def test_double_dot_rejected() -> None:
  _expect_parse_error("..ij,jk->ik")


def test_single_dot_rejected() -> None:
  _expect_parse_error(".ij,jk->ik")


def test_four_dot_ellipsis_rejected() -> None:
  _expect_parse_error("....ij,jk->ik")


def test_ellipsis_in_middle_then_dot_rejected() -> None:
  _expect_parse_error("i...j..k->ijk")


# ─────────────────────────────────────────────────────────────────────
# Arrow grammar violations
# ─────────────────────────────────────────────────────────────────────


def test_double_arrow_rejected() -> None:
  _expect_parse_error("ij->jk->ik")


def test_reverse_arrow_rejected() -> None:
  _expect_parse_error("ij<-jk")


def test_arrow_without_lhs_is_lenient() -> None:
  """Current behaviour: an empty LHS parses as a single empty operand
  rather than being rejected. The downstream einsum call will fail
  because there are no operands to consume `[0, 1]` as labels."""
  ir = moeinsum.parse_equation("->ik")
  assert ir["inputs"] == [[]]
  assert ir["has_explicit_output"] is True


# ─────────────────────────────────────────────────────────────────────
# Size mismatches (parse + validate)
# ─────────────────────────────────────────────────────────────────────


def test_size_mismatch_caught_at_einsum() -> None:
  """Same label with different sizes across operands must fail —
  the constraint surfaces during einsum() execution, not parse_equation()
  (parsing is structural; size checks need operand shapes)."""
  with pytest.raises(Exception, match="size"):
    moeinsum.einsum("ij,jk->ik", np.eye(3), np.eye(4))


def test_rank_mismatch_caught_at_einsum() -> None:
  """Operand has fewer dims than the equation labels demand."""
  with pytest.raises(Exception):  # noqa: B017
    moeinsum.einsum("ijk,jl->il", np.eye(3), np.eye(3))


# ─────────────────────────────────────────────────────────────────────
# Output-label grammar
# ─────────────────────────────────────────────────────────────────────


def test_output_label_not_in_inputs_rejected() -> None:
  """`ij,jk->iz` — `z` never appears in any input. Implementation
  detail: this may pass `parse_equation` (structural) but fail at
  einsum execution time."""
  with pytest.raises(Exception):  # noqa: B017
    moeinsum.einsum("ij,jk->iz", np.eye(3), np.eye(3))


# ─────────────────────────────────────────────────────────────────────
# Empty / minimal inputs
# ─────────────────────────────────────────────────────────────────────


def test_empty_equation_is_lenient() -> None:
  """Current behaviour: an empty string parses to a degenerate
  zero-label, single-empty-operand IR rather than raising."""
  ir = moeinsum.parse_equation("")
  assert ir["n_labels"] == 0
  assert ir["inputs"] == [[]]
  assert ir["output"] == []


def test_arrow_only_is_lenient() -> None:
  """Current behaviour: `->` parses the same as the empty string but
  with `has_explicit_output=True`."""
  ir = moeinsum.parse_equation("->")
  assert ir["has_explicit_output"] is True
  assert ir["output"] == []


def test_trailing_comma_is_lenient() -> None:
  """Current behaviour: a trailing comma yields an empty third
  operand. NumPy's parser does the same — `np.einsum('ij,jk,->ik', ...)`
  hits the "wrong number of operands" path."""
  ir = moeinsum.parse_equation("ij,jk,->ik")
  assert len(ir["inputs"]) == 3
  assert ir["inputs"][-1] == []


# ─────────────────────────────────────────────────────────────────────
# Valid edge cases (should NOT raise)
# ─────────────────────────────────────────────────────────────────────


def test_single_label_single_operand_parses() -> None:
  """`i` — implicit output is `i`. Valid grammar."""
  ir = moeinsum.parse_equation("i")
  assert ir["n_labels"] == 1


def test_pure_scalar_einsum_parses() -> None:
  """`->` is an empty einsum — no operands, scalar output. Valid
  grammar but the einsum call will reject for needing operands."""
  ir = moeinsum.parse_equation("->")
  assert ir["has_explicit_output"] is True
  assert ir["output"] == []


def test_all_lowercase_labels_parse() -> None:
  ir = moeinsum.parse_equation("abcdefghijklmnopqrstuvwxyz->abcdefghijklmnopqrstuvwxyz")
  assert ir["n_labels"] == 26


def test_all_uppercase_labels_parse() -> None:
  """Uppercase A-Z are valid einsum labels (opt_einsum + numpy convention)."""
  ir = moeinsum.parse_equation("ABCDEFGHIJ->ABCDEFGHIJ")
  assert ir["n_labels"] == 10


def test_mixed_case_labels_distinct() -> None:
  """`Aa` is two distinct labels, not one. NumPy convention."""
  ir = moeinsum.parse_equation("Aa->Aa")
  assert ir["n_labels"] == 2
