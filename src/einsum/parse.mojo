"""Einsum equation parser.

Converts equation strings like `"ij,jk->ik"` into a structured `EinsumEquation`
IR consumed by the path optimizer (`path.mojo`) and plan builder (`plan.mojo`).

Design notes:
  - Labels are interned as small `Int`s.
    - opt_einsum and PyTorch use chars and inherit the 52-char (a-zA-Z) limit.
    - We pay one `Int` per label and lift the cap.
  - Ellipsis (`...`) parses to the sentinel `ELLIPSIS_LABEL`.
    - `expand_ellipsis(eq, operand_ranks)` substitutes fresh labels in a
      separate pass once the operand ranks are known.
  - Missing output (`"ij,jk"` without `->`) follows NumPy's convention:
    - output is the sorted unique labels that appear exactly once across all
      inputs, with the ellipsis (if any) first.

Supported syntax:
  - basic            "ij,jk->ik"
  - implicit output  "ij,jk"
  - ellipsis         "...ij,jk->...ik"
  - trace            "ii->"
  - diagonal         "ii->i"
  - sum              "i->"
  - transpose        "ij->ji"
  - whitespace inside the equation is stripped.

Not handled here:
  - shape validation,
  - size-1 broadcast resolution,
  - backend-specific lowering.
"""

# `String`, `StringSlice`, `List`, `chr`, and `ord` are in the Mojo prelude.


comptime ELLIPSIS_LABEL: Int = -1
"""Sentinel for the `...` token. Expanded to fresh labels in `expand_ellipsis`."""


@fieldwise_init
struct EinsumEquation(Copyable, Movable):
    """Parsed einsum equation in canonical IR form.

    Fields:
        inputs: per-operand interned label sequences. `ELLIPSIS_LABEL` marks an
            unexpanded ellipsis token until `expand_ellipsis` substitutes it.
        output: output label sequence in the same encoding.
        n_labels: count of distinct interned labels.
        label_chars: label int to debug-display character.
        has_explicit_output: true if the equation contained `->`.
    """

    var inputs: List[List[Int]]
    var output: List[Int]
    var n_labels: Int
    var label_chars: List[String]
    var has_explicit_output: Bool

    def n_operands(self) -> Int:
        return len(self.inputs)


def _is_label_byte(c: UInt8) -> Bool:
    """True for ASCII letters a-zA-Z."""
    return (c >= UInt8(ord("a")) and c <= UInt8(ord("z"))) or (
        c >= UInt8(ord("A")) and c <= UInt8(ord("Z"))
    )


def _strip_whitespace(s: String) -> String:
    """Return `s` with ASCII space / tab / newline stripped."""
    var out = String()
    var bytes = s.as_bytes()
    for i in range(len(bytes)):
        var c = bytes[i]
        if c != UInt8(ord(" ")) and c != UInt8(ord("\t")) and c != UInt8(
            ord("\n")
        ):
            out += chr(Int(c))
    return out^


def _tokenize_operand(
    s: String,
    mut intern: List[Int],
    mut label_chars: List[String],
    operand_idx: Int,
) raises -> List[Int]:
    """Convert one operand's label string to a list of interned label ints.

    `intern[c]` is the assigned label for ASCII char code `c`, or -2 if
    unassigned. We use -2 as the "unassigned" marker because -1 is the
    ellipsis sentinel.
    """
    var labels = List[Int]()
    var bytes = s.as_bytes()
    var n = len(bytes)
    var i = 0
    while i < n:
        var c = bytes[i]
        # ellipsis: "..."
        if c == UInt8(ord(".")):
            if i + 2 >= n or bytes[i + 1] != UInt8(ord(".")) or bytes[
                i + 2
            ] != UInt8(ord(".")):
                raise Error(
                    String(
                        "einsum parse error in operand ",
                        operand_idx,
                        ": found single '.', expected '...'",
                    )
                )
            for j in range(len(labels)):
                if labels[j] == ELLIPSIS_LABEL:
                    raise Error(
                        String(
                            "einsum parse error in operand ",
                            operand_idx,
                            ": multiple '...' tokens",
                        )
                    )
            labels.append(ELLIPSIS_LABEL)
            i += 3
            continue
        if not _is_label_byte(c):
            raise Error(
                String(
                    "einsum parse error in operand ",
                    operand_idx,
                    ": unexpected character '",
                    chr(Int(c)),
                    "' (only a-zA-Z, '.', ',', '->' allowed)",
                )
            )
        var ci = Int(c)
        if intern[ci] == -2:
            intern[ci] = len(label_chars)
            label_chars.append(chr(ci))
        labels.append(intern[ci])
        i += 1
    return labels^


def parse(eq_in: String) raises -> EinsumEquation:
    """Parse an einsum equation string into an `EinsumEquation`."""
    var eq = _strip_whitespace(eq_in)

    # Split input vs output on "->" via byte-find then byte-slice.
    var arrow_idx = eq.find("->")
    var input_part: String
    var output_part: String
    var has_explicit_output: Bool
    if arrow_idx >= 0:
        input_part = String(StringSlice(eq)[byte=0:arrow_idx])
        output_part = String(
            StringSlice(eq)[byte=arrow_idx + 2 : len(eq.as_bytes())]
        )
        has_explicit_output = True
    else:
        input_part = eq
        output_part = String()
        has_explicit_output = False

    # Intern ASCII labels. -2 = unassigned, -1 is the ellipsis sentinel.
    var intern = List[Int]()
    for _ in range(128):
        intern.append(-2)
    var label_chars = List[String]()

    # Tokenize each operand. `split` returns StringSlices over `input_part`;
    # copy each into a String for the helper.
    var operand_slices = input_part.split(",")
    var inputs = List[List[Int]]()
    for i in range(len(operand_slices)):
        var op_string = String(operand_slices[i])
        var labels = _tokenize_operand(op_string, intern, label_chars, i)
        inputs.append(labels^)

    # Tokenize output, or compute implicit output.
    var output = List[Int]()
    if has_explicit_output:
        output = _tokenize_operand(output_part, intern, label_chars, -1)
    else:
        var counts = List[Int]()
        for _ in range(len(label_chars)):
            counts.append(0)
        var has_ellipsis = False
        for op_idx in range(len(inputs)):
            for lbl_idx in range(len(inputs[op_idx])):
                var lbl = inputs[op_idx][lbl_idx]
                if lbl == ELLIPSIS_LABEL:
                    has_ellipsis = True
                else:
                    counts[lbl] += 1
        if has_ellipsis:
            output.append(ELLIPSIS_LABEL)
        # ASCII order: walk char codes, emit labels that appear once.
        for ci in range(128):
            var lbl = intern[ci]
            if lbl >= 0 and counts[lbl] == 1:
                output.append(lbl)

    return EinsumEquation(
        inputs^,
        output^,
        len(label_chars),
        label_chars^,
        has_explicit_output,
    )


def expand_ellipsis(
    mut eq: EinsumEquation, operand_ranks: List[Int]
) raises:
    """In-place substitute `ELLIPSIS_LABEL` with fresh labels.

    Ellipsis width per operand is `operand_ranks[i] - explicit_label_count`.
    Shorter explicit ellipses are right-aligned against the full broadcast
    ellipsis, matching NumPy's `...ij,...jk->...ik` behavior.
    """
    if len(operand_ranks) != eq.n_operands():
        raise Error(
            String(
                "expand_ellipsis: got ",
                len(operand_ranks),
                " ranks for ",
                eq.n_operands(),
                " operands",
            )
        )

    var widths = List[Int]()
    var any_ellipsis = False
    var max_width = 0
    for i in range(eq.n_operands()):
        var explicit_count = 0
        var has_ellipsis = False
        ref op = eq.inputs[i]
        for lbl_idx in range(len(op)):
            if op[lbl_idx] == ELLIPSIS_LABEL:
                has_ellipsis = True
            else:
                explicit_count += 1
        if has_ellipsis:
            any_ellipsis = True
            var w = operand_ranks[i] - explicit_count
            if w < 0:
                raise Error(
                    String(
                        "expand_ellipsis: operand ",
                        i,
                        " has rank ",
                        operand_ranks[i],
                        " but ",
                        explicit_count,
                        " explicit labels",
                    )
                )
            widths.append(w)
            if w > max_width:
                max_width = w
        else:
            widths.append(0)
            if operand_ranks[i] != explicit_count:
                raise Error(
                    String(
                        "expand_ellipsis: operand ",
                        i,
                        " has rank ",
                        operand_ranks[i],
                        " but ",
                        explicit_count,
                        " explicit labels (no '...' to absorb the rest)",
                    )
                )

    if not any_ellipsis:
        return

    var first_fresh = eq.n_labels
    for k in range(max_width):
        eq.label_chars.append(String("?", k))
    eq.n_labels += max_width

    var new_inputs = List[List[Int]]()
    for i in range(eq.n_operands()):
        var w = widths[i]
        var out = List[Int]()
        ref op = eq.inputs[i]
        for lbl_idx in range(len(op)):
            var lbl = op[lbl_idx]
            if lbl == ELLIPSIS_LABEL:
                for k in range(w):
                    out.append(first_fresh + (max_width - w) + k)
            else:
                out.append(lbl)
        new_inputs.append(out^)
    eq.inputs = new_inputs^

    var new_output = List[Int]()
    var output_had_ellipsis = False
    for lbl_idx in range(len(eq.output)):
        var lbl = eq.output[lbl_idx]
        if lbl == ELLIPSIS_LABEL:
            output_had_ellipsis = True
            for k in range(max_width):
                new_output.append(first_fresh + k)
        else:
            new_output.append(lbl)
    if any_ellipsis and not eq.has_explicit_output and not output_had_ellipsis:
        var prefixed = List[Int]()
        for k in range(max_width):
            prefixed.append(first_fresh + k)
        for lbl_idx in range(len(new_output)):
            prefixed.append(new_output[lbl_idx])
        new_output = prefixed^
    eq.output = new_output^
