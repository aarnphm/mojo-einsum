"""Reference backend — naive global-index loop.

The simplest correct einsum: for every assignment of values to the union
of all input labels, multiply the indexed input scalars, and accumulate
into the indexed output position. Quadratic in the worst case (or worse
— exponential in label count), but it always produces the right answer.
That's the entire point: this is the golden against which every other
backend is regression-tested.

This backend operates on flat `Float64` buffers + explicit shapes /
strides. It does *not* depend on `TileTensor` or any MAX kernel, so it
is fully self-contained — useful for testing the parser, plan builder,
and dim classification in isolation from the MAX integration.

Memory model:
  - Each operand is a strided buffer over `Float64`.
  - `strides[i]` is in elements, not bytes (matches NumPy conventions).
  - Output buffer is pre-allocated by the caller; zero-initialized
    before this function runs.

Implementation notes:
  - We walk a global index vector of length `n_labels`, where slot `k`
    iterates over the size of label `k`. Total iterations = product of
    label sizes. For BMM-shaped contractions this is O(B*M*N*K) — the
    correct FLOP count.
  - For each global index, project to per-operand indices via that
    operand's label list, look up the scalar, multiply across operands.
    Project to the output index via `eq.output` and accumulate.
  - Operates on `Float64` for v0.1 simplicity; generalization to other
    dtypes lifts to a `@parameter` over `DType` later (P9).
"""

from std.collections import List
from std.memory import UnsafePointer
from einsum.parse import EinsumEquation


def _resolve_label_sizes(
    eq: EinsumEquation,
    operand_shapes: List[List[Int]],
) raises -> List[Int]:
    """Build a List[Int] of size `eq.n_labels` mapping label -> size.

    Raises if the same label is given inconsistent sizes across operands
    (other than the size-1 broadcast case, which we surface as an error
    here and let the caller decide — reference is strict by design).
    """
    var sizes = List[Int]()
    for _ in range(eq.n_labels):
        sizes.append(-1)

    for op_idx in range(eq.n_operands()):
        ref labels = eq.inputs[op_idx]
        ref shape = operand_shapes[op_idx]
        if len(labels) != len(shape):
            raise Error(
                String(
                    "_resolve_label_sizes: operand ",
                    op_idx,
                    " has ",
                    len(labels),
                    " labels but ",
                    len(shape),
                    " shape dims",
                )
            )
        for axis in range(len(labels)):
            var lbl = labels[axis]
            var dim = shape[axis]
            if sizes[lbl] == -1:
                sizes[lbl] = dim
            elif sizes[lbl] != dim:
                raise Error(
                    String(
                        "size mismatch on label ",
                        lbl,
                        ": operand ",
                        op_idx,
                        " axis ",
                        axis,
                        " has size ",
                        dim,
                        ", expected ",
                        sizes[lbl],
                    )
                )

    # Any label that survived only in the output (not in any input) is
    # an error — there's no source to read from.
    for lbl in eq.output:
        if sizes[lbl] == -1:
            raise Error(
                String(
                    "output label ",
                    lbl,
                    " does not appear in any input",
                )
            )

    return sizes^


def _flat_offset(
    label_indices: List[Int],
    operand_labels: List[Int],
    strides: List[Int],
) -> Int:
    """Compute the flat-element offset into one operand from the current
    global label-index vector."""
    var off: Int = 0
    for axis in range(len(operand_labels)):
        var lbl = operand_labels[axis]
        off += label_indices[lbl] * strides[axis]
    return off


def execute_reference(
    eq: EinsumEquation,
    operand_data: List[UnsafePointer[Float64, MutAnyOrigin]],
    operand_shapes: List[List[Int]],
    operand_strides: List[List[Int]],
    output_data: UnsafePointer[Float64, MutAnyOrigin],
    output_strides: List[Int],
) raises -> None:
    """Reference einsum: brute-force walk of every label assignment.

    `output_data` must be zero-initialized and large enough to hold the
    product of label sizes for labels in `eq.output`.
    """
    var label_sizes = _resolve_label_sizes(eq, operand_shapes)
    var n_labels = eq.n_labels

    # Output rank can be zero (scalar-result einsum like 'ii->').
    var out_rank = len(eq.output)

    # Global-index iteration state.
    var label_idx = List[Int]()
    for _ in range(n_labels):
        label_idx.append(0)

    while True:
        # Compute product over operands.
        var prod: Float64 = 1.0
        for op_idx in range(eq.n_operands()):
            var off = _flat_offset(label_idx, eq.inputs[op_idx], operand_strides[op_idx])
            prod *= operand_data[op_idx][off]

        # Accumulate into output.
        var out_off: Int = 0
        for axis in range(out_rank):
            out_off += label_idx[eq.output[axis]] * output_strides[axis]
        output_data[out_off] += prod

        # Increment label_idx as a mixed-radix counter. Carries through
        # labels in ascending order; terminate when MSB rolls over.
        var k = 0
        while k < n_labels:
            label_idx[k] += 1
            if label_idx[k] < label_sizes[k]:
                break
            label_idx[k] = 0
            k += 1
        if k == n_labels:
            break


def compute_output_shape(
    eq: EinsumEquation,
    operand_shapes: List[List[Int]],
) raises -> List[Int]:
    """Convenience: given an equation + operand shapes, compute the
    output shape (in `eq.output` label order)."""
    var sizes = _resolve_label_sizes(eq, operand_shapes)
    var out = List[Int]()
    for lbl in eq.output:
        out.append(sizes[lbl])
    return out^
