"""Reference backend: a global-index loop used as the correctness oracle.

For every assignment of values to the union of all input labels, multiply the
indexed input scalars and accumulate into the indexed output position. This is
not the fast path, but it is the golden backend every optimized lowering is
regression-tested against.

This backend operates on flat `Float64` buffers + explicit shapes /
strides. It does not depend on `TileTensor` or MAX kernels, so parser, planner,
and dimension-classification tests can run without MAX integration.

Memory model:
  - Each operand is a strided buffer over `Float64`.
  - `strides[i]` is in elements, not bytes (matches NumPy conventions).
  - Output buffer is pre-allocated by the caller; zero-initialized
    before this function runs.

Implementation notes:
  - We walk a global index vector of length `n_labels`, where slot `k`
    iterates over the size of label `k`. Total iterations = product of
    label sizes. For BMM-shaped contractions this is O(B*M*N*K) - the
    correct FLOP count.
  - For each global index, project to per-operand indices via that
    operand's label list, look up the scalar, multiply across operands.
    Project to the output index via `eq.output` and accumulate.
  - Operates on `Float64`; dtype handling is done at the Python boundary for
    v0.1. Generalization to other dtypes lifts to a parameter over `DType`
    later.
"""

from std.collections import List
from std.memory import UnsafePointer
from einsum.parse import EinsumEquation


def _resolve_label_sizes(
    eq: EinsumEquation,
    operand_shapes: List[List[Int]],
) raises -> List[Int]:
    """Build a List[Int] of size `eq.n_labels` mapping label -> size.

    Cross-operand size resolution is broadcast-aware: when one operand
    has dim=1 and another has dim=N on the same label, the resolved
    size is N (numpy's per-label broadcast). Within a single operand
    we keep strict equality - repeated labels like `ii->` are a
    diagonal-extraction, never a broadcast (numpy itself rejects
    `np.einsum('ii->', (1, 3))`).
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

        # Within-operand strict pass: repeated labels must share a size. Repeated
        # labels select a diagonal, never a broadcast.
        var local = List[Int]()
        for _ in range(eq.n_labels):
            local.append(-1)
        for axis in range(len(labels)):
            var lbl = labels[axis]
            var dim = shape[axis]
            if local[lbl] == -1:
                local[lbl] = dim
            elif local[lbl] != dim:
                raise Error(
                    String(
                        "size mismatch on label ",
                        lbl,
                        " within operand ",
                        op_idx,
                        ": axis ",
                        axis,
                        " has size ",
                        dim,
                        ", expected ",
                        local[lbl],
                    )
                )

        # Cross-operand merge with size-1 broadcast: (1, N) resolves to N;
        # (M, N) with M != N and both > 1 is a real conflict.
        for lbl in range(eq.n_labels):
            var dim = local[lbl]
            if dim == -1:
                continue
            var prev = sizes[lbl]
            if prev == -1:
                sizes[lbl] = dim
            elif prev == dim:
                continue
            elif prev == 1:
                sizes[lbl] = dim
            elif dim == 1:
                continue
            else:
                raise Error(
                    String(
                        "size mismatch on label ",
                        lbl,
                        ": operand ",
                        op_idx,
                        " has size ",
                        dim,
                        ", expected ",
                        prev,
                    )
                )

    # Any label that survived only in the output, not in any input, has no source
    # element to read.
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
    operand_shape: List[Int],
    strides: List[Int],
) -> Int:
    """Compute the flat-element offset into one operand from the current
    global label-index vector.

    Size-1 axes are stride-0 broadcasts: the operand has only one valid
    element along that axis, so the label index contributes nothing to
    the offset regardless of how the resolved label size loops outside.
    """
    var off: Int = 0
    for axis in range(len(operand_labels)):
        if operand_shape[axis] == 1:
            continue
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

    # Output rank can be zero, e.g. scalar-result einsum like 'ii->'.
    var out_rank = len(eq.output)

    # Global-index iteration state.
    var label_idx = List[Int]()
    for _ in range(n_labels):
        label_idx.append(0)

    while True:
        # Compute product over operands.
        var prod: Float64 = 1.0
        for op_idx in range(eq.n_operands()):
            var off = _flat_offset(
                label_idx,
                eq.inputs[op_idx],
                operand_shapes[op_idx],
                operand_strides[op_idx],
            )
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
