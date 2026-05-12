"""Single-operand kernels.

Implements the four unary ops a `UnaryStep` can describe:
  - REDUCE_SUM: sum out a set of axes.
  - DIAGONAL: repeated-label gather via stride summation.
  - TRACE: DIAGONAL followed by REDUCE_SUM.
  - TRANSPOSE: pure label permutation, layout-only and zero-copy.

These kernels operate on flat Float64 buffers with explicit shapes and element
strides. `diagonal_view` and `transpose_view` are pure metadata: they describe
the same underlying buffer with a different axis interpretation. `reduce_sum_axes`
is the only op here that walks data and writes an output buffer.
"""

from std.memory import UnsafePointer


@fieldwise_init
struct ShapeStrides(Copyable, Movable):
    """A shape + strides pair, both in elements. Result type for the
    metadata-only view ops (`transpose_view`, `diagonal_view`)."""

    var shape: List[Int]
    var strides: List[Int]


# ---------------------------------------------------------------------
# TRANSPOSE - pure layout permutation, zero copy
# ---------------------------------------------------------------------


def transpose_view(
    in_shape: List[Int],
    in_strides: List[Int],
    permutation: List[Int],
) raises -> ShapeStrides:
    """Compute a view of the input with axes permuted.

    `permutation[k] = src` means axis k of the output corresponds to
    axis `src` of the input. No data movement; the caller passes the
    same underlying buffer pointer to the next step.
    """
    var n = len(in_shape)
    if len(permutation) != n:
        raise Error(
            String(
                "transpose_view: permutation length ",
                len(permutation),
                " != input rank ",
                n,
            )
        )
    var out_shape = List[Int]()
    var out_strides = List[Int]()
    for k in range(n):
        var src = permutation[k]
        if src < 0 or src >= n:
            raise Error(
                String(
                    "transpose_view: permutation[",
                    k,
                    "] = ",
                    src,
                    " out of range",
                )
            )
        out_shape.append(in_shape[src])
        out_strides.append(in_strides[src])
    return ShapeStrides(out_shape^, out_strides^)


# ---------------------------------------------------------------------
# DIAGONAL - stride summation across repeated axes
# ---------------------------------------------------------------------


def diagonal_view(
    in_shape: List[Int],
    in_strides: List[Int],
    diag_groups: List[List[Int]],
) raises -> ShapeStrides:
    """Build a view representing the diagonal across `diag_groups`.

    Each inner list of `diag_groups` is a set of axis positions whose
    labels are equal - the diagonal sets `index_i == index_j == ...`
    for those axes. All axes in one group must have the same size.

    The first axis position from each group survives with stride equal
    to the sum of strides of all axes in that group. All other axes
    pass through unchanged.

    Stride math generalizes NumPy's `n+1` trick: a 2D diagonal has
    stride `row_stride + col_stride`, which is `n+1` only when row
    stride equals `n` (contiguous square). For non-contiguous inputs
    the formula still holds; PyTorch's historical #21760 bug was
    forgetting this generality.
    """
    # Which axes are consumed by being collapsed into another.
    var collapsed = List[Bool]()
    var added_stride = List[Int]()
    for _ in range(len(in_shape)):
        collapsed.append(False)
        added_stride.append(0)

    for g_idx in range(len(diag_groups)):
        ref group = diag_groups[g_idx]
        if len(group) < 2:
            continue
        var head = group[0]
        var ref_size = in_shape[head]
        for k in range(1, len(group)):
            var other = group[k]
            if in_shape[other] != ref_size:
                raise Error(
                    String(
                        "diagonal_view: axes ",
                        head,
                        " and ",
                        other,
                        " in same group have shapes ",
                        ref_size,
                        " vs ",
                        in_shape[other],
                    )
                )
            collapsed[other] = True
            added_stride[head] += in_strides[other]

    var out_shape = List[Int]()
    var out_strides = List[Int]()
    for axis in range(len(in_shape)):
        if collapsed[axis]:
            continue
        out_shape.append(in_shape[axis])
        out_strides.append(in_strides[axis] + added_stride[axis])
    return ShapeStrides(out_shape^, out_strides^)


# ---------------------------------------------------------------------
# REDUCE_SUM - walk and accumulate
# ---------------------------------------------------------------------


def reduce_sum_axes(
    in_ptr: UnsafePointer[Float64, MutAnyOrigin],
    in_shape: List[Int],
    in_strides: List[Int],
    reduce_axes: List[Int],
    out_ptr: UnsafePointer[Float64, MutAnyOrigin],
    out_strides: List[Int],
) raises:
    """Sum the input over `reduce_axes`, writing into `out_ptr`.

    `out_ptr` must be zero-initialized and large enough to hold the
    output (the input with the reduced axes removed). `out_strides`
    are in elements for the output's row-major layout.

    Two nested mixed-radix counters: outer over kept axes, inner over reduced
    axes. SIMD vectorization across contiguous inner-reduce is a later polish
    item; current form is correct for all stride patterns.
    """
    var is_reduce = List[Bool]()
    for _ in range(len(in_shape)):
        is_reduce.append(False)
    for k in range(len(reduce_axes)):
        is_reduce[reduce_axes[k]] = True

    var kept_axes = List[Int]()
    for axis in range(len(in_shape)):
        if not is_reduce[axis]:
            kept_axes.append(axis)

    var kept_idx = List[Int]()
    for _ in range(len(kept_axes)):
        kept_idx.append(0)

    while True:
        var out_off: Int = 0
        for k in range(len(kept_axes)):
            out_off += kept_idx[k] * out_strides[k]

        var red_idx = List[Int]()
        for _ in range(len(reduce_axes)):
            red_idx.append(0)
        var acc: Float64 = 0.0
        while True:
            var in_off: Int = 0
            var kc: Int = 0
            var rc: Int = 0
            for axis in range(len(in_shape)):
                if is_reduce[axis]:
                    in_off += red_idx[rc] * in_strides[axis]
                    rc += 1
                else:
                    in_off += kept_idx[kc] * in_strides[axis]
                    kc += 1
            acc += in_ptr[in_off]

            var carry = True
            for k in range(len(reduce_axes)):
                if carry:
                    red_idx[k] += 1
                    if red_idx[k] < in_shape[reduce_axes[k]]:
                        carry = False
                    else:
                        red_idx[k] = 0
            if carry:
                break

        out_ptr[out_off] += acc

        var carry = True
        for k in range(len(kept_axes)):
            if carry:
                kept_idx[k] += 1
                if kept_idx[k] < in_shape[kept_axes[k]]:
                    carry = False
                else:
                    kept_idx[k] = 0
        if carry:
            break


# TRACE = diagonal_view -> reduce_sum_axes. The backend composes these two steps
# from a single UnaryStep with both `diag_axes` and `reduce_axes` populated, so no
# separate trace kernel is needed.
