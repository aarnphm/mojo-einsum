---
title: FFI design for max backend
date: 2026/05/11
---

P5 unblocks the rest of v0.1 - P10 (GPU dispatch), P11 (native CPU GETT),
P12 (native GPU SM90), and the DLPack polish for P8 all sit downstream of
the same FFI rewire. Today's FFI copies fp64 lists in and out of
`einsum_reference`; MAX kernels want `TileTensor` over an unowned pointer,
and that mismatch is the only thing standing between the current tree and
a working `backend="max"`.

## Today's FFI surface

`src/lib.mojo` exposes four `PythonModuleBuilder` entries:

| Function                       | Operand contract                | Return                       |
| ------------------------------ | ------------------------------- | ---------------------------- |
| `parse_equation(eq)`           | pure equation parse             | `dict[str, object]`          |
| `einsum_reference(eq, fl, sh)` | flat fp64 buffers + shape lists | `(flat_out_fp64, out_shape)` |
| `einsum_path(eq, sh)`          | shape lists                     | `list[tuple[int, ...]]`      |
| `einsum_compute_path(eq, ...)`   | shape lists + algorithm name    | `list[tuple[int, ...]]`      |

`einsum_reference` allocates an `UnsafePointer[Float64]` per operand and
memcpys the Python list in. End-to-end ownership makes that fine for the
reference backend. For `MaxBackend` the same copy is a per-call tax on
buffers the caller already owns through DLPack - we want the caller's
pointer to land inside a `TileTensor` directly.

## New FFI entry - `einsum_max_py`

```mojo
def einsum_max_py(
    eq_obj: PythonObject,
    operand_caps_obj: PythonObject,    # list[capsule] - DLPack v1 tensors
    operand_shapes_obj: PythonObject,
    operand_strides_obj: PythonObject,
    operand_dtypes_obj: PythonObject,  # list[str] - "float32", "bfloat16", ...
    optimize_obj: PythonObject,
    backend_obj: PythonObject,         # "max" | "max:gpu" | ...
    accum_dtype_obj: PythonObject,
) raises -> PythonObject
```

Capsule-first means numpy / torch / jax / mlx fall out symmetrically once
`_interop.to_dlpack` exists (see below - it doesn't yet). For P5 we ship
numpy-only and lift through `numpy.dlpack`; that's a strict subset and
unblocks acceptance.

## `TileTensor` handoff - the load-bearing step

```mojo
# From ~/workspace/modular/max/kernels/src/layout/{tile_tensor,runtime_layout}.mojo
var runtime_shape  = make_runtime_shape(S)
var runtime_stride = make_runtime_stride(T)
var layout = RuntimeLayout(runtime_shape, runtime_stride)
var tile = TileTensor[DType.float32, rank=N, all_dims_known=False](
    operand_ptr, layout
)
```

`rank` is a comptime parameter - that is the one hard constraint. We
monomorphize over `rank in {1, ..., 6}` via `@parameter for r in range(1, 7)`
at the FFI boundary; ML einsums above rank 6 are rare enough that a cap
beats a Mojo-side specialization registry.

**TBD - should the rank cap stay at 6, or grow dynamically through a
generated registry?** Cap is one `@parameter for` block; registry is a
build-system change. Decision needed before P5 code lands.

## Pairwise-step lowering - four steps

For each `PairwiseStep`:

1. Permute `lhs_tile` into `(*B, *M, *K)` via
   `step.batch_axes_lhs ++ step.free_axes_lhs ++ step.contract_axes_lhs`.
   Contiguous strides -> `Layout` composition, zero-copy. Otherwise ->
   TTGT materialize into an arena buffer.
2. Same for `rhs_tile` -> `(*B, *K, *N)`.
3. `linalg.batched_matmul[transpose_a=False, transpose_b=False](out_view, lhs_view, rhs_view, ctx)`.
4. If `step.out_permutation != identity`, transpose into the working-set
   buffer.

JAX's `lax_numpy.py:3288-3300` swaps `(lhs, rhs)` when that lets the
BMM-natural output order match `out_labels` and elides step 4. ~30 lines
at the top of `_execute_pairwise`; free win.

## Unary-step lowering

`src/einsum/unary.mojo` already implements the kernels. The change is
plumbing - accept `TileTensor` instead of `(UnsafePointer, shape, stride)`:

- `transpose_view` - new `Layout`, no kernel
- `diagonal_view` - new `Layout` with stride summation, no kernel
- `reduce_sum_axes` - read from `tile.ptr()` / `tile.layout`, dispatch the
  existing SIMD reduce
- `trace` = `diagonal_view` $\circ$ `reduce_sum_axes`

## Build-config change

```toml
[tool.mohaus]
mojo-include-paths = [
    "~/workspace/modular/max/kernels/src",
]
```

`mohaus develop` then exposes `linalg.batched_matmul`, `TileTensor`,
`RuntimeLayout` as plain imports inside
`src/einsum/backends/max.mojo`. The last attempt hit an unrelated MLIR
parse error in the in-tree mojo stdlib at `~/workspace/modular`; that's
an upstream blocker, not a moeinsum bug. Fix is either pinning
`mojo-compiler` or waiting for the stdlib to settle.

## Python-side wiring

`python/moeinsum/__init__.py` already takes a `backend` parameter. Two
edits flip dispatch:

```python
_BACKENDS = ("reference", "max")  # was ("reference",)

if backend == "max":
    return _einsum_max_native(
        eq,
        [_to_dlpack(o) for o in operands],
        [list(o.shape) for o in operands],
        [list(o.strides) for o in operands],
        [str(o.dtype) for o in operands],
        optimize_str,
        target_or_default,
        accum_dtype_str,
    )
```

`_interop.to_numpy` exists; `_interop.to_dlpack` does **not** - needs to
be added. Three lines around `numpy.ndarray.__dlpack__()` plus the
inverse on the return path.

## Test plan

The harness already catches a cutover regression once `"max"` joins the
parametrize lists:

- `tests/python/test_numpy_parity.py` - every case parametrizes over
  available backends. Adding `"max"` expects bit-identical (integer
  dtypes) / atol-equivalent (float) vs. reference.
- `tests/python/test_backend_stubs.py` - flip the
  `pytest.raises(ImportError, ...)` guarding `MaxGraphBackend()` into a
  positive test once MAX is installed; mirror for a new
  `tests/python/test_max_backend.py` pinning matmul / BMM / 3-operand
  chain shapes.
- `tests/python/test_property.py` - the hypothesis suite (transpose
  involution, outer-product, full-sum, ~14 invariants total) runs against
  every backend in the parametrize list. Property failures land before
  parity failures for any layout / permute / dim-classification bug.

## Phase ladder

**P5 - MaxBackend wiring (2 days when MLIR unblocked):**

1. Add `mojo-include-paths` to `pyproject.toml`.
2. Implement `einsum_max_py` in `src/lib.mojo` with the rank
   monomorphization block.
3. Implement `_execute_pairwise(step, lhs_tile, rhs_tile, ...)` in
   `src/einsum/backends/max.mojo` per the four-step recipe.
4. Implement `_execute_unary(step, in_tile, out_tile)`.
5. Add `"max"` to `_BACKENDS` in `python/moeinsum/__init__.py`; dispatch
   `einsum(backend="max", ...)` to the new FFI.
6. Add `_interop.to_dlpack`.
7. Run `tests/python/test_numpy_parity.py` with `backend="max"`.

**P10 - GPU dispatch validation (<=1 day after P5):**

1. `target="gpu"` flows through `linalg.batched_matmul`'s SM90 / SM100 /
   Apple path - no code change.
2. CPU/GPU equivalence: atol=1e-5 (fp32) / 1e-2 (bf16).
3. `python -m moeinsum.bench` happy-path on a GPU host.

**P11 - Native CPU GETT (3 days):**

1. `src/einsum/backends/native.mojo` skeleton on the same trait as
   `MaxBackend`.
2. BLIS-style packing with index re-mapping during pack - TBLIS approach,
   `arxiv 1607.00291`.
3. Acceptance: beats `MaxBackend` on permute-dominated TTGT, e.g.
   `abcd,dcba->` square shapes.

**P12 - Native GPU SM90 GETT (4 days):**

1. WGMMA-issuing kernel, TMA tile loads, permute fused into shared-memory
   pack.
2. `TensorCoreAsync[c_type, a_type, b_type, mma_shape=Index(64,128,16)]`
   with `warpgroup_fence()` before AND after (per mojo-perf skill).
3. Acceptance: >= `linalg.batched_matmul` throughput on contractions where
   permute-into-pack pays for itself; atol=1e-5 (fp32) / 1e-2 (bf16) vs.
   `MaxBackend`.

**DLPack zero-copy polish (1/2 day after P5):**

1. `_interop.to_dlpack` for numpy / torch / jax / mlx. Receive-side
   (`to_numpy`) is dtype-preserving and ships; send-side does not.
2. `einsum(...)` already round-trips via `_from_numpy(kind)`, so the
   caller's framework / dtype survive the call. The missing piece is the
   zero-copy edge - today each operand flows through a numpy intermediate
   at the FFI boundary.
3. Tests: torch <-> jax <-> mlx round-trip + dtype preservation already
   pin in `tests/python/test_interop.py` and
   `test_property.py::test_einsum_preserves_*`. The zero-copy assertion
   waits on `to_dlpack`.

## Where the tree stands

- `src/einsum/backends/max.mojo` ships the structural skeleton with the
  four-step recipe in comments; the Mojo stub raises a "Phase 5 work"
  error.
- `python/moeinsum/_max_graph.py` ships the Python-side plan-to-graph
  translation (P14). `classify_pair(...)` and `plan_to_graph_spec(...)`
  are the dim-classifier and op-list emission - both `MaxBackend` (P5)
  and `MaxGraphBackend` (P14) consume them, emitting
  `linalg.batched_matmul` calls vs. `max.graph.ops.matmul` ops
  respectively.
- Hypothesis suite + plan-cache snapshot fix mean dim classification,
  path optimization, plan IR, and the test harness are all done. The FFI
  boundary is the only place that has to change.
