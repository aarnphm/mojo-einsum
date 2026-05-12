---
title: FFI design for max backend
date: 2026/05/11
---

P5 is now specifically the Mojo cutover, not the first working MAX path.
The Python MAX Graph backend already makes `backend="max[:cpu|gpu]"`
executable for the BMM-lowerable subset. The remaining FFI rewire matters
because P11 (native CPU GETT), P12 (native GPU SM90), deterministic
low-precision controls, and zero-copy device ownership all want
`TileTensor` over an unowned pointer instead of Python-side NumPy buffers.

## Today's FFI surface

`src/lib.mojo` exposes four `PythonModuleBuilder` entries:

| Function                       | Operand contract                | Return                       |
| ------------------------------ | ------------------------------- | ---------------------------- |
| `parse_equation(eq)`           | pure equation parse             | `dict[str, object]`          |
| `einsum_reference(eq, fl, sh)` | flat fp64 buffers + shape lists | `(flat_out_fp64, out_shape)` |
| `einsum_path(eq, sh)`          | shape lists                     | `list[tuple[int, ...]]`      |
| `einsum_compute_path(eq, ...)` | shape lists + algorithm name    | `list[tuple[int, ...]]`      |

`einsum_reference` allocates an `UnsafePointer[Float64]` per operand and
memcpys the Python list in. End-to-end ownership makes that fine for the
reference backend. The current executable MAX Graph path bypasses this
Mojo FFI and builds a `max.graph.Graph` in Python, then feeds MAX
`Buffer` objects. For the Mojo `MaxBackend` / `native` cutover, that copy
becomes a per-call tax on buffers the caller already owns through DLPack -
we want the caller's pointer to land inside a `TileTensor` directly.

## Future FFI entry - `einsum_max_py`

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

Capsule-first means numpy / torch / jax / mlx fall out symmetrically.
The public input adapter already uses DLPack opportunistically on the
Python side; the missing piece is a send-side capsule path that can cross
the Mojo boundary without detouring through NumPy host buffers.

## `TileTensor` handoff

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

Open decision: keep the rank cap at 6, or generate a wider registry.
The cap is one `@parameter for` block; the registry is a build-system
change.

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
BMM-natural output order match `out_labels` and elides step 4.

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
parse error in the in-tree mojo stdlib at `~/workspace/modular`; this is
an upstream blocker, not a moeinsum bug. Fix is either pinning
`mojo-compiler` or waiting for the stdlib to settle.

## Python-side wiring

`python/moeinsum/__init__.py` already takes a `backend` parameter. Today,
`backend.startswith("max")` dispatches to `_max_backend.execute_max`, the
Python MAX Graph bridge. The Mojo cutover replaces that call with the FFI
entry above:

```python
_BACKENDS = ("reference", "max", "max:cpu", "max:gpu", "max_graph")

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

Behavioral contract: unsupported MAX cases raise `NotImplementedError`;
they do not silently fall back to reference.

## Test plan

The harness already catches the Python MAX Graph surface and should be
expanded, not replaced, when the Mojo cutover lands:

- `tests/python/test_max_backend.py` pins matmul, BMM, 3-operand chains,
  outer product, unary transpose, reduce-sum, size-1 broadcast, and
  ellipsis on the executable MAX Graph path.
- `tests/python/test_max_backend_bf16.py` pins bf16 drift growth and
  dtype round-trip through MAX.
- `tests/python/test_backend_stubs.py` keeps `plan_to_graph_spec`
  independent of `max.graph` and now separately pins the Python
  ellipsis expansion used by the executable MAX bridge.
- `tests/python/test_property.py` remains the broad parser / planner /
  reference invariant suite. When the Mojo backend covers the full
  grammar, add it to the same parity matrix instead of making a weaker
  side harness.

## Phase ladder

**P5 - Mojo MaxBackend wiring (2 days when MLIR unblocked):**

1. Add `mojo-include-paths` to `pyproject.toml`.
2. Implement `einsum_max_py` in `src/lib.mojo` with the rank
   monomorphization block.
3. Implement `_execute_pairwise(step, lhs_tile, rhs_tile, ...)` in
   `src/einsum/backends/max.mojo` per the four-step recipe.
4. Implement `_execute_unary(step, in_tile, out_tile)`.
5. Redirect `einsum(backend="max", ...)` from the Python MAX Graph bridge
   to the new FFI once feature coverage is at least equal.
6. Add `_interop.to_dlpack`.
7. Run `tests/python/test_numpy_parity.py` with `backend="max"`.

**P10 - GPU dispatch validation:**

1. Python MAX Graph path: done for `backend="max:gpu"` on the B200 host.
2. Mojo `target="gpu"`: waits on P5, then repeats CPU/GPU equivalence at
   atol=1e-5 (fp32) / 1e-2 (bf16).
3. `python -m moeinsum.bench` happy-path on a GPU host remains the smoke
   test for both paths.

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

## Current state

- `src/einsum/backends/max.mojo` is a Mojo stub; this doc holds the
  four-step lowering recipe until the TileTensor cutover lands.
- `python/moeinsum/_max_graph.py` ships the Python-side plan-to-graph
  translation (P14). `classify_pair(...)` and `plan_to_graph_spec(...)`
  are the dim-classifier and op-list emission - both `MaxBackend` (P5)
  and `MaxGraphBackend` (P14) consume them, emitting
  `linalg.batched_matmul` calls vs. `max.graph.ops.matmul` ops
  respectively.
- Dim classification, path optimization, plan IR, and the test harness
  have coverage. The remaining work is the Mojo FFI boundary plus the
  backend-specific lowering.
