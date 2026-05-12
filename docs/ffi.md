---
title: FFI design for max backend
date: 2026/05/11
---

P5 now has two layers. The Python MAX Graph backend makes
`backend="max[:cpu|gpu]"` executable for the shipped equation grammar. The Mojo
MAX backend seam in `src/einsum/backends/max.mojo` also consumes
`ContractionPlan`, packs runtime-strided operands into BMM-shaped `TileTensor`
buffers, and calls `linalg.bmm.batched_matmul` for pairwise steps.

The remaining FFI rewire matters because P11 (native CPU GETT), P12 (native
GPU SM90), deterministic low-precision controls, and zero-copy device ownership
all want `TileTensor` over an unowned pointer instead of Python-side NumPy
buffers.

## Today's FFI surface

`src/lib.mojo` exposes six `PythonModuleBuilder` entries:

| Function                       | Operand contract                | Return                       |
| ------------------------------ | ------------------------------- | ---------------------------- |
| `parse_equation(eq)`           | pure equation parse             | `dict[str, object]`          |
| `einsum_reference(eq, fl, sh)` | flat fp64 buffers + shape lists | `(flat_out_fp64, out_shape)` |
| `einsum_native(eq, fl, sh, p)` | flat fp64 buffers + shape lists | `(flat_out_fp64, out_shape)` |
| `einsum_path(eq, sh)`          | shape lists                     | `list[tuple[int, ...]]`      |
| `einsum_compute_path(eq, ...)` | shape lists + algorithm name    | `list[tuple[int, ...]]`      |
| `path_cost(eq, sh, p)`         | shape lists + working-set path  | `dict[str, object]`          |

`einsum_reference` and `einsum_native` allocate an `UnsafePointer[Float64]` per
operand and memcpy the Python list in. End-to-end ownership makes that fine for
the reference backend. The current executable MAX Graph path bypasses this Mojo
FFI and builds a `max.graph.Graph` in Python, then feeds MAX `Buffer` objects.

`src/einsum/backends/max.mojo` now proves the Mojo-side lowering shape, but it
still receives flat `UnsafePointer[Float64]` buffers. That copy becomes a
per-call tax on buffers the caller already owns through DLPack. The next FFI
boundary should let the caller's pointer land inside a `TileTensor` directly.

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

## Build-config note

`max==26.2.0` ships `layout.mojopkg` and `linalg.mojopkg` in
`modular/lib/mojo`. Keep the Mojo compiler and MAX package versions aligned:
`max==26.2.0` expects `mojo==0.26.2.0`. A 1.0 beta compiler can import the stdlib
but rejects the 26.2 MAX Mojo packages before source checking.

Local smoke command used for the Mojo MAX seam:

```bash
env MODULAR_PATH= MODULAR_DERIVED_PATH= \
  MODULAR_MOJO_MAX_IMPORT_PATH="$MOJO_026_SITE_PACKAGES/modular/lib/mojo" \
  "$MOJO_026_BIN/mojo" run \
  -I "$MAX_262_SITE_PACKAGES/modular/lib/mojo" \
  -I src \
  tests/mojo/smoke_parse.mojo
```

`mohaus develop` should expose the same matched package set after `uv sync`.

## Python-side wiring

`python/moeinsum/__init__.py` already takes a `backend` parameter. Today,
`backend.startswith("max")` dispatches to `_interop_max.execute_max`, the
Python MAX Graph bridge. The Mojo cutover replaces that call with the FFI
entry above:

```python
_BACKENDS = ("reference", "native", "max", "max:cpu", "max:gpu")

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
- `tests/python/test_backend_stubs.py` keeps `lowering_spec`
  independent of `max.graph` and pins the same classifier consumed by
  the executable MAX bridge.
- `tests/python/test_property.py` remains the broad parser / planner /
  reference invariant suite. When the Mojo backend covers the full
  grammar, add it to the same parity matrix instead of making a weaker
  side harness.

## Phase ladder

**Mojo TileTensor MAX cutover:**

1. Packed pairwise lowering in `src/einsum/backends/max.mojo`: done.
2. Implement `einsum_max_py` in `src/lib.mojo` with the rank
   monomorphization block.
3. Replace packed TTGT staging with zero-copy `RuntimeLayout` / `TileTensor`
   views when operand strides can express the B/M/K and B/K/N layouts.
4. Add dtype-specialized entry points instead of the current fp64-only flat ABI.
5. Thread the existing Mojo `target` parameter through `einsum_max_py` so
   `backend="max:gpu"` can instantiate `execute_max[target="gpu"]`.
6. Redirect `einsum(backend="max", ...)` from the Python MAX Graph bridge
   only if the Mojo FFI equals the current `_interop_max.py` feature
   surface and wins enough perf to justify owning it.
7. Add `_interop.to_dlpack`.
8. Run `tests/python/test_numpy_parity.py` with `backend="max"`.

**P10 - GPU dispatch validation:**

1. Python MAX Graph path: done for `backend="max:gpu"` on the B200 host.
2. Future Mojo `target="gpu"`: repeat CPU/GPU equivalence at
   atol=1e-5 (fp32) / 1e-2 (bf16).
3. `python -m moeinsum._cli.bench` happy-path on a GPU host remains the smoke
   test for both paths.

**P11 - Native CPU GETT (3 days):**

1. `src/einsum/backends/native.mojo` already exposes the same flat-buffer
   executor seam as `MaxBackend`; replace the inner pairwise loop with GETT.
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

- `src/einsum/backends/max.mojo` is no longer a facade. It owns a packed
  TileTensor path for pairwise steps: classify B/M/K/N from `PairwiseStep`,
  materialize lhs as `[batch, m, k]`, materialize rhs as `[batch, k, n]`, call
  `linalg.bmm.batched_matmul`, then expose the step result in `out_labels`
  order. Unary steps still use Mojo view/reduce helpers.
- `python/moeinsum/_interop_max.py` owns the Python-side lowering spec and
  executable MAX Graph bridge. `classify_pair(...)`, `lowering_spec(...)`,
  and `MaxGraphBackend.execute(...)` share that file; `_max_graph.py` was
  deleted once MAX became a default dependency.
- Dim classification, path optimization, plan IR, and the test harness
  have coverage. The remaining work here is FFI/device/dtype polish rather than
  a missing Mojo MAX implementation.
