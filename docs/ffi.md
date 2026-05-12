---
title: FFI design for max backend
date: 2026/05/11
---

P5 now has three executable MAX layers. Public `backend="max:cpu"` enters the
Mojo MAX backend through `_native.einsum_max_f32_cpu_ptrs` or
`_native.einsum_max_f64_cpu_ptrs`. Public `backend="max:gpu"` lazily imports
`_native_gpu` and uses the same pointer payload for native fp32/fp64 accelerator
execution when that module is built. The Python MAX Graph backend remains the
graph-introspection path, the accumulator-dtype path, and the fallback for
unsupported native dtypes. The Mojo MAX backend in
`src/einsum/backends/max.mojo` consumes `ContractionPlan`, packs
runtime-strided operands into BMM-shaped `TileTensor` buffers, and calls
`linalg.bmm.batched_matmul` for pairwise steps.

The current FFI boundary borrows MAX `Buffer` storage created from DLPack
producers. Python owns lifetime and device placement; Mojo receives typed data
pointers, shape lists, stride lists, output pointer, and a working-set path.

## Today's FFI surface

`src/lib.mojo` exposes nine `PythonModuleBuilder` entries:

| Function                                  | Operand contract                      | Return                       |
| ----------------------------------------- | ------------------------------------- | ---------------------------- |
| `parse_equation(eq)`                      | pure equation parse                   | `dict[str, object]`          |
| `parse_equation_expanded(eq)`             | equation + operand shape ranks        | `dict[str, object]`          |
| `einsum_reference(eq, fl, sh)`            | flat fp64 buffers + shape lists       | `(flat_out_fp64, out_shape)` |
| `einsum_native(eq, fl, sh, p)`            | flat fp64 buffers + shape lists       | `(flat_out_fp64, out_shape)` |
| `einsum_max_f32_cpu_ptrs(eq, payload, p)` | borrowed fp32 pointers + shape/stride | `None`                       |
| `einsum_max_f64_cpu_ptrs(eq, payload, p)` | borrowed fp64 pointers + shape/stride | `None`                       |
| `einsum_path(eq, sh)`                     | shape lists                           | `list[tuple[int, ...]]`      |
| `einsum_compute_path(eq, ...)`            | shape lists + algorithm name          | `list[tuple[int, ...]]`      |
| `path_cost(eq, sh, p)`                    | shape lists + working-set path        | `dict[str, object]`          |

`src/native_gpu.mojo` exposes `_native_gpu.einsum_max_f32_gpu_ptrs` and
`_native_gpu.einsum_max_f64_gpu_ptrs`. It is an optional extension source, not a
default `mohaus` module, because macOS CPU wheel builds otherwise instantiate
the GPU specialization and hit the local Metal compiler. The Python module is
imported only when native accelerator execution is requested, which keeps the
CPU extension from instantiating GPU specializations.

Optional accelerator-host build shape:

```bash
target="${MOEINSUM_NATIVE_GPU_TARGET:-sm_90}"
suffix="$("$VENV/bin/python" - <<'PY'
import sysconfig
print(sysconfig.get_config_var("EXT_SUFFIX"))
PY
)"
"$VENV/bin/mojo" build --emit shared-lib \
  --target-accelerator "$target" \
  -o "python/moeinsum/_native_gpu${suffix}" \
  -I src \
  src/native_gpu.mojo
```

`einsum_reference` and `einsum_native` still allocate `UnsafePointer[Float64]`
buffers from Python lists. The MAX path no longer does. `_interop_max.py` creates
MAX `Buffer` objects through `Buffer.from_dlpack(...)`, allocates an output
`Buffer`, then passes `_data_ptr()` values through the typed native entrypoint.

## Native MAX pointer payload

Python sends a single payload dictionary:

```python
payload = {
    "operand_ptrs": [buffer._data_ptr(), ...],
    "operand_shapes": [[...], ...],
    "operand_strides": [[...], ...],
    "operand_numels": [buffer.num_elements, ...],
    "out_ptr": output._data_ptr(),
    "out_shape": [...],
    "out_strides": [...],
}
```

The Mojo helper `src/einsum/max_ffi.mojo::execute_max_ptr_payload` reuses the
native parser and planner, expands ellipsis from operand ranks, validates pointer
payload sizes, builds a `ContractionPlan`, and calls
`src/einsum/backends/max.mojo::execute_max[dtype=..., target=...]`.

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

`src/einsum/unary.mojo` already implements the kernels. The current MAX native
path accepts typed borrowed pointers plus runtime shape/stride lists:

- `transpose_view` - new `Layout`, no kernel
- `diagonal_view` - new `Layout` with stride summation, no kernel
- `reduce_sum_axes[dtype]` - read from typed pointer + runtime strides
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
`backend.startswith("max")` dispatches to `_interop_max.execute_max`.
That function routes native fp32/fp64 calls with `accum_dtype is None` through
the pointer payload ABI. `max:cpu` uses `_native`; `max:gpu` and accelerator-backed
`max` try `_native_gpu`; unsupported native dtypes and explicit accumulator
requests go through MAX Graph.

Behavioral contract: unsupported MAX cases raise `NotImplementedError`;
they do not silently fall back to reference.

## Test plan

The harness now separates the public native MAX CPU route from the Python MAX
Graph interop route:

- `tests/python/test_max_backend.py` pins matmul, BMM, 3-operand chains,
  outer product, unary transpose, reduce-sum, size-1 broadcast, and
  ellipsis on public `backend="max:cpu"`, plus a graph-cache bypass assertion.
- `tests/python/test_max_backend_bf16.py` pins bf16 drift growth and
  dtype round-trip through MAX.
- `tests/python/test_backend_stubs.py` keeps `lowering_spec`
  independent of `max.graph` and pins the same classifier consumed by
  the executable graph interop bridge.
- `tests/python/test_max_backend_perf.py` keeps the raw MAX Graph comparison
  on the explicit graph interop executor, since public `max:cpu` now enters
  the Mojo TileTensor backend.
- `tests/python/test_property.py` remains the broad parser / planner /
  reference invariant suite. Native MAX parity should join the broad property
  matrix once the pointer ABI has GPU coverage.

## Phase ladder

**Mojo TileTensor MAX cutover:**

1. Packed pairwise lowering in `src/einsum/backends/max.mojo`: done.
2. Flat `einsum_max_cpu_py` removed; fp32/fp64 pointer payload entries in
   `_native` and `_native_gpu`: done.
3. Replace packed TTGT staging with zero-copy `RuntimeLayout` / `TileTensor`
   views when operand strides can express the B/M/K and B/K/N layouts.
4. Add bf16/fp16 dtype-specialized entries once the native kernels can honor
   low-precision accumulator policy.
5. Thread the existing Mojo `target` parameter through the pointer payload ABI:
   done for CPU and lazy GPU module exports.
6. Redirected `einsum(backend="max:cpu", ...)` from the Python MAX Graph
   bridge to the Mojo MAX backend. Default `backend="max"` now tries native
   GPU on accelerator hosts when `_native_gpu` is built, then falls back to
   MAX Graph.
7. Add a direct `_interop.to_dlpack` path for public GPU framework operands.
8. Run `tests/python/test_numpy_parity.py` with `backend="max"`.

**P10 - GPU dispatch validation:**

1. Python MAX Graph path: done for `backend="max:gpu"` on the B200 host.
2. Mojo `target="gpu"` export: added as `_native_gpu`; repeat CPU/GPU equivalence at
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
   (`to_numpy`) is dtype-preserving and ships; MAX native send-side now uses
   `Buffer.from_dlpack` after the public API has converted operands to numpy.
2. `einsum(...)` already round-trips via `_from_numpy(kind)`, so the
   caller's framework / dtype survive the call. The missing piece is the
   direct framework-to-MAX edge for GPU tensors, bypassing the numpy
   intermediate in `__init__.py`.
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
- `python/moeinsum/_interop_max.py` owns Python-side MAX routing, lowering
  spec, and the executable MAX Graph bridge. Public `max:cpu` calls
  `_native.einsum_max_f32_cpu_ptrs` / `_native.einsum_max_f64_cpu_ptrs`;
  `max:gpu` lazily imports `_native_gpu` for the matching GPU exports.
  `classify_pair(...)`, `lowering_spec(...)`, and `MaxGraphBackend.execute(...)`
  still share the graph interop machinery.
  `_max_graph.py` was deleted once MAX became a default dependency.
- Dim classification, path optimization, plan IR, and the test harness
  have coverage. The remaining work here is FFI/device/dtype polish rather than
  a missing Mojo MAX implementation.
