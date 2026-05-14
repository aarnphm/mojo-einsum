"""MAX backend interop for public Python dispatch.

`max:cpu` routes through the native Mojo MAX TileTensor backend exposed by
`_native`. `max:gpu` lazily imports `_native_gpu` for typed pointer payloads
when that extension is built. Explicit graph introspection and graph cache
tests still use the MAX Graph lowering in this module.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import import_module
from math import prod
from threading import RLock
from typing import TYPE_CHECKING, Protocol, cast

import ml_dtypes
import numpy as np

from ._interop import dtype_of as _dtype_of
from ._interop import from_numpy as _from_numpy
from ._interop import is_dlpack_array as _is_dlpack_array
from ._interop import shape_of as _shape_of
from ._native import einsum_max_f32_cpu_ptrs as _einsum_max_f32_cpu_ptrs_native
from ._native import einsum_max_f64_cpu_ptrs as _einsum_max_f64_cpu_ptrs_native
from ._native import parse_equation_expanded as _parse_equation_expanded_native

if TYPE_CHECKING:
  from max.engine import InferenceSession, Model
  from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops


def _prefer_packaged_modular_runtime() -> None:
  """Keep PyPI MAX from loading a monorepo KGEN runtime through ambient env."""
  for key in ("MODULAR_PATH", "MODULAR_DERIVED_PATH"):
    if os.environ.get(key):
      os.environ[key] = ""


_prefer_packaged_modular_runtime()

from max.driver import CPU, Accelerator, Buffer, Device, accelerator_count  # noqa: E402
from max.dtype import DType  # noqa: E402

_MAX_GRAPH_LOADED = False


def _ensure_max_graph_loaded() -> None:
  global DeviceRef, Graph, InferenceSession, TensorType, ops
  global _MAX_GRAPH_LOADED
  if _MAX_GRAPH_LOADED:
    return
  from max.engine import InferenceSession as _InferenceSession  # noqa: PLC0415
  from max.graph import DeviceRef as _DeviceRef  # noqa: PLC0415
  from max.graph import Graph as _Graph  # noqa: PLC0415
  from max.graph import TensorType as _TensorType  # noqa: PLC0415
  from max.graph import ops as _ops  # noqa: PLC0415

  DeviceRef = _DeviceRef
  Graph = _Graph
  InferenceSession = _InferenceSession
  TensorType = _TensorType
  ops = _ops
  _MAX_GRAPH_LOADED = True


_NativeMaxEntrypoint = Callable[[str, dict[str, object], list[tuple[int, ...]]], None]
_NATIVE_MAX_CPU: dict[str, _NativeMaxEntrypoint] = {
  "float32": _einsum_max_f32_cpu_ptrs_native,
  "float64": _einsum_max_f64_cpu_ptrs_native,
}
_NATIVE_MAX_GPU_NAMES = {
  "float32": "einsum_max_f32_gpu_ptrs",
  "float64": "einsum_max_f64_gpu_ptrs",
}
_TORCH_DTYPE_NAMES = {
  "float16": "float16",
  "float32": "float32",
  "float64": "float64",
  "bfloat16": "bfloat16",
}


@dataclass(frozen=True)
class _Node:
  value: TensorValue
  labels: str
  shape: tuple[int, ...]


class _LabeledShape(Protocol):
  labels: str
  shape: tuple[int, ...]


@dataclass(frozen=True)
class _LoweringNode:
  labels: str
  shape: tuple[int, ...]


@dataclass(frozen=True)
class _PairLayout:
  lhs: _LabeledShape
  rhs: _LabeledShape
  lhs_sizes: dict[str, int]
  rhs_sizes: dict[str, int]
  sizes: dict[str, int]
  batch: str
  contract: str
  free_lhs: str
  free_rhs: str
  lhs_order: str
  rhs_order: str
  natural_labels: str
  swapped: bool


@dataclass
class _Executable:
  model: Model
  device: Device
  input_dtype: np.dtype


_MODEL_CACHE_MAX = 512
_MODEL_CACHE: OrderedDict[tuple[object, ...], _Executable] = OrderedDict()
_MODEL_CACHE_LOCK = RLock()


@dataclass(frozen=True)
class DimClassification:
  lhs_labels: str
  rhs_labels: str
  out_labels: str
  batch: tuple[str, ...]
  contract: tuple[str, ...]
  free_lhs: tuple[str, ...]
  free_rhs: tuple[str, ...]


def classify_pair(lhs_labels: str, rhs_labels: str, out_labels: str) -> DimClassification:
  lhs_set = set(lhs_labels)
  rhs_set = set(rhs_labels)
  out_set = set(out_labels)

  batch = tuple(dict.fromkeys(c for c in lhs_labels if c in rhs_set and c in out_set))
  contract = tuple(dict.fromkeys(c for c in lhs_labels if c in rhs_set and c not in out_set))
  free_lhs = tuple(dict.fromkeys(c for c in lhs_labels if c not in rhs_set and c in out_set))
  free_rhs = tuple(dict.fromkeys(c for c in rhs_labels if c not in lhs_set and c in out_set))

  return DimClassification(
    lhs_labels=lhs_labels,
    rhs_labels=rhs_labels,
    out_labels=out_labels,
    batch=batch,
    contract=contract,
    free_lhs=free_lhs,
    free_rhs=free_rhs,
  )


def _native_label_char(label: int, label_chars: list[str]) -> str:
  char = label_chars[label]
  if len(char) == 1:
    return char
  return chr(0xE000 + label)


def _parse_equation(eq: str, shapes: list[tuple[int, ...]]) -> tuple[list[str], str]:
  parsed = _parse_equation_expanded_native(eq, [list(shape) for shape in shapes])
  label_chars = cast("list[str]", parsed["label_chars"])
  inputs = cast("list[list[int]]", parsed["inputs"])
  output = cast("list[int]", parsed["output"])
  return (
    ["".join(_native_label_char(label, label_chars) for label in labels) for labels in inputs],
    "".join(_native_label_char(label, label_chars) for label in output),
  )


def _label_sizes(labels: str, shape: tuple[int, ...]) -> dict[str, int]:
  if len(labels) != len(shape):
    raise ValueError(f"operand labels {labels!r} have rank {len(labels)} but shape has rank {len(shape)}")
  out: dict[str, int] = {}
  for label, dim in zip(labels, shape, strict=True):
    previous = out.get(label)
    if previous is not None and previous != dim:
      raise ValueError(f"size conflict on label {label!r}: {previous} vs {dim}")
    out[label] = dim
  return out


def _unique_labels(labels: str) -> str:
  return "".join(dict.fromkeys(labels))


def _product(values: list[int]) -> int:
  return prod(values)


def _sizes_for(labels: str, sizes: dict[str, int]) -> list[int]:
  return [sizes[label] for label in labels]


def _merge_label_sizes(lhs_sizes: dict[str, int], rhs_sizes: dict[str, int]) -> dict[str, int]:
  sizes = dict(lhs_sizes)
  for label, dim in rhs_sizes.items():
    previous = sizes.get(label)
    if previous is None or previous == dim:
      sizes[label] = dim
    elif previous == 1:
      sizes[label] = dim
    elif dim == 1:
      continue
    else:
      raise ValueError(f"size conflict on label {label!r}: {previous} vs {dim}")
  return sizes


def _permute_if_needed(value: TensorValue, current: str, desired: str) -> TensorValue:
  if current == desired:
    return value

  dims = [current.index(label) for label in desired]
  return ops.permute(value, dims)


def _reshape(value: TensorValue, shape: list[int]) -> TensorValue:
  return ops.reshape(value, shape)


def _cast(value: TensorValue, dtype: DType) -> TensorValue:
  return ops.cast(value, dtype)


def _reduce_out_labels(node: _Node, keep: set[str], accum_dtype: DType) -> _Node:
  value = _cast(node.value, accum_dtype)
  labels = list(node.labels)
  shape = list(node.shape)
  for axis in reversed(range(len(labels))):
    if labels[axis] in keep:
      continue
    value = ops.sum(value, axis=axis)
    value = ops.squeeze(value, axis=axis)
    del labels[axis]
    del shape[axis]
  return _Node(value=value, labels="".join(labels), shape=tuple(shape))


def _output_labels_for_pair(lhs: str, rhs: str, final_output: str, others: list[str]) -> str:
  future = set(final_output)
  for labels in others:
    future.update(labels)
  labels = "".join(dict.fromkeys(label for label in lhs + rhs if label in future))
  if not others and all(label in final_output for label in labels):
    return "".join(label for label in final_output if label in labels)
  return labels


def _broadcast_if_needed(
  value: TensorValue,
  ordered_labels: str,
  source_axis_sizes: dict[str, int],
  resolved_sizes: dict[str, int],
) -> TensorValue:
  """Broadcast size-1 axes to resolved per-label sizes."""
  current_shape = [source_axis_sizes[label] for label in ordered_labels]
  target_shape = [resolved_sizes[label] for label in ordered_labels]
  if current_shape == target_shape:
    return value

  return ops.broadcast_to(value, target_shape)


def _pair_layout(lhs: _LabeledShape, rhs: _LabeledShape, out_labels: str, *, swapped: bool) -> _PairLayout:
  lhs_sizes = _label_sizes(lhs.labels, lhs.shape)
  rhs_sizes = _label_sizes(rhs.labels, rhs.shape)
  sizes = _merge_label_sizes(lhs_sizes, rhs_sizes)

  cls = classify_pair(lhs.labels, rhs.labels, out_labels)
  batch = "".join(cls.batch)
  contract = "".join(cls.contract)
  free_lhs = "".join(cls.free_lhs)
  free_rhs = "".join(cls.free_rhs)

  return _PairLayout(
    lhs=lhs,
    rhs=rhs,
    lhs_sizes=lhs_sizes,
    rhs_sizes=rhs_sizes,
    sizes=sizes,
    batch=batch,
    contract=contract,
    free_lhs=free_lhs,
    free_rhs=free_rhs,
    lhs_order=batch + free_lhs + contract,
    rhs_order=batch + contract + free_rhs,
    natural_labels=batch + free_lhs + free_rhs,
    swapped=swapped,
  )


def _diagonal_indices(labels: str, shape: tuple[int, ...]) -> tuple[np.ndarray, str, tuple[int, ...]]:
  """Return gather_nd indices that collapse repeated labels to one axis."""
  sizes = _label_sizes(labels, shape)
  unique = _unique_labels(labels)
  if unique == labels:
    return np.empty((), dtype=np.int32), labels, shape

  unique_shape = tuple(sizes[label] for label in unique)
  grids = np.indices(unique_shape, dtype=np.int32)
  label_to_grid = {label: grids[idx] for idx, label in enumerate(unique)}
  indices = np.stack([label_to_grid[label] for label in labels], axis=-1)
  return np.ascontiguousarray(indices, dtype=np.int32), unique, unique_shape


def _diagonalize_node(node: _Node, device: Device) -> _Node:
  indices, labels, shape = _diagonal_indices(node.labels, node.shape)
  if labels == node.labels:
    return node

  index_value = ops.constant(indices, dtype=DType.int32, device=DeviceRef.from_device(device))
  return _Node(
    value=ops.gather_nd(node.value, index_value, batch_dims=0),
    labels=labels,
    shape=shape,
  )


def _best_pair_layout(lhs: _LabeledShape, rhs: _LabeledShape, out_labels: str) -> _PairLayout:
  normal = _pair_layout(lhs, rhs, out_labels, swapped=False)
  swapped = _pair_layout(rhs, lhs, out_labels, swapped=True)
  if normal.natural_labels != out_labels and swapped.natural_labels == out_labels:
    return swapped
  return normal


def _lower_pair(lhs: _Node, rhs: _Node, out_labels: str, accum_dtype: DType) -> _Node:
  layout = _best_pair_layout(lhs, rhs, out_labels)
  layout_lhs = layout.lhs
  layout_rhs = layout.rhs
  if not isinstance(layout_lhs, _Node) or not isinstance(layout_rhs, _Node):
    raise TypeError("MAX lowering expected graph nodes")

  lhs_value = _permute_if_needed(layout_lhs.value, layout_lhs.labels, layout.lhs_order)
  rhs_value = _permute_if_needed(layout_rhs.value, layout_rhs.labels, layout.rhs_order)
  lhs_value = _broadcast_if_needed(lhs_value, layout.lhs_order, layout.lhs_sizes, layout.sizes)
  rhs_value = _broadcast_if_needed(rhs_value, layout.rhs_order, layout.rhs_sizes, layout.sizes)
  lhs_value = _cast(lhs_value, accum_dtype)
  rhs_value = _cast(rhs_value, accum_dtype)

  batch_shape = _sizes_for(layout.batch, layout.sizes)
  m_shape = _sizes_for(layout.free_lhs, layout.sizes)
  k_shape = _sizes_for(layout.contract, layout.sizes)
  n_shape = _sizes_for(layout.free_rhs, layout.sizes)
  m = _product(m_shape)
  k = _product(k_shape)
  n = _product(n_shape)

  lhs_value = _reshape(lhs_value, [*batch_shape, m, k])
  rhs_value = _reshape(rhs_value, [*batch_shape, k, n])
  out = ops.matmul(lhs_value, rhs_value)

  natural_shape = [*batch_shape, *m_shape, *n_shape]
  out = _reshape(out, natural_shape)
  out = _permute_if_needed(out, layout.natural_labels, out_labels)
  return _Node(
    value=out,
    labels=out_labels,
    shape=tuple(_sizes_for(out_labels, layout.sizes)),
  )


def _broadcast_payload(
  ordered_labels: str,
  source_axis_sizes: dict[str, int],
  resolved_sizes: dict[str, int],
) -> dict[str, object] | None:
  current_shape = [source_axis_sizes[label] for label in ordered_labels]
  target_shape = [resolved_sizes[label] for label in ordered_labels]
  if current_shape == target_shape:
    return None
  return {
    "labels": ordered_labels,
    "from_shape": current_shape,
    "to_shape": target_shape,
  }


def _reduce_node_for_spec(node: _LoweringNode, keep: set[str]) -> tuple[_LoweringNode, dict[str, object] | None]:
  sizes = _label_sizes(node.labels, node.shape)
  labels = "".join(label for label in node.labels if label in keep)
  if labels == node.labels:
    return node, None
  return (
    _LoweringNode(labels=labels, shape=tuple(_sizes_for(labels, sizes))),
    {
      "kind": "reduce_sum",
      "src_labels": node.labels,
      "dst_labels": labels,
      "reduced_labels": [label for label in node.labels if label not in keep],
    },
  )


def _diagonalize_node_for_spec(node: _LoweringNode, operand: int) -> tuple[_LoweringNode, dict[str, object] | None]:
  indices, labels, shape = _diagonal_indices(node.labels, node.shape)
  if labels == node.labels:
    return node, None
  return (
    _LoweringNode(labels=labels, shape=shape),
    {
      "kind": "diagonal",
      "target_op": "max.graph.ops.gather_nd",
      "operand": operand,
      "src_labels": node.labels,
      "dst_labels": labels,
      "src_shape": list(node.shape),
      "dst_shape": list(shape),
      "index_shape": list(indices.shape),
    },
  )


def lowering_spec(
  eq: str,
  shapes: list[tuple[int, ...]],
  path: list[tuple[int, ...]],
) -> dict[str, object]:
  """Return the executable MAX lowering without compiling a graph."""
  inputs, final_output = _parse_equation(eq, shapes)

  working = [_LoweringNode(labels=labels, shape=shape) for labels, shape in zip(inputs, shapes, strict=True)]
  spec_ops: list[dict[str, object]] = []

  for operand, node in enumerate(working):
    lowered, payload = _diagonalize_node_for_spec(node, operand)
    if payload is not None:
      spec_ops.append(payload)
    working[operand] = lowered

  for step_idx, step in enumerate(path):
    if len(step) == 1:
      (idx,) = step
      future = set(final_output)
      for j, node in enumerate(working):
        if j != idx:
          future.update(node.labels)
      reduced, payload = _reduce_node_for_spec(working[idx], future)
      if payload is not None:
        payload["step"] = step_idx
        payload["operand"] = idx
        payload["target_op"] = "max.graph.ops.sum + max.graph.ops.squeeze"
        spec_ops.append(payload)
      working[idx] = reduced
      continue

    li, ri = step
    lhs = working[li]
    rhs = working[ri]
    others = [node.labels for j, node in enumerate(working) if j not in (li, ri)]
    out_labels = _output_labels_for_pair(lhs.labels, rhs.labels, final_output, others)
    layout = _best_pair_layout(lhs, rhs, out_labels)

    batch_shape = _sizes_for(layout.batch, layout.sizes)
    m_shape = _sizes_for(layout.free_lhs, layout.sizes)
    k_shape = _sizes_for(layout.contract, layout.sizes)
    n_shape = _sizes_for(layout.free_rhs, layout.sizes)
    m = _product(m_shape)
    k = _product(k_shape)
    n = _product(n_shape)
    matmul_payload: dict[str, object] = {
      "kind": "matmul",
      "target_op": "max.graph.ops.matmul",
      "step": step_idx,
      "lhs": li,
      "rhs": ri,
      "swapped_operands": layout.swapped,
      "lhs_labels": lhs.labels,
      "rhs_labels": rhs.labels,
      "out_labels": out_labels,
      "matmul_lhs_labels": layout.lhs.labels,
      "matmul_rhs_labels": layout.rhs.labels,
      "batch": list(layout.batch),
      "contract": list(layout.contract),
      "free_lhs": list(layout.free_lhs),
      "free_rhs": list(layout.free_rhs),
      "lhs_order": layout.lhs_order,
      "rhs_order": layout.rhs_order,
      "natural_labels": layout.natural_labels,
      "needs_output_transpose": layout.natural_labels != out_labels,
      "bmm_shape": {
        "batch": batch_shape,
        "m": m,
        "k": k,
        "n": n,
        "lhs": [*batch_shape, m, k],
        "rhs": [*batch_shape, k, n],
        "out": [*batch_shape, m, n],
      },
      "output_shape": _sizes_for(out_labels, layout.sizes),
      "broadcasts": [
        item
        for item in (
          _broadcast_payload(layout.lhs_order, layout.lhs_sizes, layout.sizes),
          _broadcast_payload(layout.rhs_order, layout.rhs_sizes, layout.sizes),
        )
        if item is not None
      ],
    }
    spec_ops.append(matmul_payload)

    out = _LoweringNode(labels=out_labels, shape=tuple(_sizes_for(out_labels, layout.sizes)))
    for idx in sorted((li, ri), reverse=True):
      del working[idx]
    working.append(out)

  if len(working) != 1:
    raise ValueError(f"contraction path leaves {len(working)} tensors; expected 1")

  result, payload = _reduce_node_for_spec(working[0], set(final_output))
  if payload is not None:
    payload["step"] = len(spec_ops)
    payload["operand"] = 0
    payload["target_op"] = "max.graph.ops.sum + max.graph.ops.squeeze"
    spec_ops.append(payload)
  if result.labels != final_output:
    spec_ops.append({
      "kind": "transpose",
      "target_op": "max.graph.ops.permute",
      "step": len(spec_ops),
      "src_labels": result.labels,
      "dst_labels": final_output,
    })

  return {
    "inputs": list(inputs),
    "output": final_output,
    "ops": spec_ops,
    "result_shape": _sizes_for(final_output, _label_sizes(result.labels, result.shape)) if final_output else [],
  }


def _lower_graph(
  eq: str,
  shapes: list[tuple[int, ...]],
  path: list[tuple[int, ...]],
  dtype: DType,
  accum_dtype: DType,
  device: Device,
) -> Graph:
  _ensure_max_graph_loaded()
  inputs, final_output = _parse_equation(eq, shapes)

  input_types = [TensorType(dtype, shape=shape, device=DeviceRef.from_device(device)) for shape in shapes]

  with Graph("moeinsum_max", input_types=input_types) as graph:
    working = [
      _diagonalize_node(_Node(value=graph_input.tensor, labels=labels, shape=shape), device)
      for graph_input, labels, shape in zip(graph.inputs, inputs, shapes, strict=True)
    ]

    for step in path:
      if len(step) == 1:
        (idx,) = step
        future = set(final_output)
        for j, node in enumerate(working):
          if j != idx:
            future.update(node.labels)
        working[idx] = _reduce_out_labels(working[idx], future, accum_dtype)
        continue

      li, ri = step
      lhs = working[li]
      rhs = working[ri]
      others = [node.labels for j, node in enumerate(working) if j not in (li, ri)]
      out_labels = _output_labels_for_pair(lhs.labels, rhs.labels, final_output, others)
      out = _lower_pair(lhs, rhs, out_labels, accum_dtype)

      for idx in sorted((li, ri), reverse=True):
        del working[idx]
      working.append(out)

    if len(working) != 1:
      raise ValueError(f"contraction path leaves {len(working)} tensors; expected 1")

    result = _reduce_out_labels(working[0], set(final_output), accum_dtype)
    result_value = _permute_if_needed(result.value, result.labels, final_output)
    result_value = _cast(result_value, dtype)
    graph.output(result_value)
    return graph


def _select_device(backend: str) -> Device:
  if backend == "max:cpu":
    return CPU()
  if backend == "max:gpu":
    if accelerator_count() == 0:
      raise RuntimeError("backend='max:gpu' requested but MAX reports no accelerator")
    return Accelerator()
  if accelerator_count() > 0:
    return Accelerator()
  return CPU()


def _graph_input_dtype(public_dtype: np.dtype, device: Device) -> np.dtype:
  if public_dtype.name == "bfloat16" and isinstance(device, CPU):
    return np.dtype("float32")
  return public_dtype


def _compile(
  eq: str,
  shapes: list[tuple[int, ...]],
  path: list[tuple[int, ...]],
  public_dtype: np.dtype,
  accum_dtype: DType,
  backend: str,
) -> _Executable:
  _prefer_packaged_modular_runtime()

  device = _select_device(backend)
  input_dtype = _graph_input_dtype(public_dtype, device)
  graph = _lower_graph(eq, shapes, path, _max_dtype_for(input_dtype), accum_dtype, device)
  session = InferenceSession(devices=[device])
  try:
    return _Executable(model=session.load(graph), device=device, input_dtype=input_dtype)
  except RuntimeError:
    if backend != "max":
      raise

    if accelerator_count() == 0:
      raise
    cpu = CPU()
    input_dtype = _graph_input_dtype(public_dtype, cpu)
    graph = _lower_graph(eq, shapes, path, _max_dtype_for(input_dtype), accum_dtype, cpu)
    session = InferenceSession(devices=[cpu])
    return _Executable(model=session.load(graph), device=cpu, input_dtype=input_dtype)


def _max_dtype_for(dtype: np.dtype) -> DType:
  if dtype == np.dtype("float32"):
    return DType.float32
  if dtype == np.dtype("float64"):
    return DType.float64
  if dtype.name == "bfloat16":
    return DType.bfloat16
  raise NotImplementedError(
    f"backend='max' supports float32/float64/bfloat16, got {dtype}. "
    f"fp16 is GPU-only in MAX today; route through backend='max:gpu' once "
    f"that path lands."
  )


def _default_accum_dtype(dtype: np.dtype) -> np.dtype:
  if dtype == np.dtype("float16") or dtype.name == "bfloat16":
    return np.dtype("float32")
  return dtype


def _max_accum_dtype_for(dtype: np.dtype, accum_dtype: np.dtype | None) -> DType:
  resolved = _default_accum_dtype(dtype) if accum_dtype is None else accum_dtype
  if resolved == np.dtype("float32") or resolved == np.dtype("float64"):
    return _max_dtype_for(resolved)
  raise NotImplementedError(
    f"backend='max' supports accum_dtype float32/float64, got {resolved}. "
    f"Low-precision accumulation is intentionally rejected because MAX Graph "
    f"does not expose a bf16/fp16 accumulator-control knob here."
  )


def _input_buffer(arr: np.ndarray, device: Device) -> Buffer:
  """NumPy -> MAX Buffer, with a uint16 view for bf16."""
  if arr.dtype.name == "bfloat16":
    u16 = np.ascontiguousarray(arr.view(np.uint16))
    return Buffer.from_numpy(u16).view(DType.bfloat16, arr.shape).to(device)
  return Buffer.from_numpy(arr).to(device)


def _torch_dtype(dtype: np.dtype) -> object:
  import torch  # noqa: PLC0415

  name = dtype.name
  if name == "bfloat16":
    return torch.bfloat16
  torch_name = _TORCH_DTYPE_NAMES.get(name)
  if torch_name is None:
    raise NotImplementedError(f"backend='max' cannot build a torch DLPack buffer for dtype {dtype}")
  return getattr(torch, torch_name)


def _prepare_dlpack_operand(obj: object, dtype: np.dtype, keepalive: list[object]) -> object:
  """Return a contiguous DLPack source, preserving temporaries for Buffer lifetime."""
  module = type(obj).__module__.split(".", 1)[0]
  prepared = obj
  if module == "torch":
    import torch  # noqa: PLC0415

    if not isinstance(prepared, torch.Tensor):
      raise TypeError(f"expected torch.Tensor, got {type(prepared).__name__}")
    target_dtype = _torch_dtype(dtype)
    if prepared.dtype != target_dtype:
      prepared = prepared.to(dtype=target_dtype)
    if not prepared.is_contiguous():
      prepared = prepared.contiguous()
  keepalive.append(prepared)
  return prepared


def _host_numpy_operand(obj: object, dtype: np.dtype) -> np.ndarray:
  module = type(obj).__module__.split(".", 1)[0]
  if module == "torch":
    import torch  # noqa: PLC0415

    if isinstance(obj, torch.Tensor):
      obj = obj.detach().cpu().numpy()
  return np.ascontiguousarray(np.asarray(obj).astype(dtype, copy=False))


def _buffer_on_device(buffer: Buffer, device: Device) -> Buffer:
  if getattr(buffer, "device", None) == device:
    return buffer
  return buffer.to(device)


def _graph_input_buffer(obj: object, dtype: np.dtype, device: Device, keepalive: list[object]) -> Buffer:
  """Array-like -> MAX Buffer without copying CUDA DLPack inputs through NumPy."""
  if isinstance(obj, np.ndarray):
    return _input_buffer(np.ascontiguousarray(obj.astype(dtype, copy=False)), device)

  if _is_dlpack_array(obj):
    prepared = _prepare_dlpack_operand(obj, dtype, keepalive)
    try:
      return _buffer_on_device(Buffer.from_dlpack(prepared), device)
    except (TypeError, RuntimeError, BufferError):
      pass

  arr = _host_numpy_operand(obj, dtype)
  keepalive.append(arr)
  return _input_buffer(arr, device)


def _read_output(buf: Buffer) -> np.ndarray:
  """MAX Buffer -> numpy with a uint16 detour for bf16 (inverse of `_input_buffer`)."""
  host = buf.to(CPU())
  if buf.dtype == DType.bfloat16:
    u16_buf = host.view(DType.uint16, host.shape)
    return u16_buf.to_numpy().view(np.dtype(ml_dtypes.bfloat16))
  return host.to_numpy()


def _buffer_to_return_type(buf: Buffer, dtype: np.dtype, return_type: str) -> object:
  if return_type == "torch":
    import torch  # noqa: PLC0415

    out = torch.from_dlpack(buf)
    target_dtype = _torch_dtype(dtype)
    if out.dtype != target_dtype:
      out = out.to(dtype=target_dtype)
    return out
  out = _read_output(buf)
  if out.dtype != dtype:
    out = out.astype(dtype)
  return _from_numpy(out, return_type)


def _numpy_to_return_type(out: np.ndarray, dtype: np.dtype, return_type: str) -> object:
  if out.dtype != dtype:
    out = out.astype(dtype)
  return _from_numpy(out, return_type)


def _row_major_strides(shape: Sequence[int]) -> list[int]:
  rank = len(shape)
  if rank == 0:
    return []
  strides = [0] * rank
  strides[-1] = 1
  for axis in range(rank - 2, -1, -1):
    strides[axis] = strides[axis + 1] * int(shape[axis + 1])
  return strides


def _native_max_cpu_func(dtype: np.dtype) -> _NativeMaxEntrypoint | None:
  return _NATIVE_MAX_CPU.get(np.dtype(dtype).name)


def _native_max_gpu_func(dtype: np.dtype) -> _NativeMaxEntrypoint | None:
  if os.environ.get("MOEINSUM_ENABLE_NATIVE_GPU") != "1":
    return None
  func_name = _NATIVE_MAX_GPU_NAMES.get(np.dtype(dtype).name)
  if func_name is None:
    return None
  try:
    module = import_module("._native_gpu", __package__)
  except ModuleNotFoundError as exc:
    if exc.name == f"{__package__}._native_gpu":
      return None
    raise
  func = getattr(module, func_name, None)
  if func is None:
    return None
  return cast("_NativeMaxEntrypoint", func)


def _rank_preserving_contiguous(array: np.ndarray) -> np.ndarray:
  if array.shape == ():
    return array
  return np.ascontiguousarray(array)


def _native_input_buffer(
  array: np.ndarray,
  dtype: np.dtype,
  device: Device,
) -> tuple[np.ndarray, Buffer]:
  host_array = _rank_preserving_contiguous(array.astype(dtype, copy=False))
  return host_array, Buffer.from_dlpack(host_array).to(device)


def _native_output_buffer(shape: tuple[int, ...], dtype: DType, device: Device) -> Buffer:
  if shape:
    return Buffer.zeros(shape, dtype, device=device)
  return Buffer.scalar(0, dtype, device=device)


def _result_shape(eq: str, arrays: list[np.ndarray], path: list[tuple[int, ...]]) -> tuple[int, ...]:
  spec = lowering_spec(eq, [tuple(array.shape) for array in arrays], path)
  return tuple(cast("list[int]", spec["result_shape"]))


def _native_payload(inputs: list[Buffer], output: Buffer) -> dict[str, object]:
  input_shapes = [list(buffer.shape) for buffer in inputs]
  return {
    "operand_ptrs": [buffer._data_ptr() for buffer in inputs],
    "operand_shapes": input_shapes,
    "operand_strides": [_row_major_strides(shape) for shape in input_shapes],
    "operand_numels": [buffer.num_elements for buffer in inputs],
    "out_ptr": output._data_ptr(),
    "out_shape": list(output.shape),
    "out_strides": _row_major_strides(output.shape),
  }


def _execute_max_native(
  eq: str,
  arrays: list[np.ndarray],
  path: list[tuple[int, ...]],
  device: Device,
  func: _NativeMaxEntrypoint,
) -> np.ndarray:
  dtype = np.result_type(*arrays)
  max_dtype = _max_dtype_for(dtype)
  output_shape = _result_shape(eq, arrays, path)

  host_arrays: list[np.ndarray] = []
  input_buffers: list[Buffer] = []
  for array in arrays:
    host_array, buffer = _native_input_buffer(array, np.dtype(dtype), device)
    host_arrays.append(host_array)
    input_buffers.append(buffer)

  output = _native_output_buffer(output_shape, max_dtype, device)
  func(eq, _native_payload(input_buffers, output), path)
  device.synchronize()
  out = _read_output(output)
  if out.dtype != dtype:
    out = out.astype(dtype)
  return out


def _execute_max_graph(
  eq: str,
  operands: Sequence[object],
  path: list[tuple[int, ...]],
  backend: str,
  dtype: np.dtype | None = None,
  accum_dtype: np.dtype | None = None,
  return_type: str = "numpy",
) -> object:
  _prefer_packaged_modular_runtime()

  if not operands:
    raise ValueError("einsum requires at least one operand")
  if dtype is None:
    dtype = np.result_type(*[_dtype_of(operand) for operand in operands])
  else:
    dtype = np.dtype(dtype)
  _max_dtype_for(dtype)
  max_accum_dtype = _max_accum_dtype_for(dtype, accum_dtype)
  accum_key = str(_default_accum_dtype(dtype) if accum_dtype is None else accum_dtype)

  shapes = [_shape_of(operand) for operand in operands]
  key = (eq, tuple(shapes), str(dtype), accum_key, tuple(path), backend)
  with _MODEL_CACHE_LOCK:
    executable = _MODEL_CACHE.get(key)
    if executable is None:
      executable = _compile(eq, shapes, path, dtype, max_accum_dtype, backend)
      _MODEL_CACHE[key] = executable
      while len(_MODEL_CACHE) > _MODEL_CACHE_MAX:
        _MODEL_CACHE.popitem(last=False)
    else:
      _MODEL_CACHE.move_to_end(key)

  keepalive: list[object] = []
  inputs = [
    _graph_input_buffer(operand, executable.input_dtype, executable.device, keepalive)
    for operand in operands
  ]
  result = executable.model.execute(*inputs)[0]
  if not isinstance(result, Buffer):
    raise TypeError(f"MAX model returned {type(result).__name__}, expected Buffer")
  return _buffer_to_return_type(result, dtype, return_type)


def execute_max(
  eq: str,
  operands: Sequence[object],
  path: list[tuple[int, ...]],
  backend: str,
  dtype: np.dtype | None = None,
  accum_dtype: np.dtype | None = None,
  return_type: str = "numpy",
) -> object:
  if dtype is None:
    dtype = np.result_type(*[_dtype_of(operand) for operand in operands])
  else:
    dtype = np.dtype(dtype)
  if accum_dtype is None:
    if backend == "max:cpu":
      native_cpu = _native_max_cpu_func(dtype)
      if native_cpu is not None:
        arrays = [_host_numpy_operand(operand, dtype) for operand in operands]
        out = _execute_max_native(eq, arrays, path, CPU(), native_cpu)
        return _numpy_to_return_type(out, dtype, return_type)
    if backend == "max:gpu":
      native_gpu = _native_max_gpu_func(dtype)
      if native_gpu is not None:
        if accelerator_count() == 0:
          raise RuntimeError("backend='max:gpu' requested but MAX reports no accelerator")
        arrays = [_host_numpy_operand(operand, dtype) for operand in operands]
        out = _execute_max_native(eq, arrays, path, Accelerator(), native_gpu)
        return _numpy_to_return_type(out, dtype, return_type)
    if backend == "max":
      if accelerator_count() > 0:
        native_gpu = _native_max_gpu_func(dtype)
        if native_gpu is not None:
          arrays = [_host_numpy_operand(operand, dtype) for operand in operands]
          out = _execute_max_native(eq, arrays, path, Accelerator(), native_gpu)
          return _numpy_to_return_type(out, dtype, return_type)
      else:
        native_cpu = _native_max_cpu_func(dtype)
        if native_cpu is not None:
          arrays = [_host_numpy_operand(operand, dtype) for operand in operands]
          out = _execute_max_native(eq, arrays, path, CPU(), native_cpu)
          return _numpy_to_return_type(out, dtype, return_type)
  return _execute_max_graph(eq, operands, path, backend, dtype=dtype, accum_dtype=accum_dtype, return_type=return_type)


class MaxGraphBackend:
  def graph_spec_for(
    self,
    eq: str,
    shapes: list[tuple[int, ...]],
    path: list[tuple[int, ...]],
  ) -> dict[str, object]:
    return lowering_spec(eq, shapes, path)

  def execute(
    self,
    eq: str,
    shapes: list[tuple[int, ...]],
    path: list[tuple[int, ...]],
    operands: list[object],
  ) -> object:
    _ = shapes
    return _execute_max_graph(eq, [np.asarray(op) for op in operands], path, "max:cpu")
