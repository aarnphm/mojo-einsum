"""MAX Graph backend for supported einsum contractions.

This is the first executable `backend="max"` path. It deliberately lives in
Python instead of the Mojo FFI layer: the installed MAX Graph API can already
compile the B/K/M/N lowering we need, and keeping that path here lets us validate
shape semantics on CPU/GPU before threading TileTensor handles through Mojo.

Supported v0:
  - no ellipsis
  - no repeated labels inside a single operand
  - pairwise path steps lowered to batched matmul
  - unary transpose / reduce-sum steps

Unsupported cases raise `NotImplementedError` instead of silently falling back
to reference. `backend="max"` should mean "MAX executed this", not "the word MAX
appeared near a NumPy result". Extremely small but important bit of honesty.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from functools import reduce
from operator import mul
from threading import RLock
from typing import TYPE_CHECKING

import numpy as np

from . import _max_graph

if TYPE_CHECKING:
  from max.driver import Buffer, Device
  from max.dtype import DType
  from max.engine import Model
  from max.graph import Graph, TensorValue


@dataclass(frozen=True)
class _Node:
  value: TensorValue
  labels: str
  shape: tuple[int, ...]


@dataclass
class _Executable:
  model: Model
  device: Device


_MODEL_CACHE_MAX = 512
_MODEL_CACHE: OrderedDict[tuple[object, ...], _Executable] = OrderedDict()
_MODEL_CACHE_LOCK = RLock()


def _parse_equation(eq: str) -> tuple[list[str], str]:
  if "." in eq:
    raise NotImplementedError("backend='max' does not support ellipsis yet")
  if "->" in eq:
    lhs, output = eq.split("->", 1)
  else:
    lhs = eq
    counts: dict[str, int] = {}
    for c in lhs.replace(",", ""):
      counts[c] = counts.get(c, 0) + 1
    output = "".join(sorted(c for c, count in counts.items() if count == 1))
  inputs = lhs.split(",")
  return inputs, output


def _reject_repeated_labels(labels: str) -> None:
  if len(set(labels)) != len(labels):
    raise NotImplementedError("backend='max' does not support repeated labels/diagonals yet")


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


def _product(values: list[int]) -> int:
  if not values:
    return 1
  return reduce(mul, values, 1)


def _sizes_for(labels: str, sizes: dict[str, int]) -> list[int]:
  return [sizes[label] for label in labels]


def _permute_if_needed(value: TensorValue, current: str, desired: str) -> TensorValue:
  if current == desired:
    return value
  from max.graph import ops  # noqa: PLC0415

  dims = [current.index(label) for label in desired]
  return ops.permute(value, dims)


def _reshape(value: TensorValue, shape: list[int]) -> TensorValue:
  from max.graph import ops  # noqa: PLC0415

  return ops.reshape(value, shape)


def _reduce_out_labels(node: _Node, keep: set[str]) -> _Node:
  from max.graph import ops  # noqa: PLC0415

  _reject_repeated_labels(node.labels)
  value = node.value
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
  return "".join(dict.fromkeys(label for label in lhs + rhs if label in future))


def _lower_pair(lhs: _Node, rhs: _Node, out_labels: str) -> _Node:
  from max.graph import ops  # noqa: PLC0415

  _reject_repeated_labels(lhs.labels)
  _reject_repeated_labels(rhs.labels)

  sizes = _label_sizes(lhs.labels, lhs.shape)
  rhs_sizes = _label_sizes(rhs.labels, rhs.shape)
  for label, dim in rhs_sizes.items():
    previous = sizes.get(label)
    if previous is not None and previous != dim:
      raise ValueError(f"size conflict on label {label!r}: {previous} vs {dim}")
    sizes[label] = dim

  cls = _max_graph.classify_pair(lhs.labels, rhs.labels, out_labels)
  batch = "".join(cls.batch)
  contract = "".join(cls.contract)
  free_lhs = "".join(cls.free_lhs)
  free_rhs = "".join(cls.free_rhs)

  lhs_order = batch + free_lhs + contract
  rhs_order = batch + contract + free_rhs
  lhs_value = _permute_if_needed(lhs.value, lhs.labels, lhs_order)
  rhs_value = _permute_if_needed(rhs.value, rhs.labels, rhs_order)

  batch_shape = _sizes_for(batch, sizes)
  m_shape = _sizes_for(free_lhs, sizes)
  k_shape = _sizes_for(contract, sizes)
  n_shape = _sizes_for(free_rhs, sizes)
  m = _product(m_shape)
  k = _product(k_shape)
  n = _product(n_shape)

  lhs_value = _reshape(lhs_value, [*batch_shape, m, k])
  rhs_value = _reshape(rhs_value, [*batch_shape, k, n])
  out = ops.matmul(lhs_value, rhs_value)

  natural_labels = batch + free_lhs + free_rhs
  natural_shape = [*batch_shape, *m_shape, *n_shape]
  out = _reshape(out, natural_shape)
  out = _permute_if_needed(out, natural_labels, out_labels)
  return _Node(
    value=out,
    labels=out_labels,
    shape=tuple(_sizes_for(out_labels, sizes)),
  )


def _lower_graph(
  eq: str,
  shapes: list[tuple[int, ...]],
  path: list[tuple[int, ...]],
  dtype: DType,
  device: Device,
) -> Graph:
  from max.graph import DeviceRef, Graph, TensorType  # noqa: PLC0415

  inputs, final_output = _parse_equation(eq)
  if len(inputs) != len(shapes):
    raise ValueError(f"equation has {len(inputs)} operands but got {len(shapes)} shapes")
  for labels in inputs:
    _reject_repeated_labels(labels)

  input_types = [TensorType(dtype, shape=shape, device=DeviceRef.from_device(device)) for shape in shapes]

  with Graph("moeinsum_max", input_types=input_types) as graph:
    working = [
      _Node(value=graph_input.tensor, labels=labels, shape=shape)
      for graph_input, labels, shape in zip(graph.inputs, inputs, shapes, strict=True)
    ]

    for step in path:
      if len(step) == 1:
        (idx,) = step
        future = set(final_output)
        for j, node in enumerate(working):
          if j != idx:
            future.update(node.labels)
        working[idx] = _reduce_out_labels(working[idx], future)
        continue

      li, ri = step
      lhs = working[li]
      rhs = working[ri]
      others = [node.labels for j, node in enumerate(working) if j not in (li, ri)]
      out_labels = _output_labels_for_pair(lhs.labels, rhs.labels, final_output, others)
      out = _lower_pair(lhs, rhs, out_labels)

      for idx in sorted((li, ri), reverse=True):
        del working[idx]
      working.append(out)

    if len(working) != 1:
      raise ValueError(f"contraction path leaves {len(working)} tensors; expected 1")

    result = _reduce_out_labels(working[0], set(final_output))
    result_value = _permute_if_needed(result.value, result.labels, final_output)
    graph.output(result_value)
    return graph


def _select_device(backend: str) -> Device:
  from max.driver import CPU, Accelerator, accelerator_count  # noqa: PLC0415

  if backend == "max:cpu":
    return CPU()
  if backend == "max:gpu":
    if accelerator_count() == 0:
      raise RuntimeError("backend='max:gpu' requested but MAX reports no accelerator")
    return Accelerator()
  if accelerator_count() > 0:
    return Accelerator()
  return CPU()


def _compile(
  eq: str,
  shapes: list[tuple[int, ...]],
  path: list[tuple[int, ...]],
  dtype: DType,
  backend: str,
) -> _Executable:
  from max.engine import InferenceSession  # noqa: PLC0415

  device = _select_device(backend)
  graph = _lower_graph(eq, shapes, path, dtype, device)
  session = InferenceSession(devices=[device])
  return _Executable(model=session.load(graph), device=device)


def _max_dtype_for(dtype: np.dtype) -> DType:
  from max.dtype import DType  # noqa: PLC0415

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


def _input_buffer(arr: np.ndarray, device: Device) -> Buffer:
  """Numpy -> MAX Buffer with a uint16 detour for bf16.

  Numpy's DLPack export rejects ml_dtypes.bfloat16 (the bridge only
  carries IEEE-recognised dtypes). bf16 and uint16 share a byte layout,
  so we hand MAX the uint16 view and reinterpret the metadata as
  bf16 on the MAX side. Round-trip-safe because no arithmetic happens
  between the two views.
  """
  from max.driver import Buffer  # noqa: PLC0415
  from max.dtype import DType  # noqa: PLC0415

  if arr.dtype.name == "bfloat16":
    u16 = np.ascontiguousarray(arr.view(np.uint16))
    return Buffer.from_numpy(u16).view(DType.bfloat16, arr.shape).to(device)
  return Buffer.from_numpy(arr).to(device)


def _read_output(buf: Buffer) -> np.ndarray:
  """MAX Buffer -> numpy with a uint16 detour for bf16 (inverse of `_input_buffer`)."""
  from max.driver import CPU  # noqa: PLC0415
  from max.dtype import DType  # noqa: PLC0415

  host = buf.to(CPU())
  if buf.dtype == DType.bfloat16:
    import ml_dtypes  # noqa: PLC0415

    u16_buf = host.view(DType.uint16, host.shape)
    return u16_buf.to_numpy().view(np.dtype(ml_dtypes.bfloat16))
  return host.to_numpy()


def execute_max(
  eq: str,
  arrays: list[np.ndarray],
  path: list[tuple[int, ...]],
  backend: str,
) -> np.ndarray:
  from max.driver import Buffer  # noqa: PLC0415

  if not arrays:
    raise ValueError("einsum requires at least one operand")
  dtype = np.result_type(*arrays)
  max_dtype = _max_dtype_for(dtype)

  max_arrays = [np.ascontiguousarray(array.astype(dtype, copy=False)) for array in arrays]
  shapes = [tuple(array.shape) for array in max_arrays]
  key = (eq, tuple(shapes), str(dtype), tuple(path), backend)
  with _MODEL_CACHE_LOCK:
    executable = _MODEL_CACHE.get(key)
    if executable is None:
      executable = _compile(eq, shapes, path, max_dtype, backend)
      _MODEL_CACHE[key] = executable
      while len(_MODEL_CACHE) > _MODEL_CACHE_MAX:
        _MODEL_CACHE.popitem(last=False)
    else:
      _MODEL_CACHE.move_to_end(key)

  inputs = [_input_buffer(array, executable.device) for array in max_arrays]
  result = executable.model.execute(*inputs)[0]
  if not isinstance(result, Buffer):
    raise TypeError(f"MAX model returned {type(result).__name__}, expected Buffer")
  out = _read_output(result)
  if out.dtype != dtype:
    out = out.astype(dtype)
  return out
