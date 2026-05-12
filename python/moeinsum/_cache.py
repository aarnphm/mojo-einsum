"""Per-signature plan cache.

The Mojo FFI re-parses the equation, expands ellipsis, builds the
plan, and dispatches the kernel on every call. For small einsums the
parse + plan layer often dominates. This module caches the
Python-visible setup keyed by immutable call signatures, so a hot call
site does a single hash lookup before dispatching.

Once the Mojo side exposes a `plan_einsum(...) -> handle` /
`execute_plan(handle, operands)` split, the cache should hold the
Mojo-side handle directly and the FFI re-parse goes away. For now the
cache wraps the Python-visible work; it is still materially faster on
repeated calls vs. cold-cache for small inputs.

LRU eviction with a configurable size cap. Default 512 entries is enough
for any realistic ML workload - a model with 100 distinct einsum
call-sites x 5 dtype/shape variants is the upper bound.
"""

from __future__ import annotations

from collections import OrderedDict
from threading import RLock


class _PlanCache:
  """Thread-safe LRU."""

  def __init__(self, max_entries: int = 512) -> None:
    self._lock = RLock()
    self._max = max_entries
    self._data: OrderedDict[tuple[object, ...], object] = OrderedDict()

  def get(self, key: tuple[object, ...]) -> object | None:
    with self._lock:
      value = self._data.get(key)
      if value is None:
        return None
      self._data.move_to_end(key)
      return value

  def put(self, key: tuple[object, ...], value: object) -> None:
    with self._lock:
      self._data[key] = value
      self._data.move_to_end(key)
      while len(self._data) > self._max:
        self._data.popitem(last=False)

  def clear(self) -> None:
    with self._lock:
      self._data.clear()

  def __len__(self) -> int:
    with self._lock:
      return len(self._data)


# Module-level default cache. Tests can replace this via
# `moeinsum._cache.PLAN_CACHE = _PlanCache(max_entries=...)`.
PLAN_CACHE = _PlanCache()
