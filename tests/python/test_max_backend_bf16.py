"""Plan section 7 row 7 - K > 64 bf16 inputs with fp32 accumulator drift.

The reference backend always accumulates in fp64, so the drift law for
bf16 inputs can't be exercised there - hence the plan row was blocked
until a real bf16-capable backend shipped. `backend="max:cpu"` is that
backend: MAX Graph's matmul accumulates bf16 inputs in fp32 internally
and casts back to bf16 for the output.

Two assertions pin the claim:

1. **Bounded drift at K=128 and K=256.** Relative drift against the
   fp32-computed reference stays under 1% - which is well below
   the bf16-accum failure mode would produce (sqrt(K)*eps_bf16 ~ 13%
   at K=256). The output is itself rounded to bf16, so the floor is
   eps_bf16 ~ 0.008, not zero.

2. **Sub-linear growth in K.** Absolute drift from K=64 to K=256 grows
   ~2x (sqrt-K-law signature of fp32-accum), not 4x (linear-K-law
   signature of bf16-accum). An assertion at 3x slack rejects the
   bf16-accum regression cleanly while tolerating per-seed wobble
   (empirical: ratio median 2.04 across 10 seeds).

See [[derivations#4. Low-precision accumulation]] for the sqrt(K)
rounding-error growth derivation.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
  import ml_dtypes
  from moeinsum._max_graph import is_loadable as _is_loadable  # noqa: PLC0415

  HAS_BF16 = _is_loadable()
  BFLOAT16 = np.dtype(ml_dtypes.bfloat16) if HAS_BF16 else None
except ImportError:
  HAS_BF16 = False
  BFLOAT16 = None  # type: ignore[assignment]

pytestmark = pytest.mark.skipif(
  not HAS_BF16,
  reason="max.graph + ml_dtypes required and must dlopen cleanly in this env",
)


@pytest.mark.parametrize("k", [128, 256])
def test_bf16_matmul_drift_within_one_percent(k: int) -> None:
  """K > 64 bf16 input through MAX gives <1% relative drift.

  Reference is the same bf16-quantized inputs cast to fp32 and
  contracted in fp32 - so we measure only the accumulator's choice
  and the bf16-output roundoff, not the input quantization.
  """
  import moeinsum

  rng = np.random.default_rng(0)
  a = rng.standard_normal((64, k)).astype(BFLOAT16)
  b = rng.standard_normal((k, 64)).astype(BFLOAT16)

  out = moeinsum.einsum("ij,jk->ik", a, b, backend="max:cpu")
  assert out.dtype == BFLOAT16

  ref = a.astype(np.float32) @ b.astype(np.float32)
  drift = float(np.max(np.abs(out.astype(np.float32) - ref)))
  scale = float(np.max(np.abs(ref)))
  rel = drift / scale
  assert rel < 0.01, (
    f"K={k}: relative drift {rel:.4f} exceeds 1%. "
    f"Either MAX dropped to bf16 accumulation (sqrt(K)*eps_bf16 ~ {(k**0.5) * 0.0078:.3f} at K={k}) "
    f"or there is a numerical regression elsewhere. drift={drift:.4e} scale={scale:.4e}"
  )


def test_bf16_drift_grows_sublinearly_in_k() -> None:
  """Absolute drift from K=64 to K=256 grows ~2x (fp32-accum, sqrt-K).

  Linear-K growth (~4x) would indicate a bf16 accumulator regression.
  We assert <3x to leave headroom for per-element wobble while still
  catching the failure mode cleanly.
  """
  import moeinsum

  def drift_at(k: int) -> float:
    rng = np.random.default_rng(0)
    a = rng.standard_normal((32, k)).astype(BFLOAT16)
    b = rng.standard_normal((k, 32)).astype(BFLOAT16)
    out = moeinsum.einsum("ij,jk->ik", a, b, backend="max:cpu").astype(np.float32)
    ref = a.astype(np.float32) @ b.astype(np.float32)
    return float(np.max(np.abs(out - ref)))

  d64 = drift_at(64)
  d256 = drift_at(256)
  ratio = d256 / max(d64, 1e-10)
  assert ratio < 3.0, (
    f"bf16 drift grew {ratio:.2f}x from K=64 to K=256. sqrt-K law predicts ~2x "
    f"(fp32-accum, the correct default); linear-K would predict ~4x "
    f"(bf16-accum regression). d64={d64:.4e} d256={d256:.4e}"
  )


def test_bf16_dtype_round_trips_through_backend() -> None:
  """bf16 in, bf16 out - no silent fp32 demotion at the API boundary."""
  import moeinsum

  rng = np.random.default_rng(1)
  a = rng.standard_normal((4, 4)).astype(BFLOAT16)
  b = rng.standard_normal((4, 4)).astype(BFLOAT16)
  out = moeinsum.einsum("ij,jk->ik", a, b, backend="max:cpu")
  assert out.dtype == BFLOAT16


def test_bf16_size_one_K_is_exact() -> None:
  """K=1 reduces to a scalar product per (i,j) - no accumulator drift possible.

  Sanity check on the bridge: any non-zero drift here would mean we
  broke the byte-layout assumption between bf16 and uint16.
  """
  import moeinsum

  rng = np.random.default_rng(2)
  a = rng.standard_normal((8, 1)).astype(BFLOAT16)
  b = rng.standard_normal((1, 8)).astype(BFLOAT16)
  out = moeinsum.einsum("ij,jk->ik", a, b, backend="max:cpu")
  ref_bf16 = (a.astype(np.float32) @ b.astype(np.float32)).astype(BFLOAT16)
  np.testing.assert_array_equal(out, ref_bf16)
