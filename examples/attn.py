from __future__ import annotations

import math, moeinsum as mp, monpy as np


# q: [B, Q, H, D]
# k: [B, K, H, D]
# v: [B, K, H, Dv]
#
def _softmax(x, axis=-1):
  x = x - x.max(axis=axis, keepdims=True)
  e = np.exp(x)
  return e / np.sum(e, axis=axis, keepdims=True)


def attn_einsum_bshd(q, k, v, mask=None, scale=None):
  # q: [B, Q, H, D]
  # k: [B, K, H, D]
  # v: [B, K, H, Dv]
  if scale is None:
    scale = 1.0 / np.sqrt(q.shape[-1])

  scores = mp.einsum("bqhd,bkhd->bhqk", q, k) * scale

  if mask is not None:
    # mask can be [B, Q, K], [B, 1, Q, K], or [B, H, Q, K]
    # for [B, Q, K], reshape/broadcast to [B, 1, Q, K].
    scores = scores + mask

  p = _softmax(scores, axis=-1)
  return mp.einsum("bhqk,bkhd->bqhd", p, v)


if __name__ == "__main__":
  B = 2
  Q = 128
  K = 128
  H = 8
  D = 64

  q = np.random.randn(B, Q, H, D)
  k = np.random.randn(B, K, H, D)
  v = np.random.randn(B, K, H, D)

  out = attn_einsum_bshd(q, k, v)
  print("out.shape =", out.shape)
  print(out)

  # complex number
  q = np.asarray(
    [
      [
        [[1.0 + 0.0j, 0.0 + 1.0j]],
        [[0.5 + 0.5j, 1.0 + 0.0j]],
      ]
    ],
    dtype=np.complex64,
  )

  k = np.asarray(
    [
      [
        [[1.0 + 0.0j, 0.0 - 1.0j]],
        [[0.5 + 0.2j, 0.3 + 0.7j]],
      ]
    ],
    dtype=np.complex64,
  )

  v = np.asarray(
    [
      [
        [[10.0 + 1.0j, 0.0 + 2.0j]],
        [[0.0 + 3.0j, 20.0 + 4.0j]],
      ]
    ],
    dtype=np.complex64,
  )

  print("q.shape =", q.shape)
  print("k.shape =", k.shape)
  print("v.shape =", v.shape)

  out = attn_einsum_bshd(q, k, v)

  print("out.shape =", out.shape)
  print(out)
