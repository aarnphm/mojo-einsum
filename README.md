# moeinsum

An einsum implementation in Mojo with a backend-pluggable architecture, opt_einsum-equivalent path optimizer, and a numpy-compatible Python API.

```python
import numpy as np
import moeinsum

a = np.random.randn(3, 4)
b = np.random.randn(4, 5)
c = moeinsum.einsum("ij,jk->ik", a, b)
assert np.allclose(c, a @ b)
```

## What's in v0.1

- **Parser** (`parse.mojo`): full einsum equation grammar — basic, ellipsis, trace, diagonal, implicit output, multi-char-via-int-interning.
- **Plan IR** (`plan.mojo`): backend-agnostic `ContractionPlan` with B/K/M/N classification per JAX's `_einsum`.
- **Path optimizer** (`path.mojo`): native Mojo implementations of opt_einsum's `greedy`, `optimal-DP`, `branch`, `random-greedy`, and `auto` algorithms.
- **Reference backend** (`backends/reference.mojo`): naive nested-loop einsum, the correctness golden.
- **Unary kernels** (`unary.mojo`): layout-only transpose/diagonal views, reduce-sum.
- **Python API**: `einsum`, `einsum_path`, `parse_equation` over numpy / torch / jax / mlx / anything with `__dlpack__`. Per-signature LRU cache.
- **Bench CLI**: `python -m moeinsum.bench` with JSON output.
- **Tests**: 231 numpy-parity / parser / path / cache / interop cases. 3 framework-tests skip when torch/jax/mlx not installed.

## Docs

- [`docs/notation.md`](docs/notation.md) — einsum notation primer.
- [`docs/derivations.md`](docs/derivations.md) — BMM lowering math, contraction-tree cost models, GETT, √K accumulation rule.
- [`docs/perf.md`](docs/perf.md) — tuning guide, backend selection, profile triage.
- [`docs/comparisons.md`](docs/comparisons.md) — scorecard vs NumPy / PyTorch / JAX / cuTENSOR / TBLIS.

## Install

This project uses [mohaus](https://github.com/aarnphm/mohaus)

```bash
uv pip install -e .
python -c "import moeinsum; import numpy as np; print(moeinsum.einsum('ij,jk->ik', np.eye(3), np.eye(3)))"
```

## Roadmap

| Phase | Status  | What                                                                                       |
| ----- | ------- | ------------------------------------------------------------------------------------------ |
| P0    | ✅      | Scaffolding                                                                                |
| P1    | ✅      | Reference backend + parser + plan + numpy bridge                                           |
| P2    | ✅      | Parser polish (ellipsis, trace, diagonal, implicit output, multi-char)                     |
| P3    | ✅      | Unary kernels (transpose / diagonal / sum / trace)                                         |
| P4    | ✅      | Path optimizer: greedy + optimal-DP + auto + random-greedy + branch family                 |
| P5    | ⏳      | `MaxBackend` dispatching to `linalg.batched_matmul` (needs TileTensor FFI)                 |
| P6    | ✅      | Multi-operand orchestration (working-set semantics); ContractionContext arena deferred     |
| P7    | ✅      | JIT plan cache (Python-side LRU, keyed by eq+shape+optimize)                               |
| P8    | ✅\*    | DLPack interop (first pass): torch / jax / mlx / cupy / tensorflow via DLPack + np.asarray |
| P9    | ✅\*    | Precision (parameters wired; real low-precision lands with MaxBackend)                     |
| P10   | ⏳      | GPU dispatch validation (target="gpu" via batched_matmul)                                  |
| P11   | ⏳      | `NativeOptimizedBackend` CPU: GETT-style fused permute (TBLIS)                             |
| P12   | ⏳      | `NativeOptimizedBackend` GPU SM90: WGMMA + TMA + fused permute                              |
| P13   | ✅      | Benchmark CLI                                                                              |
| P14   | ⏳      | `MaxGraphBackend`: whole-graph fusion via `max.graph`                                      |
| P15   | ✅      | Docs (notation / derivations / perf / comparisons), editor-reviewed                        |

✅ shipped · ✅\* shipped first pass (real polish deferred to P5) · ⏳ pending.

See [`progress.md`](progress.md) for the local working notes (gitignored).

## Architecture

We create a small plan IR to let backend decides how to consume this.

- The parser produces an `EinsumEquation` IR with int-interned labels.
- The path optimizer chooses a pairwise contraction order.
- The plan builder classifies each step's dims into the four-bucket B/K/M/N taxonomy.
- A backend consumes the plan and decides how each step executes
  - the `reference` backend uses a naive global-index loop
  - `max` dispatches to MAX's `linalg.batched_matmul`;
  - `native` runs our own SIMD/GPU kernels with GETT-style fused permute;
  - `max_graph` lifts the plan to a MAX graph for whole-graph fusion.

See [`docs/derivations.md`](docs/derivations.md) §1 for the BMM-lowering proof.

## License

Apache 2.0. See [`LICENSE`](LICENSE).
