# moeinsum

An einsum implementation in Mojo with a backend-pluggable architecture, opt_einsum-equivalent path optimizer, and a numpy-compatible Python API.

```python
import numpy as np
import moeinsum

# Bellman matrix chain. Naive left-to-right pairs `(AB)C` and pays
# ~2e7 FLOPs. `auto` picks `A(BC)` and pays ~2e5 - 100x cheaper.
A = np.random.randn(100, 1)
B = np.random.randn(1, 100_000)
C = np.random.randn(100_000, 1)

moeinsum.einsum_path("ij,jk,kl->il", A.shape, B.shape, C.shape, optimize="auto")
# -> [(1, 2), (0, 1)]    <- (BC) first, then A*(BC)

out = moeinsum.einsum("ij,jk,kl->il", A, B, C)            # default optimize="auto"
# -> shape (100, 1), agrees with np.einsum within 1e-10
```

Pass tensors from any DLPack-capable framework - `numpy`, `torch`,
`jax`, `mlx`, `cupy`, `tensorflow`. The return type mirrors the first
operand's framework (torch in -> torch out), or use `return_type="numpy"`
to force.

## What's in v0.1

- **Parser** (`parse.mojo`): full einsum equation grammar - basic, ellipsis, trace, diagonal, implicit output, multi-char-via-int-interning.
- **Plan IR** (`plan.mojo`): backend-agnostic `ContractionPlan` with B/K/M/N classification per JAX's `_einsum`.
- **Path optimizer** (`path.mojo`): native Mojo implementations of opt_einsum's `greedy`, `optimal-DP`, `branch`, `random-greedy`, and `auto` algorithms.
- **Reference backend** (`backends/reference.mojo`): naive nested-loop einsum, the correctness golden.
- **MAX Graph backend** (`python/moeinsum/_max_backend.py`): executable `backend="max[:cpu|gpu]"` for the BMM-lowerable subset, including ellipsis, size-1 broadcast, unary transpose, and reduce-sum. Diagonal / trace still route to `reference`.
- **Unary kernels** (`unary.mojo`): layout-only transpose/diagonal views, reduce-sum.
- **Python API**: `einsum`, `einsum_path`, `parse_equation` over numpy / torch / jax / mlx / anything with `__dlpack__`. Per-signature LRU cache.
- **Bench CLI**: `moeinsum-bench` script (installed by `pip install -e .`), JSON output, optional stderr progress bars, and compare rows for numpy / opt_einsum / jax / torch / mlx.
- **Tests**: 481 passing Python cases in the current local run: numpy-parity / JAX-corpus / opt_einsum-path-parity / parser / path / branch / explicit-path / cache / interop / bench-CLI / hypothesis-property / MAX-backend coverage. Optional framework and MAX-runtime tests skip when the dependency or dlopen path is unavailable.

## Docs

- [`docs/notation.md`](docs/notation.md) - einsum notation primer.
- [`docs/derivations.md`](docs/derivations.md) - BMM lowering math, contraction-tree cost models, GETT, $\sqrt{K}$ accumulation rule.
- [`docs/perf.md`](docs/perf.md) - tuning guide, backend selection, profile triage.
- [`docs/comparisons.md`](docs/comparisons.md) - scorecard vs NumPy / PyTorch / JAX / cuTENSOR / TBLIS.
- [`docs/ffi.md`](docs/ffi.md) - FFI cutover design-spike for the Mojo P5/P11/P12 work.
- [`docs/plan-verification.md`](docs/plan-verification.md) - claim -> test map for the plan's `## Verification` section.

## Install

This project uses [mohaus](https://github.com/aarnphm/mohaus)

```bash
uv pip install -e .
python -c "import moeinsum; import numpy as np; print(moeinsum.einsum('ij,jk->ik', np.eye(3), np.eye(3)))"
```

## Roadmap

| Phase | Status   | What                                                                                                 |
| ----- | -------- | ---------------------------------------------------------------------------------------------------- |
| P0    | done     | Scaffolding                                                                                          |
| P1    | done     | Reference backend + parser + plan + numpy bridge                                                     |
| P2    | done     | Parser polish (ellipsis, trace, diagonal, implicit output, multi-char)                               |
| P3    | done     | Unary kernels (transpose / diagonal / sum / trace)                                                   |
| P4    | done     | Path optimizer: greedy + optimal-DP + auto + random-greedy(-N) + branch family                       |
| P5    | partial  | Python MAX Graph backend shipped; Mojo `TileTensor` / `linalg.batched_matmul` cutover pending        |
| P6    | done     | Multi-operand orchestration (working-set semantics); ContractionContext arena deferred               |
| P7    | done     | JIT plan cache (Python-side LRU, keyed by eq+shape+optimize)                                         |
| P8    | done     | DLPack interop: dtype-preserving in/out, framework-native return (torch in -> torch out)             |
| P9    | done\*   | Precision parameters wired; bf16 covered on the MAX Graph path                                       |
| P10   | done     | Python-side `backend="max:gpu"` validation on B200; Mojo-side `target="gpu"` waits on P5             |
| P11   | skeleton | Native CPU GETT skeleton at `src/einsum/backends/native.mojo`; kernel work pending                   |
| P12   | skeleton | Native GPU SM90 skeleton in the same module; WGMMA kernel pending                                    |
| P13   | done     | Benchmark CLI (`moeinsum-bench` script)                                                              |
| P14   | partial  | `MaxGraphBackend`: plan-to-graph spec + executable bridge shipped; whole-graph fusion polish pending |
| P15   | done     | Docs (notation / derivations / perf / comparisons / ffi), editor-reviewed                            |

`done` = shipped; `done\*` = shipped first pass, real polish deferred to P5; `partial` = useful executable surface shipped with a named lower-level cutover still pending; `skeleton` = scaffolding landed, full implementation pending; `pending` = not shipped yet.

See [`progress.md`](progress.md) for the local working notes (gitignored).

## Architecture

We create a small plan IR to let backend decides how to consume this.

- The parser produces an `EinsumEquation` IR with int-interned labels.
- The path optimizer chooses a pairwise contraction order.
- The plan builder classifies each step's dims into the four-bucket B/K/M/N taxonomy.
- A backend consumes the plan and decides how each step executes
  - the `reference` backend uses a naive global-index loop
  - `max` builds a MAX Graph over pairwise batched-matmul lowerings;
  - `native` will run our own SIMD/GPU kernels with GETT-style fused permute;
  - `max_graph` exposes the MAX Graph bridge and plan-spec inspection surface.

See [`docs/derivations.md`](docs/derivations.md) Section 1 for the BMM-lowering proof.

## License

Apache 2.0. See [`LICENSE`](LICENSE).
