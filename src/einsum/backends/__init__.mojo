"""Backends consume a `ContractionPlan` and execute it.

All backends implement the same logical contract (single
`execute(plan, operands) -> result` entry point), but specialize the
*how*: the reference backend walks all indices naively for correctness
testing; `max_kernels` lowers to `linalg.batched_matmul`; `native` ships
our own SIMD/GPU kernels; `max_graph` lifts to a MAX graph for
whole-graph fusion. Per the plan, only `reference` and `max_kernels`
ship in P1.
"""
