"""moeinsum core: parser, plan IR, and unary kernels.

Sub-packages:
  - `parse`: equation parser, `EinsumEquation` IR.
  - `plan`: backend-agnostic `ContractionPlan` IR and dim classification.
  - `backends`: pluggable execution backends consuming a plan.

Public-facing API lives in `lib.mojo` (the FFI entry point); this package
is the implementation library.
"""
