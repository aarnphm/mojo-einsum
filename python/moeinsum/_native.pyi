def parse_equation(eq: str) -> dict[str, object]: ...
def parse_equation_expanded(
  eq: str,
  operand_shapes: list[list[int]],
) -> dict[str, object]: ...
def einsum_reference(
  eq: str,
  operands_flat: list[list[float]],
  operand_shapes: list[list[int]],
) -> tuple[list[float], list[int]]: ...
def einsum_native(
  eq: str,
  operands_flat: list[list[float]],
  operand_shapes: list[list[int]],
  path: list[tuple[int, ...]],
) -> tuple[list[float], list[int]]: ...
def einsum_max_f32_cpu_ptrs(
  eq: str,
  payload: dict[str, object],
  path: list[tuple[int, ...]],
) -> None: ...
def einsum_max_f64_cpu_ptrs(
  eq: str,
  payload: dict[str, object],
  path: list[tuple[int, ...]],
) -> None: ...
def einsum_path(
  eq: str,
  operand_shapes: list[list[int]],
) -> list[tuple[int, ...]]: ...
def einsum_compute_path(
  eq: str,
  operand_shapes: list[list[int]],
  algorithm: str,
) -> list[tuple[int, ...]]: ...
def path_cost(
  eq: str,
  operand_shapes: list[list[int]],
  path: list[tuple[int, ...]],
) -> dict[str, object]: ...
