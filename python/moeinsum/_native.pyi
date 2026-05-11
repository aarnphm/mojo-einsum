def parse_equation(eq: str) -> dict[str, object]: ...
def einsum_reference(
  eq: str,
  operands_flat: list[list[float]],
  operand_shapes: list[list[int]],
) -> tuple[list[float], list[int]]: ...
def einsum_path(
  eq: str,
  operand_shapes: list[list[int]],
) -> list[tuple[int, ...]]: ...
def einsum_compute_path(
  eq: str,
  operand_shapes: list[list[int]],
  algorithm: str,
) -> list[tuple[int, ...]]: ...
def max_graph_spec(
  eq: str,
  operand_shapes: list[tuple[int, ...]],
  path: list[tuple[int, ...]],
) -> dict[str, object]: ...
