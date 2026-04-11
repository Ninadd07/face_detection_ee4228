from __future__ import annotations


def normalize_input_size(value: int | list[int] | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if isinstance(value, (list, tuple)) and len(value) == 2:
        height, width = int(value[0]), int(value[1])
        return height, width
    raise ValueError(f"Unsupported input size: {value!r}")
