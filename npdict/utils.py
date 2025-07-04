

from typing import Tuple


class DuplicatedKeyError(Exception):
    def __init__(self):
        self.message = "Duplicated keys!"


class WrongArrayDimensionException(Exception):
    def __init__(self, expected_dimension: int, given_dimensions: int):
        self.message = f"Expected dimension: {expected_dimension}, but {given_dimensions} dimensions are given!"


class WrongArrayShapeException(Exception):
    def __init__(self, expected_shape: Tuple[int, ...], given_shape: Tuple[int, ...]):
        self.message = f"Expected shape: {', '.join(list(expected_shape))}, but the given array shape is {', '.join(list(given_shape))}!"
