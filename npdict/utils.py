

from typing import Tuple


class DuplicatedKeyError(Exception):
    """
    Exception raised when duplicate keys are found in a list of keys.
    
    This exception is raised when attempting to create a dictionary with duplicate keys,
    which would lead to ambiguous behavior.
    """
    def __init__(self):
        self.message = "Duplicated keys!"


class WrongArrayDimensionException(Exception):
    """
    Exception raised when an array has an incorrect number of dimensions.
    
    This exception is raised when the number of dimensions in an array does not match
    the expected number of dimensions.
    """
    def __init__(self, expected_dimension: int, given_dimensions: int):
        self.message = f"Expected dimension: {expected_dimension}, but {given_dimensions} dimensions are given!"


class WrongArrayShapeException(Exception):
    """
    Exception raised when an array has an incorrect shape.
    
    This exception is raised when the shape of an array does not match the expected shape,
    even if the number of dimensions is correct.
    """
    def __init__(self, expected_shape: Tuple[int, ...], given_shape: Tuple[int, ...]):
        self.message = f"Expected shape: {', '.join(str(dim_len) for dim_len in expected_shape)}, but the given array shape is {', '.join(str(dim_len) for dim_len in given_shape)}!"
