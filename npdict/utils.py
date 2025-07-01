

class DuplicatedKeyError(Exception):
    def __init__(self):
        self.message = "Duplicated keys!"


class WrongArrayDimensionException(Exception):
    def __init__(self, expected_dimension: int, given_dimensions: int):
        self.message = f"Expected dimension: {expected_dimension}, but {given_dimensions} dimensions are given!"
