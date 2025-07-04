
from typing import Tuple, Generator
import sys
from itertools import product
from functools import reduce

import numpy as np

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

from .utils import DuplicatedKeyError, WrongArrayDimensionException, WrongArrayShapeException


class NumpyNDArrayWrappedDict(dict):
    def __init__(
            self,
            lists_keystrings: list[list[str]],
            default_initial_value: float=0.0
    ):
        super(dict, self).__init__()
        for list_keystrings in lists_keystrings:
            if (len(list_keystrings)) != len(set(list_keystrings)):
                raise DuplicatedKeyError()
        self._lists_keystrings = lists_keystrings
        self._keystrings_to_indices = [
            {
                keyword: idx for idx, keyword in enumerate(list_keystrings)
            }
            for list_keystrings in self._lists_keystrings
        ]

        self._tensor_dimensions = len(self._lists_keystrings)
        self._dimension_sizes = [len(l) for l in self._lists_keystrings]
        self._total_size = reduce(lambda a, b: a*b, self._dimension_sizes)

        self._numpyarray = np.empty(tuple(len(l) for l in self._lists_keystrings))
        self._numpyarray.fill(default_initial_value)

    def _get_indices(self, item: Tuple[str, ...]) -> list[int]:
        return [
            mapping[keyword]
            for mapping, keyword in zip(self._keystrings_to_indices, item)
        ]

    def __getitem__(self, item: Tuple[str, ...]) -> float:
        if len(item) != self.tensor_dimensions:
            raise WrongArrayDimensionException(self.tensor_dimensions, len(item))
        indices = self._get_indices(item)
        return self._numpyarray[tuple(indices)]

    def __setitem__(self, key: Tuple[str, ...], value: float) -> None:
        if len(key) != self.tensor_dimensions:
            raise WrongArrayDimensionException(self.tensor_dimensions, len(key))
        indices = self._get_indices(key)
        self._numpyarray[tuple(indices)] = value

    def update(self, new_dict: dict):
        raise TypeError("We cannot update this kind of dict this way!")

    def __iter__(self) -> Generator[Tuple[str, ...], None, None]:
        for keywords_tuple in product(*self._lists_keystrings):
            yield keywords_tuple

    def keys(self):
        return list(self.__iter__())

    def values(self):
        return [self.__getitem__(keywords_tuple) for keywords_tuple in self.__iter__()]

    def items(self):
        return [
            (keywords_tuple, self.__getitem__(keywords_tuple))
            for keywords_tuple in self.__iter__()
        ]

    def to_numpy(self) -> np.ndarray:
        return self._numpyarray

    def generate_dict(self, nparray: np.ndarray) -> Self:
        if len(nparray.shape) != self.tensor_dimensions:
            raise WrongArrayDimensionException(self.tensor_dimensions, len(nparray.shape))
        if nparray.shape != self._numpyarray.shape:
            raise WrongArrayShapeException(self._numpyarray.shape, nparray.shape)
        wrapped_dict = NumpyNDArrayWrappedDict(self._lists_keystrings)
        wrapped_dict._numpyarray = nparray
        return wrapped_dict

    def __repr__(self) -> str:
        return f"<NumpyNDArrayWrappedDict: dimensions ({', '.join(map(str, self.dimension_sizes))})>"

    def __str__(self) -> str:
        return self.__repr__()

    def __len__(self) -> int:
        return self._total_size

    def to_dict(self) -> dict[Tuple[str, ...], float]:
        return {
            keywords_tuple: value for keywords_tuple, value in self.items()
        }

    @classmethod
    def from_dict_given_keywords(
            cls,
            lists_keywords: list[list[str]],
            oridict: dict[Tuple[str, ...], float],
            default_initial_value: float = 0.0
    ) -> Self:
        wrapped_dict = NumpyNDArrayWrappedDict(
            lists_keywords,
            default_initial_value=default_initial_value
        )
        for keywords_tuple in product(*lists_keywords):
            wrapped_dict[keywords_tuple] = oridict.get(keywords_tuple, default_initial_value)
        return wrapped_dict

    @classmethod
    def from_dict(
            cls,
            oridict: dict[Tuple[str, ...], float],
            default_initial_value: float = 0.0
    ) -> Self:
        nbdims = len(next(iter(oridict)))
        lists_keystrings = [
            list(set(keystring[i] for keystring in oridict.keys()))
            for i in range(nbdims)
        ]
        return cls.from_dict_given_keywords(
            lists_keystrings,
            oridict,
            default_initial_value=default_initial_value
        )

    @property
    def tensor_dimensions(self) -> int:
        return self._tensor_dimensions

    @property
    def dimension_sizes(self) -> list[int]:
        return self._dimension_sizes
