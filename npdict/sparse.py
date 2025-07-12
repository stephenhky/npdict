
from functools import reduce
from typing import Tuple, Union
import sys
from itertools import product

import numpy as np
import sparse

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

from .wrap import NumpyNDArrayWrappedDict
from .utils import DuplicatedKeyError, WrongArrayDimensionException, WrongArrayShapeException


class SparseArrayWrappedDict(NumpyNDArrayWrappedDict):
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

        self._sparsearray = sparse.DOK(
            tuple(len(l) for l in self._lists_keystrings),
            fill_value=default_initial_value
        )

    def __getitem__(self, item: Tuple[str, ...]) -> float:
        if len(item) != self.tensor_dimensions:
            raise WrongArrayDimensionException(self.tensor_dimensions, len(item))
        indices = self._get_indices(item)
        return self._sparsearray[tuple(indices)]

    def __setitem__(self, key: Tuple[str, ...], value: float) -> None:
        if len(key) != self.tensor_dimensions:
            raise WrongArrayDimensionException(self.tensor_dimensions, len(key))
        indices = self._get_indices(key)
        self._sparsearray[tuple(indices)] = value

    def to_numpy(self) -> np.ndarray:
        return self._sparsearray.todense()

    def to_coo(self) -> sparse.COO:
        return self._sparsearray.to_coo()

    def to_dok(self) -> sparse.DOK:
        return self._sparsearray

    def generate_dict(
            self,
            new_array: Union[np.ndarray, sparse.SparseArray],
            dense: bool=False
    ) -> Self:
        if len(new_array.shape) != self.tensor_dimensions:
            raise WrongArrayDimensionException(self.tensor_dimensions, len(new_array.shape))
        if new_array.shape != self._sparsearray.shape:
            raise WrongArrayShapeException(self._sparsearray.shape, new_array.shape)
        if dense:
            wrapped_dict = NumpyNDArrayWrappedDict(self._lists_keystrings)
            if isinstance(new_array, sparse.SparseArray):
                wrapped_dict._numpyarray = new_array.todense()
            else:
                wrapped_dict._numpyarray = new_array
        else:
            wrapped_dict = SparseArrayWrappedDict(self._lists_keystrings)
            if isinstance(new_array, sparse.SparseArray):
                wrapped_dict._sparsearray = new_array if isinstance(new_array, sparse.DOK) else sparse.DOK(new_array)
            else:
                wrapped_dict._sparsearray = sparse.DOK(new_array)
        return wrapped_dict

    def __repr__(self) -> str:
        return f"<SparseArrayWrappedDict: dimensions ({', '.join(map(str, self.dimension_sizes))})>"

    @classmethod
    def from_dict_given_keywords(
            cls,
            lists_keywords: list[list[str]],
            oridict: dict[Tuple[str, ...], float],
            default_initial_value: float = 0.0
    ) -> Self:
        wrapped_dict = SparseArrayWrappedDict(
            lists_keywords,
            default_initial_value=default_initial_value
        )
        for keywords_tuple in product(*lists_keywords):
            wrapped_dict[keywords_tuple] = oridict.get(keywords_tuple, default_initial_value)
        return wrapped_dict

