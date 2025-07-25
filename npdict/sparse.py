
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
    """
    A dictionary-like class that wraps a sparse array.
    
    This class provides a dictionary interface to a sparse array, where the keys are tuples of strings
    and the values are the corresponding elements in the array. The class maintains a mapping between
    string keys and array indices, allowing for more intuitive access to array elements.
    
    This implementation uses sparse arrays instead of NumPy arrays, which is more memory-efficient
    for arrays with many zero values.
    """
    def __init__(
            self,
            lists_keystrings: list[list[str]],
            default_initial_value: float=0.0
    ):
        """
        Initialize a new SparseArrayWrappedDict.
        
        Parameters
        ----------
        lists_keystrings : list[list[str]]
            A list of lists of strings, where each inner list contains the keys for one dimension of the array.
            For example, [['a', 'b'], ['c', 'd']] would create a 2x2 array with keys ('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd').
        default_initial_value : float, optional
            The default value to fill the array with, by default 0.0.
            
        Raises
        ------
        DuplicatedKeyError
            If there are duplicate keys in any of the lists of keys.
        """
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
        """
        Get the value at the specified keys.
        
        Parameters
        ----------
        item : Tuple[str, ...]
            A tuple of string keys, one for each dimension of the array.
            
        Returns
        -------
        float
            The value at the specified keys.
            
        Raises
        ------
        WrongArrayDimensionException
            If the number of keys does not match the number of dimensions in the array.
        """
        if len(item) != self.tensor_dimensions:
            raise WrongArrayDimensionException(self.tensor_dimensions, len(item))
        indices = self._get_indices(item)
        return self._sparsearray[tuple(indices)]

    def __setitem__(self, key: Tuple[str, ...], value: float) -> None:
        """
        Set the value at the specified keys.
        
        Parameters
        ----------
        key : Tuple[str, ...]
            A tuple of string keys, one for each dimension of the array.
        value : float
            The value to set at the specified keys.
            
        Raises
        ------
        WrongArrayDimensionException
            If the number of keys does not match the number of dimensions in the array.
        """
        if len(key) != self.tensor_dimensions:
            raise WrongArrayDimensionException(self.tensor_dimensions, len(key))
        indices = self._get_indices(key)
        self._sparsearray[tuple(indices)] = value

    def to_numpy(self) -> np.ndarray:
        """
        Convert the wrapped sparse array to a dense NumPy array.
        
        Returns
        -------
        np.ndarray
            A dense NumPy array containing the values of the sparse array.
        """
        return self._sparsearray.todense()

    def to_coo(self) -> sparse.COO:
        """
        Convert the wrapped sparse array to a COO (Coordinate) format sparse array.
        
        Returns
        -------
        sparse.COO
            A COO format sparse array containing the same values as the wrapped array.
        """
        return self._sparsearray.to_coo()

    def to_dok(self) -> sparse.DOK:
        """
        Get the underlying DOK (Dictionary of Keys) format sparse array.
        
        Returns
        -------
        sparse.DOK
            The underlying DOK format sparse array.
        """
        return self._sparsearray

    def generate_dict(
            self,
            new_array: Union[np.ndarray, sparse.SparseArray],
            dense: bool=False
    ) -> Self:
        """
        Generate a new dictionary with the same keys but different values.
        
        Parameters
        ----------
        new_array : Union[np.ndarray, sparse.SparseArray]
            The array containing the new values. Can be either a NumPy array or a sparse array.
        dense : bool, optional
            If True, returns a NumpyNDArrayWrappedDict. If False, returns a SparseArrayWrappedDict.
            Default is False.
            
        Returns
        -------
        Union[NumpyNDArrayWrappedDict, SparseArrayWrappedDict]
            A new dictionary with the same keys but different values.
            
        Raises
        ------
        WrongArrayDimensionException
            If the number of dimensions in the array does not match the number of dimensions in the dictionary.
        WrongArrayShapeException
            If the shape of the array does not match the shape of the dictionary.
        """
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
        """
        Return a string representation of the dictionary.
        
        Returns
        -------
        str
            A string representation of the dictionary.
        """
        return f"<SparseArrayWrappedDict: dimensions ({', '.join(map(str, self.dimension_sizes))})>"

    @classmethod
    def from_dict_given_keywords(
            cls,
            lists_keywords: list[list[str]],
            oridict: dict[Tuple[str, ...], float],
            default_initial_value: float = 0.0
    ) -> Self:
        """
        Create a new SparseArrayWrappedDict from a standard Python dictionary with given keywords.
        
        Parameters
        ----------
        lists_keywords : list[list[str]]
            A list of lists of strings, where each inner list contains the keys for one dimension of the array.
        oridict : dict[Tuple[str, ...], float]
            A standard Python dictionary with keys as tuples of strings and values as floats.
        default_initial_value : float, optional
            The default value to fill the array with for keys not present in oridict, by default 0.0.
            
        Returns
        -------
        SparseArrayWrappedDict
            A new SparseArrayWrappedDict with the same keys and values as oridict.
        """
        wrapped_dict = SparseArrayWrappedDict(
            lists_keywords,
            default_initial_value=default_initial_value
        )
        for keywords_tuple in product(*lists_keywords):
            wrapped_dict[keywords_tuple] = oridict.get(keywords_tuple, default_initial_value)
        return wrapped_dict

