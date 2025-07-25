
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
    """
    A dictionary-like class that wraps a NumPy n-dimensional array.
    
    This class provides a dictionary interface to a NumPy array, where the keys are tuples of strings
    and the values are the corresponding elements in the array. The class maintains a mapping between
    string keys and array indices, allowing for more intuitive access to array elements.
    """
    def __init__(
            self,
            lists_keystrings: list[list[str]],
            default_initial_value: float=0.0
    ):
        """
        Initialize a new NumpyNDArrayWrappedDict.
        
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

        self._numpyarray = np.empty(tuple(len(l) for l in self._lists_keystrings))
        self._numpyarray.fill(default_initial_value)

    def _get_indices(self, item: Tuple[str, ...]) -> list[int]:
        """
        Convert a tuple of string keys to a list of integer indices.
        
        Parameters
        ----------
        item : Tuple[str, ...]
            A tuple of string keys, one for each dimension of the array.
            
        Returns
        -------
        list[int]
            A list of integer indices corresponding to the string keys.
        """
        return [
            mapping[keyword]
            for mapping, keyword in zip(self._keystrings_to_indices, item)
        ]

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
        return self._numpyarray[tuple(indices)]

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
        self._numpyarray[tuple(indices)] = value

    def update(self, new_dict: dict):
        """
        This method is not supported for NumpyNDArrayWrappedDict.
        
        Raises
        ------
        TypeError
            Always raises this exception as the method is not supported.
        """
        raise TypeError("We cannot update this kind of dict this way!")

    def __iter__(self) -> Generator[Tuple[str, ...], None, None]:
        """
        Iterate over all possible key tuples in the dictionary.
        
        Yields
        ------
        Tuple[str, ...]
            A tuple of string keys, one for each dimension of the array.
        """
        for keywords_tuple in product(*self._lists_keystrings):
            yield keywords_tuple

    def keys(self):
        """
        Get all possible key tuples in the dictionary.
        
        Returns
        -------
        list[Tuple[str, ...]]
            A list of all possible key tuples.
        """
        return list(self.__iter__())

    def values(self):
        """
        Get all values in the dictionary.
        
        Returns
        -------
        list[float]
            A list of all values in the dictionary.
        """
        return [self.__getitem__(keywords_tuple) for keywords_tuple in self.__iter__()]

    def items(self):
        """
        Get all key-value pairs in the dictionary.
        
        Returns
        -------
        list[Tuple[Tuple[str, ...], float]]
            A list of all key-value pairs in the dictionary.
        """
        return [
            (keywords_tuple, self.__getitem__(keywords_tuple))
            for keywords_tuple in self.__iter__()
        ]

    def to_numpy(self) -> np.ndarray:
        """
        Convert the wrapped dictionary to a NumPy array.
        
        Returns
        -------
        np.ndarray
            The NumPy array containing the values of the dictionary.
        """
        return self._numpyarray

    def generate_dict(self, nparray: np.ndarray) -> Self:
        """
        Generate a new NumpyNDArrayWrappedDict with the same keys but different values.
        
        Parameters
        ----------
        nparray : np.ndarray
            The NumPy array containing the new values.
            
        Returns
        -------
        NumpyNDArrayWrappedDict
            A new NumpyNDArrayWrappedDict with the same keys but different values.
            
        Raises
        ------
        WrongArrayDimensionException
            If the number of dimensions in the array does not match the number of dimensions in the dictionary.
        WrongArrayShapeException
            If the shape of the array does not match the shape of the dictionary.
        """
        if len(nparray.shape) != self.tensor_dimensions:
            raise WrongArrayDimensionException(self.tensor_dimensions, len(nparray.shape))
        if nparray.shape != self._numpyarray.shape:
            raise WrongArrayShapeException(self._numpyarray.shape, nparray.shape)
        wrapped_dict = NumpyNDArrayWrappedDict(self._lists_keystrings)
        wrapped_dict._numpyarray = nparray
        return wrapped_dict

    def __repr__(self) -> str:
        """
        Return a string representation of the dictionary.
        
        Returns
        -------
        str
            A string representation of the dictionary.
        """
        return f"<NumpyNDArrayWrappedDict: dimensions ({', '.join(map(str, self.dimension_sizes))})>"

    def __str__(self) -> str:
        """
        Return a string representation of the dictionary.
        
        Returns
        -------
        str
            A string representation of the dictionary.
        """
        return self.__repr__()

    def __len__(self) -> int:
        """
        Return the total number of elements in the dictionary.
        
        Returns
        -------
        int
            The total number of elements in the dictionary.
        """
        return self._total_size

    def to_dict(self) -> dict[Tuple[str, ...], float]:
        """
        Convert the wrapped dictionary to a standard Python dictionary.
        
        Returns
        -------
        dict[Tuple[str, ...], float]
            A standard Python dictionary with the same keys and values as the wrapped dictionary.
        """
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
        """
        Create a new NumpyNDArrayWrappedDict from a standard Python dictionary with given keywords.
        
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
        NumpyNDArrayWrappedDict
            A new NumpyNDArrayWrappedDict with the same keys and values as oridict.
        """
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
        """
        Create a new NumpyNDArrayWrappedDict from a standard Python dictionary.
        
        This method automatically extracts the keys for each dimension from the dictionary.
        
        Parameters
        ----------
        oridict : dict[Tuple[str, ...], float]
            A standard Python dictionary with keys as tuples of strings and values as floats.
        default_initial_value : float, optional
            The default value to fill the array with for keys not present in oridict, by default 0.0.
            
        Returns
        -------
        NumpyNDArrayWrappedDict
            A new NumpyNDArrayWrappedDict with the same keys and values as oridict.
        """
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
        """
        Get the number of dimensions in the array.
        
        Returns
        -------
        int
            The number of dimensions in the array.
        """
        return self._tensor_dimensions

    @property
    def dimension_sizes(self) -> list[int]:
        """
        Get the size of each dimension in the array.
        
        Returns
        -------
        list[int]
            A list of the size of each dimension in the array.
        """
        return self._dimension_sizes
