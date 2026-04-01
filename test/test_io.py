
import numpy as np
import sparse
from npdict import NumpyNDArrayWrappedDict, SparseArrayWrappedDict

import pytest


def test_npdict_io():
    lists_keystrings = [
        ['a', 'b', 'c'],
        ['d', 'e']
    ]
    array = np.array([[2., 3.], [-1.5, 2.5], [-3., 2.1]])

    npdict1 = NumpyNDArrayWrappedDict.from_numpyarray_given_keywords(
        lists_keystrings,
        array
    )
    npdict1.save("temp1.npy")

    npdict2 = NumpyNDArrayWrappedDict.load("temp1.npy")
    for item in npdict2.keys():
        assert npdict2[item] == pytest.approx(npdict1[item])

    assert npdict1.dimension_sizes == npdict2.dimension_sizes

    for list_of_strings1, list_of_strings2 in zip(npdict1._lists_keystrings, npdict2._lists_keystrings):
        for str1, str2 in zip(list_of_strings1, list_of_strings2):
            assert str1 == str2


def test_sparse_npdict_io():
    lists_keystrings = [
        ['a', 'b', 'c'],
        ['d', 'e']
    ]
    array = sparse.COO([[0, 1, 2], [1, 0, 1]], [-1.2, 3.4, 0.1], shape=(3, 2))

    npdict1 = SparseArrayWrappedDict.from_sparsearray_given_keywords(
        lists_keystrings,
        array
    )
    npdict1.save("temp2.npy")

    npdict2 = SparseArrayWrappedDict.load("temp2.npy")
    for item in npdict2.keys():
        assert npdict2[item] == pytest.approx(npdict1[item])

    assert npdict1.dimension_sizes == npdict2.dimension_sizes

    for list_of_strings1, list_of_strings2 in zip(npdict1._lists_keystrings, npdict2._lists_keystrings):
        for str1, str2 in zip(list_of_strings1, list_of_strings2):
            assert str1 == str2
