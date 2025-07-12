import unittest
import numpy as np
import sparse

from npdict import SparseArrayWrappedDict
from npdict.utils import DuplicatedKeyError, WrongArrayDimensionException, WrongArrayShapeException


class TestSparseArrayWrappedDict(unittest.TestCase):

    def setUp(self):
        self.lists_keystrings = [
            ['a', 'b', 'c'],
            ['d', 'e']
        ]
        self.wrapped_dict = SparseArrayWrappedDict(self.lists_keystrings, default_initial_value=1.0)

    def test_initialization(self):
        self.assertEqual(self.wrapped_dict.tensor_dimensions, 2)
        self.assertEqual(self.wrapped_dict.dimension_sizes, [3, 2])
        self.assertEqual(len(self.wrapped_dict), 6)
        np.testing.assert_array_equal(self.wrapped_dict.to_numpy(), np.ones((3, 2)))

    def test_initialization_with_duplicated_keys(self):
        with self.assertRaises(DuplicatedKeyError):
            SparseArrayWrappedDict([['a', 'a'], ['b', 'c']])

    def test_getitem_setitem(self):
        self.wrapped_dict[('a', 'd')] = 2.0
        self.assertEqual(self.wrapped_dict[('a', 'd')], 2.0)
        self.assertEqual(self.wrapped_dict[('b', 'e')], 1.0)

    def test_getitem_wrong_dimension(self):
        with self.assertRaises(WrongArrayDimensionException):
            _ = self.wrapped_dict[('a',)]

    def test_setitem_wrong_dimension(self):
        with self.assertRaises(WrongArrayDimensionException):
            self.wrapped_dict[('a', 'b', 'c')] = 1.0

    def test_to_numpy(self):
        self.wrapped_dict[('a', 'd')] = 3.0
        expected_array = np.array([[3., 1.], [1., 1.], [1., 1.]])
        np.testing.assert_array_equal(self.wrapped_dict.to_numpy(), expected_array)

    def test_to_coo(self):
        self.wrapped_dict[('b', 'e')] = 4.0
        coo = self.wrapped_dict.to_coo()
        self.assertIsInstance(coo, sparse.COO)
        self.assertEqual(coo[1, 1], 4.0)
        self.assertEqual(coo[0, 1], 1.0)

    def test_to_dok(self):
        self.wrapped_dict[('c', 'd')] = 5.0
        dok = self.wrapped_dict.to_dok()
        self.assertIsInstance(dok, sparse.DOK)
        self.assertEqual(dok[2, 0], 5.0)
        self.assertEqual(dok[1, 1], 1.0)

    def test_generate_dict(self):
        new_sparse_array = sparse.random((3, 2), density=0.5)
        
        # Test generating a new sparse dict
        new_sparse_dict = self.wrapped_dict.generate_dict(new_sparse_array, dense=False)
        self.assertIsInstance(new_sparse_dict, SparseArrayWrappedDict)
        np.testing.assert_array_equal(new_sparse_dict.to_numpy(), new_sparse_array.todense())

        # Test generating a new dense dict
        new_dense_dict = self.wrapped_dict.generate_dict(new_sparse_array, dense=True)
        from npdict import NumpyNDArrayWrappedDict
        self.assertIsInstance(new_dense_dict, NumpyNDArrayWrappedDict)
        np.testing.assert_array_equal(new_dense_dict.to_numpy(), new_sparse_array.todense())

    def test_generate_dict_wrong_shape(self):
        new_array = sparse.DOK((2, 3))
        with self.assertRaises(WrongArrayShapeException):
            self.wrapped_dict.generate_dict(new_array)

    def test_repr(self):
        self.assertEqual(repr(self.wrapped_dict), f"<SparseArrayWrappedDict: dimensions ({', '.join(map(str, self.wrapped_dict.dimension_sizes))})>")

    def test_from_dict_given_keywords(self):
        d = {('a', 'd'): 10.0, ('c', 'e'): 20.0}
        keywords = [['a', 'b', 'c'], ['d', 'e']]
        wrapped = SparseArrayWrappedDict.from_dict_given_keywords(keywords, d, default_initial_value=-1.0)
        self.assertEqual(wrapped[('a', 'd')], 10.0)
        self.assertEqual(wrapped[('c', 'e')], 20.0)
        self.assertEqual(wrapped[('b', 'd')], -1.0)


if __name__ == '__main__':
    unittest.main()