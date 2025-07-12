import unittest
import numpy as np

from npdict import NumpyNDArrayWrappedDict
from npdict.utils import DuplicatedKeyError, WrongArrayDimensionException, WrongArrayShapeException


class TestNumpyNDArrayWrappedDict(unittest.TestCase):

    def setUp(self):
        self.lists_keystrings = [
            ['a', 'b', 'c'],
            ['d', 'e']
        ]
        self.wrapped_dict = NumpyNDArrayWrappedDict(self.lists_keystrings, default_initial_value=1.0)

    def test_initialization(self):
        self.assertEqual(self.wrapped_dict.tensor_dimensions, 2)
        self.assertEqual(self.wrapped_dict.dimension_sizes, [3, 2])
        self.assertEqual(len(self.wrapped_dict), 6)
        np.testing.assert_array_equal(self.wrapped_dict.to_numpy(), np.ones((3, 2)))

    def test_initialization_with_duplicated_keys(self):
        with self.assertRaises(DuplicatedKeyError):
            NumpyNDArrayWrappedDict([['a', 'a'], ['b', 'c']])

    def test_getitem(self):
        self.wrapped_dict[('a', 'd')] = 2.0
        self.assertEqual(self.wrapped_dict[('a', 'd')], 2.0)
        self.wrapped_dict[('a', 'e')] = 1.0   # test empty item

    def test_getitem_wrong_dimension(self):
        with self.assertRaises(WrongArrayDimensionException):
            _ = self.wrapped_dict[('a',)]

    def test_setitem(self):
        self.wrapped_dict[('b', 'e')] = 3.0
        self.assertEqual(self.wrapped_dict[('b', 'e')], 3.0)
        self.assertEqual(self.wrapped_dict.to_numpy()[1, 1], 3.0)

    def test_setitem_wrong_dimension(self):
        with self.assertRaises(WrongArrayDimensionException):
            self.wrapped_dict[('a', 'b', 'c')] = 1.0

    def test_iteration(self):
        keys = list(self.wrapped_dict)
        self.assertEqual(len(keys), 6)
        self.assertIn(('a', 'd'), keys)
        self.assertIn(('c', 'e'), keys)

    def test_keys(self):
        keys = self.wrapped_dict.keys()
        self.assertEqual(len(keys), 6)
        self.assertIn(('a', 'd'), keys)

    def test_values(self):
        self.wrapped_dict[('a', 'd')] = 5.0
        self.wrapped_dict[('c', 'e')] = 6.0
        values = self.wrapped_dict.values()
        self.assertIn(1.0, values)
        self.assertIn(5.0, values)
        self.assertIn(6.0, values)

    def test_items(self):
        self.wrapped_dict[('a', 'd')] = 7.0
        items = self.wrapped_dict.items()
        self.assertIn((('a', 'd'), 7.0), items)
        self.assertIn((('b', 'd'), 1.0), items)

    def test_to_dict(self):
        self.wrapped_dict[('a', 'd')] = 8.0
        d = self.wrapped_dict.to_dict()
        self.assertEqual(d[('a', 'd')], 8.0)
        self.assertEqual(d[('b', 'e')], 1.0)

    def test_from_dict(self):
        d = {('a', 'x'): 1, ('b', 'y'): 2}
        wrapped = NumpyNDArrayWrappedDict.from_dict(d)
        self.assertEqual(wrapped[('a', 'x')], 1)
        self.assertEqual(wrapped[('b', 'y')], 2)
        self.assertEqual(wrapped[('a', 'y')], 0.0)
        self.assertEqual(wrapped[('b', 'x')], 0.0)

    def test_from_dict_given_keywords(self):
        d = {('a', 'x'): 1}
        keywords = [['a', 'b'], ['x', 'y']]
        wrapped = NumpyNDArrayWrappedDict.from_dict_given_keywords(keywords, d, default_initial_value=-1)
        self.assertEqual(wrapped[('a', 'x')], 1)
        self.assertEqual(wrapped[('b', 'y')], -1)

    def test_repr_str(self):
        self.assertEqual(repr(self.wrapped_dict), f"<NumpyNDArrayWrappedDict: dimensions ({', '.join(map(str, self.wrapped_dict.dimension_sizes))})>")
        self.assertEqual(str(self.wrapped_dict), repr(self.wrapped_dict))

    def test_update(self):
        with self.assertRaises(TypeError):
            self.wrapped_dict.update({('a', 'd'): 1})

    def test_generate_dict(self):
        new_array = np.zeros((3, 2))
        new_wrapped_dict = self.wrapped_dict.generate_dict(new_array)
        np.testing.assert_array_equal(new_wrapped_dict.to_numpy(), new_array)
        self.assertEqual(new_wrapped_dict.dimension_sizes, self.wrapped_dict.dimension_sizes)

    def test_generate_dict_wrong_shape(self):
        new_array = np.zeros((2, 3))
        with self.assertRaises(WrongArrayShapeException):
            self.wrapped_dict.generate_dict(new_array)


if __name__ == '__main__':
    unittest.main()
