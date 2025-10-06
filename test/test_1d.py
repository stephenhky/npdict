
import unittest

from npdict import NumpyNDArrayWrappedDict, SparseArrayWrappedDict


class Test1DCases(unittest.TestCase):
    def test_wrapped_dict(self):
        array = NumpyNDArrayWrappedDict([['a', 'b', 'c', 'd']], default_initial_value=0.0)
        array['a'] = 1.0
        array[('b',)] = 2.0
        array['c'] = 3.0
        array[('d',)] = 4.0

        self.assertAlmostEqual(array[('a',)], 1.0)
        self.assertAlmostEqual(array['b'], 2.0)
        self.assertAlmostEqual(array[('c',)], 3.0)
        self.assertAlmostEqual(array['d'], 4.0)

    def test_sparse_dict(self):
        array = SparseArrayWrappedDict([['a', 'b', 'c', 'd']], default_initial_value=100.0)
        array['a'] = 1.0
        array[('d',)] = 2.5

        self.assertAlmostEqual(array[('a',)], 1.0)
        self.assertAlmostEqual(array['b'], 100.0)
        self.assertAlmostEqual(array[('c',)], 100.0)
        self.assertAlmostEqual(array['d'], 2.5)


if __name__ == '__main__':
    unittest.main()
