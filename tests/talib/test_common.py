import unittest

from omicron.talib import *


class CommonTest(unittest.TestCase):
    def test_find_runs(self):
        a = [
            1,
            1,
            2,
            2,
            3,
            3,
            3,
        ]
        value, pos, length = find_runs(a)
        self.assertListEqual([1, 2, 3], value.tolist())
        self.assertListEqual([0, 2, 4], pos.tolist())
        self.assertListEqual([2, 2, 3], length.tolist())

    def test_barssince(self):
        condition = [False, True]
        self.assertEqual(0, bars_since(condition))

        condition = [True, False]
        self.assertEqual(1, bars_since(condition))

        condition = [True, True, False]
        self.assertEqual(1, bars_since(condition))

        condition = [True, True, False, True]
        self.assertEqual(0, bars_since(condition))

        condition = [True, True, False, False]
        self.assertEqual(2, bars_since(condition))

    def test_normalize(self):
        # unit_vector
        X = [[1.0, -1.0, 2.0], [2.0, 0.0, 0.0], [0.0, 1.0, -1.0]]

        expected = [[0.4082, -0.4082, 0.8165], [1.0, 0.0, 0.0], [0.0, 0.7071, -0.7071]]

        X_hat = normalize(X, scaler="unit_vector")
        np.testing.assert_array_almost_equal(expected, X_hat, decimal=4)

        # max_abs
        X = np.array([[1.0, -1.0, 2.0], [2.0, 0.0, 0.0], [0.0, 1.0, -1.0]])

        expected = [[0.5, -1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, -0.5]]

        X_hat = normalize(X, scaler="maxabs")
        np.testing.assert_array_almost_equal(expected, X_hat, decimal=2)

        # min_max
        expected = [[0.5, 0.0, 1.0], [1.0, 0.5, 0.33333333], [0.0, 1.0, 0.0]]
        X_hat = normalize(X, scaler="minmax")
        np.testing.assert_array_almost_equal(expected, X_hat, decimal=3)

        # standard
        X = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = [[-1.0, -1.0], [-1.0, -1.0], [1.0, 1.0], [1.0, 1.0]]
        X_hat = normalize(X, scaler="standard")
        np.testing.assert_array_almost_equal(expected, X_hat, decimal=3)

    def test_top_n_argpos(self):
        arr = [4, 3, 9, 8, 5, 2, 1, 0, 6, 7]
        actual = top_n_argpos(arr, 2)
        exp = [2, 3]
        self.assertListEqual(exp, actual.tolist())
