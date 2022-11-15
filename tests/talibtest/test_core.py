import unittest

import numpy as np

from omicron.talib.core import (
    clustering,
    exp_moving_average,
    mean_absolute_error,
    moving_average,
    normalize,
    pct_error,
    weighted_moving_average,
)


class TaLibCoreTest(unittest.TestCase):
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

    def test_moving_average(self):
        ts = [i for i in range(5)]

        # without padding
        ma = moving_average(ts, 3, padding=False)
        self.assertEqual(3, len(ma))
        self.assertListEqual([1.0, 2.0, 3.0], ma.tolist())

        ma = moving_average(ts, 3)
        self.assertEqual(5, len(ma))
        self.assertListEqual([1.0, 2.0, 3.0], ma.tolist()[2:])
        self.assertTrue(np.isnan(ma[0]))

        ts[4] = np.nan
        ma = moving_average(ts, 3)
        self.assertEqual(5, len(ma))
        np.testing.assert_array_equal([np.nan, np.nan, 1.0, 2.0, np.nan], ma)

    def test_weighted_moving_average(self):
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        wma = weighted_moving_average(data, 5)
        print(wma)

    def test_exponential_moving_average(self):
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        ema = exp_moving_average(data, 5)
        print(ema)

    def test_mean_absolute_error(self):
        y = np.array([i for i in range(5)])
        y_hat = y.copy()
        y_hat[4] = 0

        self.assertEqual(0, mean_absolute_error(y, y))
        self.assertAlmostEquals(0.8, mean_absolute_error(y, y_hat))
        self.assertAlmostEquals(0.8, mean_absolute_error(y_hat, y))

    def test_relative_error(self):
        y = np.arange(5)
        y_hat = y.copy()
        y_hat[4] = 0

        print(pct_error(y, y_hat))

    def test_clustering(self):
        numbers = np.array([1, 1, 1, 2, 4, 6, 8, 7, 4, 5, 6])
        actual = clustering(numbers, 2)
        self.assertListEqual(actual, [(0, 4), (4, 7)])
