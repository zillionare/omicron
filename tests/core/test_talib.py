import math
import unittest
from copy import copy

import numpy as np

from omicron.core import talib as ta


class LibTest(unittest.TestCase):
    def test_barssince(self):
        condition = [False, True]
        self.assertEqual(0, ta.barssince(condition))

        condition = [True, False]
        self.assertEqual(1, ta.barssince(condition))

        condition = [True, True, False]
        self.assertEqual(1, ta.barssince(condition))

        condition = [True, True, False, True]
        self.assertEqual(0, ta.barssince(condition))

        condition = [True, True, False, False]
        self.assertEqual(2, ta.barssince(condition))

    def test_cross(self):
        y1 = np.array([i + 5 for i in range(10)])
        y2 = np.array([0.3 * i ** 2 for i in range(10)])

        flag, index = ta.cross(y1, y2)
        self.assertEqual(-1, flag)
        self.assertEqual(6, index)

        flag, index = ta.cross(y2, y1)
        self.assertEqual(1, flag)
        self.assertEqual(6, index)

        # y1 == y2 when index == 4
        y2 = np.array([0.5 * i ** 2 for i in range(10)])
        flag, index = ta.cross(y1, y2)
        self.assertEqual(-1, flag)
        self.assertEqual(4, index)

        flag, index = ta.cross(y2, y1)
        self.assertEqual(1, flag)
        self.assertEqual(4, index)

        # no cross
        y2 = np.array([i + 3 for i in range(10)])
        flag, index = ta.cross(y1, y2)
        self.assertEqual(0, flag)

    def test_vcross(self):
        f = np.array([3 * i ** 2 - 20 * i + 2 for i in range(10)])
        g = np.array([i - 5 for i in range(10)])

        flag, indices = ta.vcross(f, g)
        self.assertTrue(flag)
        self.assertTupleEqual((0, 6), indices)

    def test_moving_average(self):
        ts = [i for i in range(5)]
        ma = ta.moving_average(ts, 3)
        self.assertEqual(3, len(ma))
        self.assertListEqual([1., 2., 3.], ma.tolist())
        self.assertFalse(math.isnan(ma[0]))

    def test_mae(self):
        y = np.array([i for i in range(5)])
        y_hat = copy(y)
        y_hat[4] = 0

        self.assertEqual(0, ta.mean_absolute_error(y, y))
        self.assertAlmostEquals(0.8, ta.mean_absolute_error(y, y_hat))
        self.assertAlmostEquals(0.8, ta.mean_absolute_error(y_hat, y))

    def test_relative_error(self):
        y = np.arange(5)
        y_hat = copy(y)
        y_hat[4] = 0

        print(ta.relative_error(y, y_hat))

    def test_normalize(self):
        # unit_vector
        X = [[1.0, -1.0, 2.0], [2.0, 0.0, 0.0], [0.0, 1.0, -1.0]]

        expected = [[0.4082, -0.4082, 0.8165], [1.0, 0.0, 0.0], [0.0, 0.7071, -0.7071]]

        X_hat = ta.normalize(X, scaler="unit_vector")
        np.testing.assert_array_almost_equal(expected, X_hat, decimal=4)

        # max_abs
        X = np.array([[1.0, -1.0, 2.0], [2.0, 0.0, 0.0], [0.0, 1.0, -1.0]])

        expected = [[0.5, -1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, -0.5]]

        X_hat = ta.normalize(X, scaler="maxabs")
        np.testing.assert_array_almost_equal(expected, X_hat, decimal=2)

        # min_max
        expected = [[0.5, 0.0, 1.0], [1.0, 0.5, 0.33333333], [0.0, 1.0, 0.0]]
        X_hat = ta.normalize(X, scaler="minmax")
        np.testing.assert_array_almost_equal(expected, X_hat, decimal=3)

        # standard
        X = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = [[-1.0, -1.0], [-1.0, -1.0], [1.0, 1.0], [1.0, 1.0]]
        X_hat = ta.normalize(X, scaler="standard")
        np.testing.assert_array_almost_equal(expected, X_hat, decimal=3)

    def test_polyfit(self):
        ts = [i for i in range(5)]

        err, (a, b) = ta.polyfit(ts, deg=1)
        self.assertTrue(err < 1e-13)
        self.assertAlmostEquals(1, a)
        self.assertAlmostEqual(0, b, 7)

        ts = np.array([0.2 * i ** 2 + 2 * i + 3 for i in range(5)])
        err, (a, b, c), (x, y) = ta.polyfit(ts)
        self.assertTrue(err < 1e-13)
        self.assertAlmostEquals(0.2, a)
        self.assertAlmostEquals(2, b)
        self.assertAlmostEquals(3, c)

        ts[2] = np.NaN
        err, _, _ = ta.polyfit(ts)
        self.assertTrue(err >= 1e9)

    def test_angle(self):
        ts = np.array([i for i in range(5)])
        err, angle = ta.angle(ts)
        self.assertTrue(err < 0.01)

        self.assertAlmostEquals(0.707, angle, places=3)  # degree: 45, rad: pi/2

        ts = np.array([np.sqrt(3) / 3 * i for i in range(10)])
        err, angle = ta.angle(ts)
        self.assertTrue(err < 0.01)
        self.assertAlmostEquals(0.866, angle, places=3)  # degree: 30, rad: pi/6

        ts = np.array([-np.sqrt(3) / 3 * i for i in range(7)])
        err, angle = ta.angle(ts)
        self.assertTrue(err < 0.01)
        self.assertAlmostEquals(-0.866, angle, places=3)  # degree: 150, rad: 5*pi/6

    def test_inverse_vcross(self):
        f = np.array([-3 * i ** 2 + 20 * i - 10 for i in range(10)])
        g = np.array([i - 5 for i in range(10)])

        flag, indices = ta.inverse_vcross(f, g)
        self.assertTrue(flag)
        self.assertTupleEqual((0, 6), indices)

    def test_slope(self):
        ts = [i for i in range(5)]

        err, a = ta.slope(ts)
        self.assertTrue(err < 1e-13)
        self.assertAlmostEquals(1, a, places=7)

    def test_max_drawdown(self):
        ts = np.sin(np.arange(10) * np.pi / 10)
        dd, start, end = ta.max_drawdown(ts)

        self.assertAlmostEquals(-0.691, dd, places=3)
        self.assertListEqual([5, 9], [start, end])
