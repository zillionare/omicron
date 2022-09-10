import unittest

import numpy as np

from omicron.talib import (
    BreakoutFlag,
    CrossFlag,
    angle,
    breakout,
    cross,
    inverse_vcross,
    peaks_and_valleys,
    polyfit,
    slope,
    support_resist_lines,
    vcross,
)


class MorphTest(unittest.TestCase):
    def test_cross(self):
        y1 = np.array([i + 5 for i in range(10)])
        y2 = np.array([0.3 * i**2 for i in range(10)])

        flag, index = cross(y1, y2)
        self.assertEqual(CrossFlag.DOWNCROSS, flag)
        self.assertEqual(6, index)

        flag, index = cross(y2, y1)
        self.assertEqual(CrossFlag.UPCROSS, flag)
        self.assertEqual(6, index)

        # y1 == y2 when index == 4
        y2 = np.array([0.5 * i**2 for i in range(10)])
        flag, index = cross(y1, y2)
        self.assertEqual(CrossFlag.DOWNCROSS, flag)
        self.assertEqual(4, index)

        flag, index = cross(y2, y1)
        self.assertEqual(CrossFlag.UPCROSS, flag)
        self.assertEqual(4, index)

        # no cross
        y2 = np.array([i + 3 for i in range(10)])
        flag, index = cross(y1, y2)
        self.assertEqual(CrossFlag.NONE, flag)

    def test_vcross(self):
        f = np.array([3 * i**2 - 20 * i + 2 for i in range(10)])
        g = np.array([i - 5 for i in range(10)])

        flag, indices = vcross(f, g)
        self.assertTrue(flag)
        self.assertTupleEqual((0, 6), indices)

    def test_polyfit(self):
        ts = [i for i in range(5)]

        err, (a, b) = polyfit(ts, deg=1)
        self.assertTrue(err < 1e-13)
        self.assertAlmostEquals(1, a)
        self.assertAlmostEquals(0, b)

        ts = np.array([0.2 * i**2 + 2 * i + 3 for i in range(5)])
        err, (a, b, c), (x, y) = polyfit(ts)
        self.assertTrue(err < 1e-13)
        self.assertAlmostEquals(0.2, a)
        self.assertAlmostEquals(2, b)
        self.assertAlmostEquals(3, c)

        ts[2] = np.NaN
        err, _, _ = polyfit(ts)
        self.assertTrue(err >= 1e9)

    def test_angle(self):
        ts = np.array([i for i in range(5)])
        err, angle_ = angle(ts)
        self.assertTrue(err < 0.01)

        self.assertAlmostEquals(0.707, angle_, places=3)  # degree: 45, rad: pi/2

        ts = np.array([np.sqrt(3) / 3 * i for i in range(10)])
        err, angle_ = angle(ts)
        self.assertTrue(err < 0.01)
        self.assertAlmostEquals(0.866, angle_, places=3)  # degree: 30, rad: pi/6

        ts = np.array([-np.sqrt(3) / 3 * i for i in range(7)])
        err, angle_ = angle(ts)
        self.assertTrue(err < 0.01)
        self.assertAlmostEquals(-0.866, angle_, places=3)  # degree: 150, rad: 5*pi/6

    def test_inverse_vcross(self):
        f = np.array([-3 * i**2 + 20 * i - 10 for i in range(10)])
        g = np.array([i - 5 for i in range(10)])

        flag, indices = inverse_vcross(f, g)
        self.assertTrue(flag)
        self.assertTupleEqual((0, 6), indices)

    def test_slope(self):
        ts = [i for i in range(5)]

        err, a = slope(ts)
        self.assertTrue(err < 1e-13)
        self.assertAlmostEquals(1, a, places=7)

    def test_peaks_and_valleys(self):
        np.random.seed(1997)
        X = np.cumprod(1 + np.random.randn(100) * 0.01)

        actual = peaks_and_valleys(X, 0.03, -0.03)
        exp = np.array(
            [
                -1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                -1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                -1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                -1,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                -1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
            ]
        )

        np.testing.assert_array_equal(exp, actual)

        actual = peaks_and_valleys(X.astype(np.float32), 0.03, -0.03)
        exp = np.array(
            [
                -1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                -1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                -1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                -1,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                -1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
            ]
        )

        np.testing.assert_array_equal(exp, actual)

    def test_support_resist_lines(self):
        np.random.seed(1978)
        X = np.cumprod(1 + np.random.randn(100) * 0.01)

        # test resist, support by predict next point
        support, resist, start = support_resist_lines(X)
        self.assertAlmostEqual(1.0021215790349554, resist(100))
        self.assertAlmostEqual(0.94672668920434, support(100))

    def test_breakout(self):
        np.random.seed(1978)
        X = np.cumprod(1 + np.random.randn(100) * 0.01)

        # test resist, support by predict next point
        self.assertEqual(BreakoutFlag.NONE, breakout(X))

        y = np.concatenate([X, [1.1]])
        self.assertEqual(BreakoutFlag.UP, breakout(y))

        y = np.concatenate([X, [0.8]])
        self.assertEqual(BreakoutFlag.DOWN, breakout(y))

        y = np.concatenate([X, [0.9, 0.8]])
        self.assertEqual(BreakoutFlag.NONE, breakout(y, 0.03, -0.03, confirm=2))

        y = np.concatenate([X, [0.8, 0.75]])
        self.assertEqual(BreakoutFlag.DOWN, breakout(y, 0.03, -0.03, confirm=2))
