import math
import unittest
from copy import copy

import numpy as np

from omicron import talib as ta


class TaLibTest(unittest.TestCase):
    def test_cross(self):
        y1 = np.array([i + 5 for i in range(10)])
        y2 = np.array([0.3 * i ** 2 for i in range(10)])

        flag, index = ta.cross(y1, y2)
        self.assertEqual(ta.CrossFlag.DOWNCROSS, flag)
        self.assertEqual(6, index)

        flag, index = ta.cross(y2, y1)
        self.assertEqual(ta.CrossFlag.UPCROSS, flag)
        self.assertEqual(6, index)

        # y1 == y2 when index == 4
        y2 = np.array([0.5 * i ** 2 for i in range(10)])
        flag, index = ta.cross(y1, y2)
        self.assertEqual(ta.CrossFlag.DOWNCROSS, flag)
        self.assertEqual(4, index)

        flag, index = ta.cross(y2, y1)
        self.assertEqual(ta.CrossFlag.UPCROSS, flag)
        self.assertEqual(4, index)

        # no cross
        y2 = np.array([i + 3 for i in range(10)])
        flag, index = ta.cross(y1, y2)
        self.assertEqual(ta.CrossFlag.NONE, flag)

    def test_vcross(self):
        f = np.array([3 * i ** 2 - 20 * i + 2 for i in range(10)])
        g = np.array([i - 5 for i in range(10)])

        flag, indices = ta.vcross(f, g)
        self.assertTrue(flag)
        self.assertTupleEqual((0, 6), indices)

    def test_polyfit(self):
        ts = [i for i in range(5)]

        err, (a, b) = ta.polyfit(ts, deg=1)
        self.assertTrue(err < 1e-13)
        self.assertAlmostEquals(1, a)
        self.assertAlmostEqual(0, b)

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
