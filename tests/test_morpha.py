import unittest

import numpy as np

from omicron.features import maths
from omicron.features.maths import polyfit
from omicron.features.morpha import cross, ncross, polyfit_inflextion, vcross


class MyTestCase(unittest.TestCase):
    async def test_polyfit(self):
        ts = np.arange(10)
        err, (a, b) = polyfit(ts, deg=1)

        self.assertAlmostEqual(0, err, places=9)
        self.assertEqual(1, a)

        ts = np.array([(x ** 2 - 8 * x + 16) for x in range(10)])
        err, (a, b, c), vert = polyfit(ts)

        self.assertAlmostEqual(0, err, places=9)
        self.assertAlmostEqual(1, a, places=9)
        self.assertAlmostEqual(-8, b, places=9)
        self.assertAlmostEqual(16, c, places=9)

        self.assertAlmostEqual(4, vert[0], places=9)

    def test_polyfit_inflextion(self):
        x = np.arange(100) / 10
        y = [np.sin(xi) for xi in x]
        # plt.plot(x, y)

        peaks, valleys = polyfit_inflextion(y, 10)

        self.assertListEqual([16, 79], peaks)
        self.assertListEqual([48], valleys)

    async def test_cross(self):
        x = np.arange(100) / 10
        y1 = np.array([np.sin(xi) for xi in x])
        y2 = np.array([(0.5 * xi - 0.5) for xi in x])

        flag, idx = cross(y2, y1)
        self.assertEqual(1, flag)
        self.assertEqual(23, idx)

        flag, idx = cross(y1, y2)
        self.assertEqual(-1, flag)

        y3 = 2 + x / 10
        flag, idx = cross(y3, y1)
        self.assertEqual(0, flag)
        self.assertEqual(0, idx)

        y4 = 0.5 + x / 100
        flag, idx = cross(y1, y4)
        self.assertEqual(-1, flag)
        self.assertEqual(87, idx)

    async def test_vcross(self):
        x = np.arange(100) / 10
        y1 = np.array([np.sin(xi) for xi in x])
        y5 = -0.75 + x / 100

        flag, indice = vcross(y1, y5)
        self.assertTrue(flag)
        self.assertListEqual([39, 55], indice)

    async def test_ncross(self):
        x = (np.arange(100) / 10)[60:95]
        y1 = np.array([np.sin(xi) for xi in x])
        y5 = 0.5 + x / 100

        flag, indice = ncross(y1, y5)
        self.assertTrue(flag)
        self.assertTupleEqual((8, 27), indice)


if __name__ == "__main__":
    unittest.main()
