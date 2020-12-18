import unittest

import numpy as np

from omicron.features.maths import (
    NormMethod,
    momentum,
    moving_average,
    norm,
    resample,
    sigmoid,
    slope,
    slope2degree,
)


class TestMaths(unittest.TestCase):
    def test_slope(self):
        c = np.array([1.1 ** x for x in range(10)]) * 5
        ma = moving_average(c, 5)

        slp = slope(ma, with_norm=NormMethod.l2)[0]
        self.assertAlmostEqual(0.03819, slp, places=5)

        slp = slope(ma, with_norm=None)[0]
        self.assertAlmostEqual(0.744, slp, places=3)

        slp = slope(ma, with_norm=NormMethod.start_scale)[0]
        self.assertAlmostEqual(0.1218, slp, places=3)

    def test_slope2degree(self):
        c = np.array([1.1 ** x for x in range(10)]) * 5
        ma = moving_average(c, 5)

        slp = slope(ma)[0]
        degree = slope2degree(slp * 10)
        self.assertAlmostEqual(50.63, degree, places=2)

    def test_moving_average(self):
        ts = np.arange(5)
        ma = moving_average(ts, 5)
        self.assertEqual(1, len(ma))
        self.assertAlmostEqual(2, ma[0], places=9)

    def test_norm(self):
        ts = np.arange(1, 5)

        expected = [2, 0.5, 0.3333, -0.4472, 0.2, 0.3651, 0.5]
        for i, method in enumerate(
            [
                "start_scale",
                "end_scale",
                "minmax_scale",
                "zscore",
                "l1",
                "l2",
                "max_abs_scale",
            ]
        ):
            normed = norm(ts, NormMethod[method])
            self.assertAlmostEqual(expected[i], normed[1], places=4)

    def test_momentum(self):
        ts = np.arange(5)

        mom = momentum(ts, with_norm=NormMethod.max_abs_scale)
        self.assertEqual(4, len(mom))
        self.assertEqual(0.25, mom[0])

        ts = np.array([(0.5 * x ** 2 + 2 * x + 3) for x in range(5)])
        mom = momentum(ts, deg=2, with_norm=None)

        # mom should be 2 * a
        self.assertEqual(1, mom[0])

    def test_resample(self):
        arr = np.arange(9)
        result = resample(arr, 3, np.sum)
        print(result)
        self.assertListEqual([3, 12, 21], result.tolist())

    def test_sigmoid(self):
        self.assertAlmostEqual(0.6224593312018546, sigmoid(0.5), places=7)
        self.assertAlmostEqual(0.3775406687981454, sigmoid(-0.5), places=7)


if __name__ == "__main__":
    unittest.main()
