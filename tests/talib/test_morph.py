import unittest

import numpy as np

from omicron.talib import (
    BreakoutFlag,
    CrossFlag,
    angle,
    breakout,
    cross,
    energy_hump,
    inverse_vcross,
    peaks_and_valleys,
    plateaus,
    polyfit,
    rsi_bottom_dev_detect,
    rsi_predict_price,
    rsi_watermarks,
    slope,
    support_resist_lines,
    valley_detect,
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
        self.assertAlmostEqual(1.0021215790349554, resist(100), places=2)
        self.assertAlmostEqual(0.927, support(100), places=2)

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
        self.assertEqual(BreakoutFlag.NONE, breakout(y, 0.03, -0.03, confirm=2))

    def test_rsi_bottom_dev_detect(self):

        # test rsi bottom deviation type and disance by zigzag
        X = [
            7.340107,
            7.011891,
            6.9919987,
            7.688215,
            7.8473496,
            7.8871336,
            7.9766474,
            7.946809,
            8.105944,
            8.314809,
            8.354592,
            7.5589175,
            7.1710258,
            7.1411877,
            7.210809,
            7.2406473,
            7.260539,
            7.1909175,
            7.310269,
            7.1610794,
            7.2307014,
            7.69816,
            7.767782,
            7.8771877,
            7.7478905,
            7.5788093,
            7.6583767,
            8.06616,
            7.837404,
            7.8970795,
            7.6782684,
            7.588755,
            8.255134,
            8.056215,
            7.946809,
            7.8473496,
            7.827458,
            8.006485,
            7.7081065,
            7.9667015,
            8.195457,
            8.672863,
            8.155674,
            8.48389,
            8.1158905,
            8.095999,
            7.8871336,
            7.8970795,
            7.827458,
            7.399782,
            7.5091877,
            7.42962,
            7.5191336,
            7.52908,
            7.588755,
            7.5191336,
            7.5489717,
            7.489296,
            7.3599987,
            7.4793496,
            7.3898363,
            7.3599987,
            7.21,
            7.27,
            7.2,
            7.26,
            7.09,
            6.92,
            6.96,
            7.01,
            7.03,
            7.1,
            7.1,
            7.02,
            7.08,
            7.01,
            6.92,
            7.05,
            6.9,
            6.83,
            6.96,
            6.97,
            7.15,
            7.11,
            7.29,
            7.21,
            7.22,
            7.06,
            7.08,
            7.12,
            7.25,
            7.09,
            7.24,
            7.31,
            7.32,
            7.72,
            7.38,
            6.71,
            7.38,
            7.54,
            7.48,
            7.19,
            7.26,
            7.33,
            7.56,
            7.54,
            7.6,
            7.77,
            7.98,
            8.46,
            8.55,
            8.69,
            8.55,
            8.32,
            8.33,
            8.9,
            9.79,
            10.07,
            11.08,
            11.3,
            11.13,
            11.96,
            11.26,
            12.39,
            12.69,
            12.44,
            12.5,
            13.75,
            15.13,
            14.35,
            14.86,
            13.71,
            13.52,
            13.29,
            13.15,
            13.27,
            13.33,
            12.7,
            12.02,
            11.77,
            10.59,
            10.89,
            10.73,
            10.4,
            10.18,
            10.1,
            10.18,
            10.23,
            10.36,
            10.8,
            11.88,
            11.61,
            12.77,
            13.37,
            13.63,
            13.18,
            12.64,
            13.06,
            13.59,
            13.24,
            13.56,
            13.03,
            12.4,
            12.51,
            12.46,
            12.28,
            13.26,
            14.59,
            14.5,
            14.03,
            15.43,
            15.01,
            14.8,
            14.83,
            14.54,
            14.19,
            14.02,
            14.65,
            14.38,
            15.26,
            14.58,
            14.11,
            14.09,
            13.26,
            12.88,
            13.4,
            13.2,
            13.26,
            13.38,
            13.02,
            12.9,
            12.94,
            12.64,
            12.24,
            12.38,
            12.63,
            12.74,
            12.0,
            11.99,
            12.1,
            12.48,
            12.43,
            12.32,
            12.52,
            12.21,
            12.24,
            12.09,
            11.32,
            11.46,
            12.61,
            12.02,
            12.29,
            12.67,
            12.9,
            12.78,
            12.58,
            12.26,
            12.79,
            12.58,
            12.6,
            12.15,
            11.95,
            12.4,
            12.56,
            12.42,
            12.75,
            12.38,
            11.38,
            11.57,
            11.5,
            11.68,
            11.52,
            11.39,
            11.23,
            11.38,
            11.28,
            11.29,
            11.45,
            11.78,
            12.26,
        ]
        X = np.array(X)
        bottom_dev_type, bottom_dev_distance = rsi_bottom_dev_detect(
            X, (0.04, -0.04), 30
        )
        self.assertTrue(bottom_dev_distance == 6)
        self.assertTrue(bottom_dev_type == 1)

        X = [
            5.2812963,
            5.8084316,
            6.385296,
            7.0218363,
            7.7279983,
            7.180971,
            7.0914583,
            7.7976203,
            7.7578363,
            7.499242,
            7.7478905,
            7.7976203,
            8.06616,
            7.270485,
            7.0218363,
            6.544431,
            6.524539,
            6.3057284,
            6.614053,
            6.8229175,
            6.922377,
            6.783134,
            6.5046473,
            6.42508,
            6.335566,
            6.554377,
            6.405188,
            6.5742693,
            6.5842147,
            6.405188,
            6.186377,
            6.246053,
            6.1267014,
            5.7885394,
            5.7885394,
            6.047134,
            6.027242,
            6.0670257,
            6.236107,
            6.8627014,
            7.5489717,
            8.304864,
            7.9667015,
            8.762377,
            9.637619,
            9.945944,
            10.254269,
            9.2298355,
            9.319349,
            9.299458,
            9.050809,
            8.503782,
            9.060755,
            8.812106,
            9.190052,
            9.17016,
            9.040863,
            8.155674,
            8.21535,
            8.086053,
            8.105944,
            8.086053,
            8.086053,
            7.6782684,
            7.718053,
            7.688215,
            7.5788093,
            8.334702,
            8.314809,
            8.384431,
            7.787674,
            7.8871336,
            7.7578363,
            8.185512,
            8.384431,
            7.7578363,
            8.245188,
            8.294917,
            8.4142685,
            8.4142685,
            8.046268,
            7.9667015,
            7.489296,
            7.041728,
            7.1014037,
            6.8726473,
            6.5842147,
            6.7731876,
            6.9024854,
            6.8726473,
            7.0914583,
            7.270485,
            6.9919987,
            6.8328633,
            6.7632422,
            6.6836743,
            6.6239986,
            6.365404,
            6.5842147,
            6.544431,
            6.69362,
            6.6041064,
            7.260539,
            7.121296,
            7.2804313,
            7.081512,
            7.3301606,
            7.628539,
            8.394376,
            8.384431,
            7.7976203,
            7.857296,
            7.459458,
            7.6086473,
            7.3500524,
            7.69816,
            7.718053,
            7.489296,
            7.340107,
            7.011891,
            6.9919987,
            7.688215,
            7.8473496,
            7.8871336,
            7.9766474,
            7.946809,
            8.105944,
            8.314809,
            8.354592,
            7.5589175,
            7.1710258,
            7.1411877,
            7.210809,
            7.2406473,
            7.260539,
            7.1909175,
            7.310269,
            7.1610794,
            7.2307014,
            7.69816,
            7.767782,
            7.8771877,
            7.7478905,
            7.5788093,
            7.6583767,
            8.06616,
            7.837404,
            7.8970795,
            7.6782684,
            7.588755,
            8.255134,
            8.056215,
            7.946809,
            7.8473496,
            7.827458,
            8.006485,
            7.7081065,
            7.9667015,
            8.195457,
            8.672863,
            8.155674,
            8.48389,
            8.1158905,
            8.095999,
            7.8871336,
            7.8970795,
            7.827458,
            7.399782,
            7.5091877,
            7.42962,
            7.5191336,
            7.52908,
            7.588755,
            7.5191336,
            7.5489717,
            7.489296,
            7.3599987,
            7.4793496,
            7.3898363,
            7.3599987,
            7.21,
            7.27,
            7.2,
            7.26,
            7.09,
            6.92,
            6.96,
            7.01,
            7.03,
            7.1,
            7.1,
            7.02,
            7.08,
            7.01,
            6.92,
            7.05,
            6.9,
            6.83,
            6.96,
            6.97,
            7.15,
            7.11,
            7.29,
            7.21,
            7.22,
            7.06,
            7.08,
            7.12,
            7.25,
            7.09,
            7.24,
            7.31,
            7.32,
            7.72,
        ]
        X = np.array(X)

        bottom_dev_type, bottom_dev_distance = rsi_bottom_dev_detect(
            X, (0.01, -0.01), 30
        )
        self.assertTrue(bottom_dev_distance == 19)
        self.assertTrue(bottom_dev_type == 2)

        bottom_dev_type, bottom_dev_distance = rsi_bottom_dev_detect(
            X, (0.05, -0.05), 20
        )
        self.assertTrue(bottom_dev_distance == None)
        self.assertTrue(bottom_dev_type == 0)

    def test_valley_detect(self):
        data = np.array(
            [
                19.798132,
                19.621275,
                19.277388,
                19.6311,
                19.807957,
                19.611448,
                19.680227,
                18.835245,
                18.422579,
                17.666025,
                18.147469,
                19.965162,
                21.959715,
                22.961905,
                20.662766,
                18.835245,
                19.238085,
                17.901834,
                18.176945,
                16.359251,
                15.268634,
                15.55357,
                14.993524,
                15.966235,
                16.231522,
                15.976061,
                16.3789,
                16.654013,
                16.555758,
                16.988075,
                16.879995,
                16.968424,
                16.712965,
                16.821045,
                17.105978,
                17.184582,
                17.449867,
                16.663836,
                16.879995,
                17.09,
                17.1,
                17.12,
                17.24,
                17.45,
                17.6,
                17.85,
                17.68,
                17.56,
                17.13,
                17.26,
                17.33,
                17.1,
                17.11,
                17.38,
                16.86,
                17.04,
                16.98,
                16.75,
                16.88,
                16.96,
                17.2,
                17.49,
                17.26,
                17.45,
                17.74,
                17.66,
                17.45,
                17.33,
                17.41,
                17.47,
                17.27,
                17.08,
                17.31,
                17.3,
                16.94,
                18.63,
                19.1,
                18.5,
                18.37,
                18.52,
                18.26,
                18.36,
                18.54,
                18.39,
                18.15,
                18.03,
                16.85,
                16.82,
                17.03,
                17.21,
                17.25,
                17.23,
                17.17,
                17.51,
                17.55,
                17.55,
                17.5,
                17.62,
                17.71,
                17.84,
                17.89,
                18.35,
                18.07,
                18.15,
                17.5,
                17.88,
                17.76,
                17.24,
                17.2,
                17.54,
                17.57,
                17.49,
                17.85,
                17.3,
                17.13,
                17.18,
                17.35,
                16.89,
                16.21,
                17.83,
            ]
        )
        actual = valley_detect(data, thresh=(0.05, -0.02))
        exp = (int(1), 0.099938)
        self.assertAlmostEqual(actual[0], exp[0], 4)
        self.assertAlmostEqual(actual[1], exp[1], 4)

        actual = valley_detect(data, thresh=(0.5, -0.5))
        exp = (None, None)
        self.assertEqual(actual[0], exp[0])
        self.assertEqual(actual[1], exp[1])

    def test_rsi_predict_price(self):

        np.random.seed(78)
        pct_change = np.random.random(120) / 10
        signal = np.array(
            np.random.choice(
                list(np.repeat([1, -1], len(pct_change) / 2)), len(pct_change)
            )
        )
        pct_change *= signal
        price = 10 + np.cumprod(pct_change + 1)

        data = np.array(price, dtype=[("close", "f4")])["close"]
        thresh = [0.01, -0.01]
        actual = rsi_predict_price(data[:-1], thresh=thresh)
        exp = (11.181273684882008, 12.931359226859442)

        self.assertAlmostEqual(actual[0], exp[0], places=2)
        self.assertAlmostEqual(actual[1], exp[1], places=2)

        thresh = [0.1, -0.1]
        actual = rsi_predict_price(data[:-1], thresh=thresh)
        exp = None

        self.assertTrue(actual[0] == exp)
        self.assertTrue(actual[1] == exp)

    def test_rsi_watermarks(self):
        np.random.seed(78)
        pct_change = np.random.random(120) / 10
        signal = np.array(
            np.random.choice(
                list(np.repeat([1, -1], len(pct_change) / 2)), len(pct_change)
            )
        )
        pct_change *= signal
        price = 10 + np.cumprod(pct_change + 1)

        data = np.array(price, dtype=[("close", "f4")])
        thresh = [0.01, -0.01]
        low_watermark, high_watermark, _ = rsi_watermarks(data, thresh)
        exp = (16.72408, 78.37833)

        self.assertAlmostEqual(low_watermark, exp[0], places=2)
        self.assertAlmostEqual(high_watermark, exp[1], places=2)

        thresh = [0.1, -0.1]
        low_watermark, high_watermark, _ = rsi_watermarks(data, thresh)

        self.assertTrue(low_watermark == None)
        self.assertTrue(high_watermark == None)

    def test_plateaus(self):
        # fmt off
        numbers = np.array(
            [
                4.69,
                4.44,
                4.56,
                4.61,
                4.71,
                4.76,
                4.78,
                4.76,
                4.75,
                4.78,
                4.93,
                4.91,
                4.89,
                4.95,
                5.03,
                5.16,
                5.06,
                5.1,
                5.54,
                5.89,
                5.52,
                5.73,
                5.59,
                6.15,
                5.64,
                5.55,
                5.0,
                4.96,
                4.54,
                4.99,
                4.82,
                4.68,
                4.92,
                4.87,
                4.76,
                4.9,
                4.93,
                4.91,
                5.0,
                5.07,
                5.19,
                5.01,
                5.22,
                5.16,
                5.08,
                5.08,
                4.85,
                5.1,
                5.11,
                5.15,
                5.2,
                5.36,
                5.29,
                5.17,
                5.19,
                5.17,
                5.09,
                4.96,
                4.99,
                4.99,
                4.99,
                4.97,
                5.0,
                4.97,
                5.02,
                4.99,
                4.93,
                5.0,
                5.03,
                5.07,
                5.15,
                5.04,
                5.1,
                5.13,
                5.19,
                5.12,
                5.02,
                5.05,
                5.07,
                5.18,
                5.17,
                5.2,
                5.11,
                4.86,
                5.05,
                5.11,
                5.12,
                5.08,
                5.08,
                5.13,
                5.13,
                5.12,
                5.11,
                5.11,
                5.01,
                4.82,
                4.75,
                4.81,
                4.82,
                4.82,
                4.8,
                4.78,
                4.86,
                4.86,
                4.84,
                4.87,
                4.87,
                4.87,
                4.89,
                4.92,
                4.94,
                4.88,
                4.91,
                4.94,
                4.98,
                5.01,
                4.92,
                5.41,
                5.28,
                5.26,
            ],
            dtype="f4",
        )
        # fmt on

        actual = plateaus(numbers, 20)
        exp = [(38, 57), (94, 22)]
        self.assertListEqual(actual, exp)

        actual = plateaus(numbers, 40)
        exp = [(25, 94)]
        self.assertListEqual(actual, exp)

    def test_energy_hump(self):
        # fmt off
        # code: 600698, 2022-9-23
        vol = np.array(
            [
                13305200.0,
                9819300.0,
                16486890.0,
                11092872.0,
                22371092.0,
                10501620.0,
                10192300.0,
                9899800.0,
                23570250.0,
                15113720.0,
                14216630.0,
                11755430.0,
                25220294.0,
                52902921.0,
                29877256.0,
                15860442.0,
                12088481.0,
                17869751.0,
                18616500.0,
                31650412.0,
                18756039.0,
                16046600.0,
                21236860.0,
                72464686.0,
                54331935.0,
                20780070.0,
                14145201.0,
                18330900.0,
                12892611.0,
                10807260.0,
                15612139.0,
                10050136.0,
                23375539.0,
                14055551.0,
                13175853.0,
                11281100.0,
                8726912.0,
                7535731.0,
                10113600.0,
                9583151.0,
                7149251.0,
                5331765.0,
                5905100.0,
                6559300.0,
                4236761.0,
                7642360.0,
                16267304.0,
                9449934.0,
                7464631.0,
                35333324.0,
                18455600.0,
                10340800.0,
                8233600.0,
                11451096.0,
                8448476.0,
                5718243.0,
                48649542.0,
                75480001.0,
                42613115.0,
                36603489.0,
            ]
        )

        frames = np.array(
            [
                "2022-07-01T00:00:00",
                "2022-07-04T00:00:00",
                "2022-07-05T00:00:00",
                "2022-07-06T00:00:00",
                "2022-07-07T00:00:00",
                "2022-07-08T00:00:00",
                "2022-07-11T00:00:00",
                "2022-07-12T00:00:00",
                "2022-07-13T00:00:00",
                "2022-07-14T00:00:00",
                "2022-07-15T00:00:00",
                "2022-07-18T00:00:00",
                "2022-07-19T00:00:00",
                "2022-07-20T00:00:00",
                "2022-07-21T00:00:00",
                "2022-07-22T00:00:00",
                "2022-07-25T00:00:00",
                "2022-07-26T00:00:00",
                "2022-07-27T00:00:00",
                "2022-07-28T00:00:00",
                "2022-07-29T00:00:00",
                "2022-08-01T00:00:00",
                "2022-08-02T00:00:00",
                "2022-08-03T00:00:00",
                "2022-08-04T00:00:00",
                "2022-08-05T00:00:00",
                "2022-08-08T00:00:00",
                "2022-08-09T00:00:00",
                "2022-08-10T00:00:00",
                "2022-08-11T00:00:00",
                "2022-08-12T00:00:00",
                "2022-08-15T00:00:00",
                "2022-08-16T00:00:00",
                "2022-08-17T00:00:00",
                "2022-08-18T00:00:00",
                "2022-08-19T00:00:00",
                "2022-08-22T00:00:00",
                "2022-08-23T00:00:00",
                "2022-08-24T00:00:00",
                "2022-08-25T00:00:00",
                "2022-08-26T00:00:00",
                "2022-08-29T00:00:00",
                "2022-08-30T00:00:00",
                "2022-08-31T00:00:00",
                "2022-09-01T00:00:00",
                "2022-09-02T00:00:00",
                "2022-09-05T00:00:00",
                "2022-09-06T00:00:00",
                "2022-09-07T00:00:00",
                "2022-09-08T00:00:00",
                "2022-09-09T00:00:00",
                "2022-09-13T00:00:00",
                "2022-09-14T00:00:00",
                "2022-09-15T00:00:00",
                "2022-09-16T00:00:00",
                "2022-09-19T00:00:00",
                "2022-09-20T00:00:00",
                "2022-09-21T00:00:00",
                "2022-09-22T00:00:00",
                "2022-09-23T00:00:00",
            ],
            dtype="datetime64[s]",
        )
        # fmt on
        bars = np.empty(
            (len(vol),), dtype=[("volume", "f8"), ("frame", "datetime64[s]")]
        )
        bars["volume"] = vol
        bars["frame"] = frames

        dist, length = energy_hump(bars)
        self.assertEqual(3, dist)
        self.assertEqual(44, length)

        result = energy_hump(bars, 10)
        self.assertIsNone(result)
