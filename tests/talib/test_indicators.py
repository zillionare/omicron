import unittest

import numpy as np

from omicron.talib import *


class IndicatorTest(unittest.TestCase):
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
