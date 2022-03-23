import unittest

import numpy as np

from omicron.talib.metrics import mean_absolute_error, pct_error


class MetricsTest(unittest.TestCase):
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
