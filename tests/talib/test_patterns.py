import unittest

import numpy as np

from omicron.talib.patterns import (
    BreakoutFlag,
    breakout,
    peaks_and_valleys,
    support_resist_lines,
)


class TestPatterns(unittest.TestCase):
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

    def test_support_resist_lines(self):
        np.random.seed(1978)
        X = np.cumprod(1 + np.random.randn(100) * 0.01)

        # test resist, support by predict next point
        support, resist = support_resist_lines(X)
        self.assertAlmostEqual(1.013318988311881, resist(100))
        self.assertAlmostEqual(0.9539414071057044, support(100))

    def test_breakout(self):
        np.random.seed(1978)
        X = np.cumprod(1 + np.random.randn(100) * 0.01)

        # test resist, support by predict next point
        self.assertEqual(BreakoutFlag.NONE, breakout(X))

        y = np.concatenate([X, [1.1]])
        self.assertEqual(BreakoutFlag.UP, breakout(y))

        y = np.concatenate([X, [0.9]])
        self.assertEqual(BreakoutFlag.DOWN, breakout(y))

        y = np.concatenate([X, [0.9, 0.8]])
        self.assertEqual(BreakoutFlag.NONE, breakout(y, confirm=2))

        y = np.concatenate([X, [0.83, 0.8]])
        self.assertEqual(BreakoutFlag.DOWN, breakout(y, confirm=2))
