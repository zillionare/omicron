import unittest

from omicron.extensions.decimals import math_round, price_equal


class TestDecimal(unittest.TestCase):
    def test_math_round(self):
        self.assertEqual(1.0, math_round(0.5, 0))
        self.assertEqual(0.0, round(0.5, 0))
        self.assertEqual(-1.0, math_round(-0.5, 0))
        self.assertEqual(-1.0, math_round(-0.99, 1))
        self.assertEqual(-1.0, math_round(-0.995, 2))

    def test_price_equal(self):
        self.assertTrue(price_equal(1.0, 1.0))
        self.assertTrue(price_equal(1.01, 1.0098))
        self.assertTrue(price_equal(1.01444, 1.0099555))
