import unittest

from omicron.extensions.decimals import equal_price, math_round


class TestDecimal(unittest.TestCase):
    def test_math_round(self):
        self.assertEqual(1.0, math_round(0.5, 0))
        self.assertEqual(0.0, round(0.5, 0))

    def test_equal_price(self):
        self.assertTrue(equal_price(1.0, 1.0))
        self.assertTrue(equal_price(1.01, 1.0098))
        self.assertTrue(equal_price(1.01444, 1.0099555))
