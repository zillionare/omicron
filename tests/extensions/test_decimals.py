import unittest

from omicron.extensions.decimals import math_round


class TestDecimal(unittest.TestCase):
    def test_math_round(self):
        self.assertEqual(1.0, math_round(0.5, 0))
        self.assertEqual(0.0, round(0.5, 0))
