import unittest

import numpy as np

from omicron.core.accelerate import (
    count_between,
    floor,
    join_by_left,
    minute_frames_floor,
    numpy_append_fields,
    shift,
)
from omicron.core.timeframe import tf


class AccelerateTest(unittest.TestCase):
    def test_count_between(self):
        """day frames:
         20050104, 20050105, 20050106, 20050107, 20050110, 20050111,
        20050112, 20050113, 20050114, 20050117
        """
        arr = tf.day_frames

        actual = count_between(arr, start=20050104, end=20050111)
        self.assertEqual(6, actual)

        actual = count_between(arr, 20050104, 20050109)
        self.assertEqual(4, actual)

        actual = count_between(arr, 20050101, 20050109)
        self.assertEqual(4, actual)

        actual = count_between(arr[:5], 20050101, 20050120)
        self.assertEqual(4, actual)

    def test_shift(self):
        """day frames:
        20050104, 20050105, 20050106, 20050107, 20050110, 20050111,
        20050112, 20050113, 20050114, 20050117
        """
        arr = tf.day_frames

        self.assertEqual(20050105, shift(arr, 20050104, 1))
        self.assertEqual(20050104, shift(arr, 20050105, -1))
        self.assertEqual(20050110, shift(arr, 20050107, 1))
        self.assertEqual(20050104, shift(arr, 20050101, 1))
        self.assertEqual(20050107, shift(arr[:5], 20050120, -1))
        self.assertEqual(20050120, shift(arr[:5], 20050120, 1))

    def test_numpy_append_fields(self):
        old = np.array([i for i in range(10)], dtype=[("col1", "<f4")])

        new_list = [2 * i for i in range(10)]

        actual = numpy_append_fields(old, "new_col", new_list, [("new_col", "<f4")])
        expected = np.array(
            [(i, i * 2) for i in range(10)], dtype=[("col1", "<f4"), ("new_col", "<f4")]
        )
        self.assertListEqual(expected.tolist(), actual.tolist())

        multi_cols = [actual["col1"].tolist(), actual["new_col"].tolist()]
        numpy_append_fields(
            old, ("col3", "col4"), multi_cols, [("col3", "<f4"), ("col4", "<f4")]
        )

    def test_join_by_left(self):
        r1 = np.array([(1, 2), (1, 3), (2, 3)], dtype=[("seq", "i4"), ("score", "i4")])
        r2 = np.array([(1, 5), (4, 7)], dtype=[("seq", "i4"), ("age", "i4")])

        actual = join_by_left("seq", r1, r2)
        self.assertListEqual([(1, 2, 5), (1, 3, 5), (2, 3, None)], actual.tolist())

        actual = join_by_left("seq", r1, r2, False)

        # actual[2][2]是随机数
        self.assertListEqual([(1, 2, 5), (1, 3, 5)], actual.tolist()[:2])

    def test_floor(self):
        a = [3, 6, 9]
        self.assertEqual(3, floor(a, -1))
        self.assertEqual(9, floor(a, 9))
        self.assertEqual(3, floor(a, 4))
        self.assertEqual(9, floor(a, 10))

    def test_minute_frames_floor(self):
        ticks = [600, 630, 660, 690, 810, 840, 870, 900]
        self.assertTupleEqual((900, -1), minute_frames_floor(ticks, 545))
        self.assertTupleEqual((600, 0), minute_frames_floor(ticks, 600))
        self.assertTupleEqual((600, 0), minute_frames_floor(ticks, 605))
        self.assertTupleEqual((870, 0), minute_frames_floor(ticks, 899))
        self.assertTupleEqual((900, 0), minute_frames_floor(ticks, 900))
        self.assertTupleEqual((900, 0), minute_frames_floor(ticks, 905))
