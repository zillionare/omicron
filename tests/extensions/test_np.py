import unittest

import numpy as np

from omicron.extensions.np import (
    count_between,
    fillnan,
    filternan,
    find_runs,
    floor,
    join_by_left,
    numpy_append_fields,
    replace_zero,
    rolling,
    shift,
    top_n_argpos,
)
from omicron.models.calendar import Calendar as cal


class NpTest(unittest.TestCase):
    def test_count_between(self):
        """day frames:
         20050104, 20050105, 20050106, 20050107, 20050110, 20050111,
        20050112, 20050113, 20050114, 20050117
        """
        arr = cal.day_frames

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
        arr = cal.day_frames

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

    def test_find_runs(self):
        a = [
            1,
            1,
            2,
            2,
            3,
            3,
            3,
        ]
        value, pos, length = find_runs(a)
        self.assertListEqual([1, 2, 3], value.tolist())
        self.assertListEqual([0, 2, 4], pos.tolist())
        self.assertListEqual([2, 2, 3], length.tolist())

    def test_filternan(self):
        a = np.array([1, 2, np.nan, 3, np.nan, 4, 5, 6])
        actual = filternan(a)
        exp = [1, 2, 3, 4, 5, 6]
        self.assertListEqual(exp, actual.tolist())

    def test_fillnan(self):
        arr = np.arange(10) / 3.0
        arr[0:2] = np.nan

        actual = fillnan(arr)
        exp = arr.copy()
        exp[0:2] = 2 / 3.0

        np.testing.assert_array_almost_equal(exp, actual, 3)

        arr = np.arange(10) / 3.0
        arr[2:5] = np.nan

        actual = fillnan(arr)
        exp = arr.copy()
        exp[2:5] = 1 / 3.0
        np.testing.assert_array_almost_equal(exp, actual, 3)

        arr = np.array([np.nan] * 5)
        try:
            fillnan(arr)
        except ValueError:
            self.assertTrue(True)

    def test_replace_zero(self):
        arr = np.array([0, 1, 2, 3, 4])
        actual = replace_zero(arr)
        self.assertListEqual([1, 1, 2, 3, 4], actual.tolist())

        arr = np.array([1, 0, 2, 3, 4])
        self.assertListEqual([1, 1, 2, 3, 4], replace_zero(arr).tolist())

        arr = np.array([1, 2, 3, 0, 4])
        self.assertListEqual([1, 2, 3, 3, 4], replace_zero(arr).tolist())

        arr = np.array([1, 2, 3, 4, 0])
        self.assertListEqual([1, 2, 3, 4, 4], replace_zero(arr).tolist())

        arr = np.array([1, 2, 0, 4, 5])
        self.assertListEqual([1, 2, 0.001, 4, 5], replace_zero(arr, 0.001).tolist())

    def test_top_n_argpos(self):
        arr = [4, 3, 9, 8, 5, 2, 1, 0, 6, 7]
        actual = top_n_argpos(arr, 2)
        exp = [2, 3]
        self.assertListEqual(exp, actual.tolist())

    def test_rolling(self):
        arr = np.arange(10)
        func = np.mean
        win = 3
        actual = rolling(arr, win, func)
        exp = np.convolve(arr, np.ones(win) / win, mode="valid")
        np.testing.assert_array_almost_equal(exp, actual, 3)
