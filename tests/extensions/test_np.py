import datetime
import unittest

import numpy as np
from numpy.testing import assert_array_equal

from omicron.extensions.np import (
    array_math_round,
    array_price_equal,
    bars_since,
    bin_cut,
    count_between,
    dataframe_to_structured_array,
    fill_nan,
    find_runs,
    floor,
    join_by_left,
    numpy_append_fields,
    remove_nan,
    replace_zero,
    rolling,
    shift,
    smallest_n_argpos,
    to_pydatetime,
    top_n_argpos,
)


class NpTest(unittest.TestCase):
    def test_count_between(self):
        """day frames:
         20050104, 20050105, 20050106, 20050107, 20050110, 20050111,
        20050112, 20050113, 20050114, 20050117
        """
        arr = [
            20050104,
            20050105,
            20050106,
            20050107,
            20050110,
            20050111,
            20050112,
            20050113,
            20050114,
            20050117,
        ]

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
        arr = [
            20050104,
            20050105,
            20050106,
            20050107,
            20050110,
            20050111,
            20050112,
            20050113,
            20050114,
            20050117,
        ]

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

    def test_remove_nan(self):
        a = np.array([1, 2, np.nan, 3, np.nan, 4, 5, 6])
        actual = remove_nan(a)
        exp = [1, 2, 3, 4, 5, 6]
        self.assertListEqual(exp, actual.tolist())

    def test_fill_nan(self):
        arr = np.arange(10) / 3.0
        arr[0:2] = np.nan

        actual = fill_nan(arr)
        exp = arr.copy()
        exp[0:2] = 2 / 3.0

        np.testing.assert_array_almost_equal(exp, actual, 3)

        arr = np.arange(10) / 3.0
        arr[2:5] = np.nan

        actual = fill_nan(arr)
        exp = arr.copy()
        exp[2:5] = 1 / 3.0
        np.testing.assert_array_almost_equal(exp, actual, 3)

        arr = np.array([np.nan] * 5)
        try:
            fill_nan(arr)
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

    def test_rolling(self):
        arr = np.arange(10)
        func = np.mean
        win = 3
        actual = rolling(arr, win, func)
        exp = np.convolve(arr, np.ones(win) / win, mode="valid")
        np.testing.assert_array_almost_equal(exp, actual, 3)

    def test_dataframe_to_structured_array(self):
        import pandas as pd

        data = np.ones(
            (3,),
            dtype=[("a", "f4"), ("b", "f4"), ("c", "f4"), ("d", "f4"), ("e", "f4")],
        )
        df = pd.DataFrame(data.tolist(), columns=["a", "b", "c", "d", "e"])

        # 不提供dtype，默认为dataframe的dtype
        actual = dataframe_to_structured_array(df)

        np.testing.assert_array_equal(data.tolist(), actual.tolist())
        self.assertEqual([("", "|O")], actual.dtype.descr)

        # 提供了dtype
        actual = dataframe_to_structured_array(
            df, dtypes=[("a", "f4"), ("b", "f4"), ("c", "f4"), ("d", "f4"), ("e", "f4")]
        )
        np.testing.assert_array_equal(data, actual)

        # 如果需要将index也转换成structured array
        dtypes = [
            ("a", "f4"),
            ("b", "f4"),
            ("c", "f4"),
            ("d", "f4"),
            ("e", "f4"),
            ("index", "i4"),
        ]
        actual = dataframe_to_structured_array(df, dtypes=dtypes)
        np.testing.assert_array_equal(data, actual[["a", "b", "c", "d", "e"]])
        np.testing.assert_array_equal(df.index.values, actual["index"])

    def test_bin_cut(self):
        arr = [1, 2, 3, 4, 5]

        expected = [
            [[1, 2, 3, 4, 5]],
            [[1, 3, 5], [2, 4]],
            [[1, 4], [2, 5], [3]],
            [[1], [2], [3], [4], [5]],
            [[1], [2], [3], [4], [5]],
        ]
        for i, bins in enumerate([1, 2, 3, 5, 10]):
            self.assertListEqual(expected[i], bin_cut(arr, bins))

    def test_array_math_round(self):
        raw = [i / 10 for i in range(10)]
        arr = np.array(raw)

        exp_raw = [0] * 5 + [1] * 5

        np.testing.assert_array_equal(np.array(exp_raw), array_math_round(arr, 0))

        actual = array_math_round(raw, 0)
        np.testing.assert_array_equal(np.array(exp_raw), actual)

        self.assertEqual(0.16, array_math_round(0.155, 2))
        self.assertEqual(0.15, array_math_round(0.154, 2))

    def test_array_price_equal(self):
        limits = np.array(
            [
                (datetime.date(2022, 3, 23), 3.45, 3.45, 2.83),
                (datetime.date(2022, 3, 24), 3.8, 3.8, 3.11),
                (datetime.date(2022, 3, 25), 3.76, 4.18, 3.42),
                (datetime.date(2022, 3, 28), 3.76, 4.14, 3.38),
                (datetime.date(2022, 3, 29), 3.38, 4.14, 3.38),
                (datetime.date(2022, 3, 30), 3.07, 3.72, 3.04),
                (datetime.date(2022, 3, 31), 3.14, 3.38, 2.76),
                (datetime.date(2022, 4, 1), 2.94, 3.45, 2.83),
                (datetime.date(2022, 4, 6), 3.01, 3.23, 2.65),
            ],
            dtype=[
                ("frame", "O"),
                ("close", "<f4"),
                ("high_limit", "<f4"),
                ("low_limit", "<f4"),
            ],
        )

        # ensure not equal
        limits[0]["close"] += 1e-6
        with self.assertRaises(AssertionError):
            assert all(limits["close"][:2] == limits["high_limit"][:2])

        # now all elements should equal
        expected = [True, True, False, False, False, False, False, False, False]
        actual = array_price_equal(limits["close"], limits["high_limit"])

        assert_array_equal(expected, actual)

    def test_to_pydatetime(self):
        dt64 = np.datetime64("2017-10-24 05:34:20.123456")
        self.assertEqual(
            datetime.datetime(2017, 10, 24, 5, 34, 20, 123456), to_pydatetime(dt64)
        )

    def test_find_runs(self):
        a = [1, 1, 2, 2, 3, 3, 3]
        value, pos, length = find_runs(a)
        self.assertListEqual([1, 2, 3], value.tolist())
        self.assertListEqual([0, 2, 4], pos.tolist())
        self.assertListEqual([2, 2, 3], length.tolist())

    def test_barssince(self):
        condition = [False, True]
        self.assertEqual(0, bars_since(condition))

        condition = [True, False]
        self.assertEqual(1, bars_since(condition))

        condition = [True, True, False]
        self.assertEqual(1, bars_since(condition))

        condition = [True, True, False, True]
        self.assertEqual(0, bars_since(condition))

        condition = [True, True, False, False]
        self.assertEqual(2, bars_since(condition))

    def test_top_n_argpos(self):
        arr = [4, 3, 9, 8, 5, 2, 1, 0, 6, 7]
        actual = top_n_argpos(arr, 2)
        exp = [2, 3]
        self.assertListEqual(exp, actual.tolist())

    def test_smallest_n_argpos(self):
        arr = [7, 5, 1, 4, 0, 3, 2, 6, 9, 8]
        actual = smallest_n_argpos(arr, 2)
        exp = [4, 2]
        self.assertListEqual(exp, actual.tolist())
