import unittest

import numpy as np

from omicron.extensions.np import (
    array_math_round,
    bin_cut,
    count_between,
    dataframe_to_structured_array,
    fill_nan,
    floor,
    join_by_left,
    numpy_append_fields,
    remove_nan,
    replace_zero,
    rolling,
    shift,
)
from omicron.models.timeframe import TimeFrame


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

    def test_math_round(self):
        raw = [i / 10 for i in range(10)]
        arr = np.array(raw)

        exp_raw = [0] * 5 + [1] * 5

        np.testing.assert_array_equal(np.array(exp_raw), array_math_round(arr, 0))

        actual = array_math_round(raw, 0)
        np.testing.assert_array_equal(np.array(exp_raw), actual)

        self.assertEqual(0.16, array_math_round(0.155, 2))
        self.assertEqual(0.15, array_math_round(0.154, 2))
