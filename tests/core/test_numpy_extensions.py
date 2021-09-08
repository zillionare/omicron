from omicron.core.numpy_extensions import (
    count_between,
    dataframe_to_structured_array,
    ffill_na,
    shift,
    numpy_array_to_dict,
    numpy_append_fields,
    dict_to_numpy_array,
    join_by_left,
    floor,
)
import unittest


from omicron.core.timeframe import tf
import numpy as np
import pandas as pd


class NumpyExtensionsTest(unittest.TestCase):
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

    def test_dict_to_numpy_array(self):
        arr = {"aaron": 5, "jack": 6}

        dtype = [("name", "<S8"), ("score", "<i4")]
        actual = dict_to_numpy_array(arr, dtype)
        expected = np.array(
            [("aaron", 5), ("jack", 6)], dtype=[("name", "S8"), ("score", "<i4")]
        )
        np.testing.assert_array_equal(expected, actual)

    def test_ffill_na(self):
        arr = np.arange(6, dtype=np.float32)
        arr[3:5] = np.nan
        np.testing.assert_array_equal([0.0, 1.0, 2.0, 2.0, 2.0, 5.0], ffill_na(arr))

        arr[0:2] = np.nan
        np.testing.assert_almost_equal(
            [np.nan, np.nan, 2.0, 2.0, 2.0, 5.0], ffill_na(arr)
        )

    def test_dataframe_to_structured_array(self):
        df = pd.DataFrame(data=[2 * i for i in range(3)], columns=["seq"])
        arr = dataframe_to_structured_array(df, [("frame", "<i8"), ("seq", "<i8")])

        expected = np.array(
            [(0, 0), (1, 2), (2, 4)], dtype=[("frame", "<i8"), ("seq", "<i8")]
        )

        np.testing.assert_array_equal(expected, arr)

        arr = dataframe_to_structured_array(df, [("seq", "<f8")])
        expected = np.array([(0,), (2,), (4,)], dtype=[("seq", "<f8")])

        # not sure why we cannot use np.testing.assert_array_almost_equal here
        self.assertTrue(np.all(arr == expected))

        arr = dataframe_to_structured_array(df)
        self.assertTrue(np.all(arr == expected))
