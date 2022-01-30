import unittest

import numpy as np

from omicron.dal.influx.serialize import DataFrameDeserializer, NumpyDeserializer


class SerializerTest(unittest.TestCase):
    def test_dataframe_unserializer(self):
        data = ",result,table,_time,code,name,close,open\r\n,_result,0,2019-01-01T09:31:00Z,000002.XSHE,国联证券,0.2,0.1\r\n,_result,0,2019-01-01T09:36:00Z,000002.XSHE,国联证券,0.2,0.1\r\n,_result,1,2019-01-01T09:32:00Z,000003.XSHE,上海银行,0.2,0.1\r\n,_result,1,2019-01-01T09:37:00Z,000003.XSHE,上海银行,0.2,0.1\r\n,_result,2,2019-01-01T09:33:00Z,000004.XSHE,中国银行,0.2,0.1\r\n,_result,2,2019-01-01T09:38:00Z,000004.XSHE,中国银行,0.2,0.1\r\n,_result,3,2019-01-01T09:34:00Z,000005.XSHE,中国平安,0.2,0.1\r\n,_result,3,2019-01-01T09:39:00Z,000005.XSHE,中国平安,0.2,0.1\r\n\r\n"

        # default
        des = DataFrameDeserializer(
            parse_dates=["_time"],
        )
        df = des(data)
        self.assertEqual(8, len(df))
        self.assertEqual(8, len(df.columns))
        self.assertIn("_time", df.columns)
        self.assertEqual(np.dtype("datetime64[ns]"), df["_time"].dtype)

        # it's not ordered
        frames = df["_time"].values
        self.assertTrue(frames[2] < frames[1])

        # keep cols
        des = DataFrameDeserializer(
            names=["frame", "open", "close"], usecols=[3, 6, 7], parse_dates="frame"
        )
        df = des(data)
        self.assertEqual(3, len(df.columns))
        self.assertListEqual(["frame", "open", "close"], df.columns.tolist())
        self.assertEqual(8, len(df))

        # sort by

        des = DataFrameDeserializer(
            sort_values="frame",
            parse_dates="frame",
            names=["frame", "open", "close"],
            usecols=[3, 6, 7],
        )

        df = des(data)
        # it's ordered
        frames = df["frame"].values
        self.assertTrue(frames[2] > frames[1])

        # `data` is bytes array
        data = data.encode("utf-8")
        des = DataFrameDeserializer(encoding="utf-8")
        df = des(data)
        self.assertEqual(8, len(df))

    def test_numpy_serializer(self):
        data = ",result,table,_time,code,name,close,open\r\n,_result,0,2019-01-01T09:31:00Z,000002.XSHE,国联证券,0.2,0.1\r\n,_result,0,2019-01-01T09:36:00Z,000002.XSHE,国联证券,0.2,0.1\r\n,_result,1,2019-01-01T09:32:00Z,000003.XSHE,上海银行,0.2,0.1\r\n,_result,1,2019-01-01T09:37:00Z,000003.XSHE,上海银行,0.2,0.1\r\n,_result,2,2019-01-01T09:33:00Z,000004.XSHE,中国银行,0.2,0.1\r\n,_result,2,2019-01-01T09:38:00Z,000004.XSHE,中国银行,0.2,0.1\r\n,_result,3,2019-01-01T09:34:00Z,000005.XSHE,中国平安,0.2,0.1\r\n,_result,3,2019-01-01T09:39:00Z,000005.XSHE,中国平安,0.2,0.1\r\n\r\n"

        dtype = [
            ("frame", "datetime64[s]"),
            ("code", "<U12"),
            ("name", "<U4"),
            ("close", "<f4"),
            ("open", "<f4"),
        ]

        deserializer = NumpyDeserializer(dtype, use_cols=[3, 4, 5, 6, 7], skip_rows=1)
        actual = deserializer(data)

        exp = np.array(
            [
                ("2019-01-01T09:31:00", "000002.XSHE", "国联证券", 0.2, 0.1),
                ("2019-01-01T09:36:00", "000002.XSHE", "国联证券", 0.2, 0.1),
                ("2019-01-01T09:32:00", "000003.XSHE", "上海银行", 0.2, 0.1),
                ("2019-01-01T09:37:00", "000003.XSHE", "上海银行", 0.2, 0.1),
                ("2019-01-01T09:33:00", "000004.XSHE", "中国银行", 0.2, 0.1),
                ("2019-01-01T09:38:00", "000004.XSHE", "中国银行", 0.2, 0.1),
                ("2019-01-01T09:34:00", "000005.XSHE", "中国平安", 0.2, 0.1),
                ("2019-01-01T09:39:00", "000005.XSHE", "中国平安", 0.2, 0.1),
            ],
            dtype=dtype,
        )
        self.assertEqual(str(exp), str(actual))

        # decode bytes array
        des = NumpyDeserializer(
            dtype, use_cols=[3, 4, 5, 6, 7], skip_rows=1, encoding="utf-8"
        )
        actual = des(data.encode("utf-8"))

        self.assertEqual(str(exp), str(actual))
