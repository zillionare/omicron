import datetime
import unittest

import arrow
import cfg4py
import numpy as np
import pandas as pd
from coretypes import bars_cols, bars_dtype

import omicron
from omicron.core.errors import EmptyResult, SerializationError
from omicron.dal.influx.flux import Flux
from omicron.dal.influx.influxclient import InfluxClient
from omicron.dal.influx.serialize import (
    DataframeDeserializer,
    DataframeSerializer,
    NumpyDeserializer,
    NumpySerializer,
)
from tests import init_test_env

cfg = cfg4py.get_instance()


class SerializerTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        await init_test_env()
        await omicron.init()

        url = cfg.influxdb.url
        token = cfg.influxdb.token
        org = cfg.influxdb.org
        bucket_name = cfg.influxdb.bucket_name

        self.client = InfluxClient(url, token, bucket=bucket_name, org=org)

        return await super().asyncSetUp()

    async def asyncTearDown(self) -> None:
        await omicron.close()
        return await super().asyncTearDown()

    def test_dataframe_deserializer(self):
        data = ",result,table,_time,code,name,close,open\r\n,_result,0,2019-01-01T09:31:00Z,000002.XSHE,国联证券,0.2,0.1\r\n,_result,0,2019-01-01T09:36:00Z,000002.XSHE,国联证券,0.2,0.1\r\n,_result,1,2019-01-01T09:32:00Z,000003.XSHE,上海银行,0.2,0.1\r\n,_result,1,2019-01-01T09:37:00Z,000003.XSHE,上海银行,0.2,0.1\r\n,_result,2,2019-01-01T09:33:00Z,000004.XSHE,中国银行,0.2,0.1\r\n,_result,2,2019-01-01T09:38:00Z,000004.XSHE,中国银行,0.2,0.1\r\n,_result,3,2019-01-01T09:34:00Z,000005.XSHE,中国平安,0.2,0.1\r\n,_result,3,2019-01-01T09:39:00Z,000005.XSHE,中国平安,0.2,0.1\r\n\r\n"

        # default
        des = DataframeDeserializer(
            time_col="_time", usecols=["_time", "code", "name", "close"]
        )
        df = des(data)
        self.assertEqual(8, len(df))
        self.assertEqual(4, len(df.columns))
        self.assertIn("_time", df.columns)
        self.assertEqual(np.dtype("datetime64[ns]"), df["_time"].dtype)

        # it's not ordered
        frames = df["_time"].values
        self.assertTrue(frames[2] < frames[1])

        # keep cols
        des = DataframeDeserializer(
            names=",result,table,frame,code,name,close,open".split(","),
            usecols=["frame", "open", "close"],
        )

        df = des(data)
        self.assertEqual(3, len(df.columns))
        self.assertListEqual(["frame", "open", "close"], df.columns.tolist())
        self.assertEqual(8, len(df))

        # sort by
        des = DataframeDeserializer(
            sort_values="frame",
            names=",result,table,frame,code,name,close,open".split(","),
            usecols=["frame", "open", "close"],
        )

        df = des(data)
        # it's ordered
        frames = df["frame"].values
        self.assertTrue(frames[2] > frames[1])

        # `data` is bytes array
        data = data.encode("utf-8")
        des = DataframeDeserializer(encoding="utf-8")
        df = des(data)
        self.assertEqual(8, len(df))

        data = ",result,table,_time,code,amount,close,factor,high,low,open,volume\r\n,_result,0,2019-01-05T00:00:00Z,000001.XSHE,100000000,5.15,1.23,5.2,5,5.1,1000000\r\n,_result,0,2019-01-06T00:00:00Z,000001.XSHE,100000000,5.15,1.23,5.2,5,5.1,1000000\r\n\r\n"

        # use string cols and names
        des = DataframeDeserializer(
            time_col="frame",
            names=[
                "_",
                "result",
                "table",
                "frame",
                "code",
                "amount",
                "close",
                "factor",
                "high",
                "low",
                "open",
                "volume",
            ],
            usecols=bars_cols,
            header=0,
        )
        df = des(data)
        self.assertSetEqual(set(bars_cols), set(df.columns))

    def test_numpy_deserializer(self):
        data = ",result,table,_time,code,name,close,open\r\n,_result,0,2019-01-01T09:31:00Z,000002.XSHE,国联证券,0.2,0.1\r\n,_result,0,2019-01-01T09:36:00Z,000002.XSHE,国联证券,0.2,0.1\r\n,_result,1,2019-01-01T09:32:00Z,000003.XSHE,上海银行,0.2,0.1\r\n,_result,1,2019-01-01T09:37:00Z,000003.XSHE,上海银行,0.2,0.1\r\n,_result,2,2019-01-01T09:33:00Z,000004.XSHE,中国银行,0.2,0.1\r\n,_result,2,2019-01-01T09:38:00Z,000004.XSHE,中国银行,0.2,0.1\r\n,_result,3,2019-01-01T09:34:00Z,000005.XSHE,中国平安,0.2,0.1\r\n,_result,3,2019-01-01T09:39:00Z,000005.XSHE,中国平安,0.2,0.1\r\n\r\n"

        dtype = [
            ("frame", "datetime64[s]"),
            ("code", "<U12"),
            ("name", "<U4"),
            ("close", "<f4"),
            ("open", "<f4"),
        ]

        # input is already string
        deserializer = NumpyDeserializer(
            dtype, use_cols=[3, 4, 5, 6, 7], skip_rows=1, encoding=None
        )
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

        # input is bytes array
        des = NumpyDeserializer(dtype, use_cols=[3, 4, 5, 6, 7], skip_rows=1)
        actual = des(data.encode("utf-8"))

        self.assertEqual(str(exp), str(actual))

        # test sort_values
        des = NumpyDeserializer(
            dtype, use_cols=[3, 4, 5, 6, 7], skip_rows=1, sort_values="frame"
        )
        actual = des(data)
        exp = np.array(
            [
                ("2019-01-01T09:31:00", "000002.XSHE", "国联证券", 0.2, 0.1),
                ("2019-01-01T09:32:00", "000003.XSHE", "上海银行", 0.2, 0.1),
                ("2019-01-01T09:33:00", "000004.XSHE", "中国银行", 0.2, 0.1),
                ("2019-01-01T09:34:00", "000005.XSHE", "中国平安", 0.2, 0.1),
                ("2019-01-01T09:36:00", "000002.XSHE", "国联证券", 0.2, 0.1),
                ("2019-01-01T09:37:00", "000003.XSHE", "上海银行", 0.2, 0.1),
                ("2019-01-01T09:38:00", "000004.XSHE", "中国银行", 0.2, 0.1),
                ("2019-01-01T09:39:00", "000005.XSHE", "中国平安", 0.2, 0.1),
            ],
            dtype=dtype,
        )
        self.assertEqual(str(exp), str(actual))

        # specify use_cols by column name, and make order change
        data = ",result,table,_time,code,name,close,open\r\n,_result,0,2019-01-01T09:31:00Z,000002.XSHE,国联证券,0.2,0.1\r\n,_result,0,2019-01-01T09:36:00Z,000002.XSHE,国联证券,0.2,0.1\r\n"

        dtype = [
            ("frame", "datetime64[s]"),
            ("close", "<f4"),
            ("open", "<f4"),
            ("code", "<U12"),
            ("name", "<U4"),
        ]

        des = NumpyDeserializer(
            dtype, use_cols=["_time", "close", "open", "code", "name"], skip_rows=1
        )

        exp = np.array(
            [
                ("2019-01-01T09:31:00", 0.2, 0.1, "000002.XSHE", "国联证券"),
                ("2019-01-01T09:36:00", 0.2, 0.1, "000002.XSHE", "国联证券"),
            ],
            dtype=dtype,
        )

        actual = des(data)
        self.assertEqual(str(exp), str(actual))

        # wrong headers
        data = "#this is a test\n,result,table,_time,code,name,close,open\r\n,_result,0,2019-01-01T09:31:00Z,000002.XSHE,国联证券,0.2,0.1\r\n,_result,0,2019-01-01T09:36:00Z,000002.XSHE,国联证券,0.2,0.1\r\n"
        des = NumpyDeserializer(
            dtype,
            use_cols=["_time", "close", "open", "code", "name"],
            skip_rows=1,
            header_line=2,
        )
        with self.assertRaises(SerializationError):
            des(data)

        # no header line
        data = ",_result,0,2019-01-01T09:31:00Z,000002.XSHE,国联证券,0.2,0.1\r\n,_result,0,2019-01-01T09:36:00Z,000002.XSHE,国联证券,0.2,0.1\r\n"

        with self.assertRaises(AssertionError):
            des = NumpyDeserializer(
                bars_dtype,
                use_cols=["_time"] + bars_cols[1:],
                skip_rows=None,
                converters={"_time": lambda x: arrow.get(x).date()},
                header_line=None,
            )

        with self.assertRaises(AssertionError):
            des = NumpyDeserializer(
                bars_dtype,
                parse_date=None,
                use_cols=["_time"] + bars_cols[1:],
                converters={"_time": lambda x: arrow.get(x).date()},
                header_line=None,
            )

        # content is empty
        data = "\r\n"
        des = NumpyDeserializer(bars_dtype, use_cols=["_time"] + bars_cols[1:])
        self.assertEqual(0, des(data).size)

        data = "\r\nabceefg\r\n"
        with self.assertRaises(SerializationError):
            des = NumpyDeserializer(bars_dtype, use_cols=["_time"] + bars_cols[1:])
            des(data)

    async def test_dataframe_serializer(self):
        df = pd.DataFrame(
            [("000001.XSHE", 1, 2, "payh"), ("000002.XSHE", 2, 3, "jsyh")],
            columns=["code", "a", "b", "name"],
            index=[datetime.datetime(1990, 1, 1), datetime.datetime(1990, 1, 2)],
        )
        serializer = DataframeSerializer(df, "test", tag_keys="code")

        actual = []
        for lp in serializer.serialize(1):
            actual.append(lp)

        expected = [
            'test,code=000001.XSHE a=1i,b=2i,name="payh" 631152000',
            'test,code=000002.XSHE a=2i,b=3i,name="jsyh" 631238400',
        ]
        self.assertListEqual(expected, actual)

        # contains nan
        df = pd.DataFrame(
            [
                ("000001.XSHE", np.nan, 2, "payh", datetime.datetime(1990, 1, 1)),
                ("000002.XSHE", 2, 3, "jsyh", datetime.datetime(1990, 1, 2)),
            ],
            columns=["code", "a", "b", "name", "frame"],
        )
        serializer = DataframeSerializer(df, "test", tag_keys="code", time_key="frame")
        expected = 'test,code=000001.XSHE b=2i,name="payh" 631152000\ntest,code=000002.XSHE a=2.0,b=3i,name="jsyh" 631238400'
        for actual in serializer.serialize(len(df)):
            self.assertEqual(expected, actual)

    async def test_numpy_serializer(self):
        dtype = [
            ("frame", "datetime64[s]"),
            ("code", "<U12"),
            ("name", "<U4"),
            ("close", "<f4"),
            ("open", "<f4"),
        ]

        data = np.array(
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

        serializer = NumpySerializer(
            data, "test", "frame", ["code", "name"], precisions={"open": 1, "close": 1}
        )

        actual = []
        for lines in serializer.serialize(3):
            actual.append(lines)

        exp = [
            "test,code=000002.XSHE,name=国联证券 close=0.2,open=0.1 1546335060\ntest,code=000002.XSHE,name=国联证券 close=0.2,open=0.1 1546335360\ntest,code=000003.XSHE,name=上海银行 close=0.2,open=0.1 1546335120",
            "test,code=000003.XSHE,name=上海银行 close=0.2,open=0.1 1546335420\ntest,code=000004.XSHE,name=中国银行 close=0.2,open=0.1 1546335180\ntest,code=000004.XSHE,name=中国银行 close=0.2,open=0.1 1546335480",
            "test,code=000005.XSHE,name=中国平安 close=0.2,open=0.1 1546335240\ntest,code=000005.XSHE,name=中国平安 close=0.2,open=0.1 1546335540",
        ]
        self.assertEqual(exp, actual)

        # ms write precision
        serializer = NumpySerializer(
            data,
            "test",
            "frame",
            ["code", "name"],
            precisions={"open": 1, "close": 1},
            time_precision="ms",
        )
        for actual in serializer.serialize(8):
            exp = "test,code=000002.XSHE,name=国联证券 close=0.2,open=0.1 1546335060000\n"
            self.assertLessEqual(exp, actual[:65])

        # no tag_keys
        bars = np.array(
            [
                (
                    datetime.date(2019, 1, 1),
                    5.1,
                    5.2,
                    5.0,
                    5.15,
                    1000000,
                    100000000,
                    1.23,
                )
            ],
            dtype=bars_dtype,
        )

        serializer = NumpySerializer(
            bars,
            "test",
            "frame",
            precisions={
                "open": 2,
                "close": 2,
                "high": 2,
                "low": 2,
                "volume": 1,
                "amount": 1,
                "factor": 3,
            },
        )

        exp = "test amount=1e+08,close=5.2,factor=1.23,high=5.2,low=5.0,open=5.1,volume=1e+06 1546300800"

        for lines in serializer.serialize(len(bars)):
            self.assertEqual(exp, lines)

        # frame is datetime.date
        dtype = bars_dtype.descr.copy()
        dtype[0] = ("frame", "O")
        bars = bars.astype(dtype)
        serializer = NumpySerializer(bars, "test", "frame")
        exp = "test amount=100000000.0,close=5.150000095367432,factor=1.2300000190734863,high=5.199999809265137,low=5.0,open=5.099999904632568,volume=1000000.0 1546300800"
        for lines in serializer.serialize(len(bars)):
            self.assertEqual(exp, lines)

        # no precisions
        serializer = NumpySerializer(bars, "test", "frame")

        exp = "test amount=100000000.0,close=5.150000095367432,factor=1.2300000190734863,high=5.199999809265137,low=5.0,open=5.099999904632568,volume=1000000.0 1546300800"

        for lines in serializer.serialize(len(bars)):
            print(lines)
            self.assertEqual(exp, lines)

        # no tm_key
        serializer = NumpySerializer(bars, "test")

        for actual in serializer.serialize(len(bars)):
            exp = 'test amount=100000000.0,close=5.150000095367432,factor=1.2300000190734863,frame="2019-01-01 00:00:00",high=5.199999809265137,low=5.0,open=5.099999904632568,volume=1000000.0'
            self.assertEqual(exp, actual)

        # global keys
        serializer = NumpySerializer(
            bars, "test", "frame", global_tags={"code": "000002.XSHE"}
        )

        for actual in serializer.serialize(len(bars)):
            exp = "test,code=000002.XSHE amount=100000000.0,close=5.150000095367432,factor=1.2300000190734863,high=5.199999809265137,low=5.0,open=5.099999904632568,volume=1000000.0 1546300800"
            self.assertEqual(exp, actual)
