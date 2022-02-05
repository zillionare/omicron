import datetime
import unittest
from tkinter import S

import cfg4py
import numpy as np
import pandas as pd
from coretypes import stock_bars_dtype
from influxdb_client.client.write_api import PointSettings

import omicron
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

    def test_dataframe_deserializer(self):
        data = ",result,table,_time,code,name,close,open\r\n,_result,0,2019-01-01T09:31:00Z,000002.XSHE,国联证券,0.2,0.1\r\n,_result,0,2019-01-01T09:36:00Z,000002.XSHE,国联证券,0.2,0.1\r\n,_result,1,2019-01-01T09:32:00Z,000003.XSHE,上海银行,0.2,0.1\r\n,_result,1,2019-01-01T09:37:00Z,000003.XSHE,上海银行,0.2,0.1\r\n,_result,2,2019-01-01T09:33:00Z,000004.XSHE,中国银行,0.2,0.1\r\n,_result,2,2019-01-01T09:38:00Z,000004.XSHE,中国银行,0.2,0.1\r\n,_result,3,2019-01-01T09:34:00Z,000005.XSHE,中国平安,0.2,0.1\r\n,_result,3,2019-01-01T09:39:00Z,000005.XSHE,中国平安,0.2,0.1\r\n\r\n"

        # default
        des = DataframeDeserializer(
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
        des = DataframeDeserializer(
            names=["frame", "open", "close"], usecols=[3, 6, 7], parse_dates="frame"
        )
        df = des(data)
        self.assertEqual(3, len(df.columns))
        self.assertListEqual(["frame", "open", "close"], df.columns.tolist())
        self.assertEqual(8, len(df))

        # sort by

        des = DataframeDeserializer(
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
        des = DataframeDeserializer(encoding="utf-8")
        df = des(data)
        self.assertEqual(8, len(df))

    def test_numpy_deserializer(self):
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
            data,
            "test",
            "frame",
            ["code", "name"],
            precisions={"open": 1, "close": 1},
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
            dtype=stock_bars_dtype,
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

        exp = "test amount=1e+08,close=5.2,factor=1.23,high=5.2,low=5.0,open=5.1,volume=1e+06 1546272000"

        for lines in serializer.serialize(len(bars)):
            self.assertEqual(exp, lines)

        # no precisions
        serializer = NumpySerializer(bars, "test", "frame")

        exp = "test amount=1e+08,close=5.15,factor=1.23,high=5.2,low=5.0,open=5.1,volume=1e+06 1546272000"

        for lines in serializer.serialize(len(bars)):
            print(lines)
            self.assertEqual(exp, lines)

        # no tm_key
        serializer = NumpySerializer(bars, "test")

        for actual in serializer.serialize(len(bars)):
            exp = 'test amount=1e+08,close=5.15,factor=1.23,frame="2019-01-01",high=5.2,low=5.0,open=5.1,volume=1e+06'
            self.assertEqual(exp, actual)

        # global keys
        serializer = NumpySerializer(
            bars, "test", "frame", global_tags={"code": "000002.XSHE"}
        )

        for actual in serializer.serialize(len(bars)):
            exp = "test,code=000002.XSHE amount=1e+08,close=5.15,factor=1.23,high=5.2,low=5.0,open=5.1,volume=1e+06 1546272000"
            self.assertEqual(exp, actual)
