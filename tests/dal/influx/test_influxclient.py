import datetime
import functools
import time
import unittest

import arrow
import cfg4py
import numpy as np
from coretypes import stock_bars_dtype

import omicron
from omicron.dal.influx.flux import Flux
from omicron.dal.influx.influxclient import InfluxClient
from omicron.dal.influx.serialize import unserialize
from tests import init_test_env

cfg = cfg4py.get_instance()


class InfluxClientTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        await init_test_env()
        await omicron.init()

        url = cfg.influxdb.url
        token = cfg.influxdb.token
        org = cfg.influxdb.org
        bucket_name = cfg.influxdb.bucket_name

        self.client = InfluxClient(
            url,
            token,
            bucket=bucket_name,
            org=org,
            debug=True,
        )

        return await super().asyncSetUp()

    async def test_to_line_protocol(self):
        measurement = "stock_bars_1d"
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

        client = InfluxClient("", "", "")

        tags = "code=000001.XSHE"

        actual = client.nparray_to_line_protocol(
            measurement, bars, tags, tm_key="frame"
        )
        print(actual)

        actual = client.nparray_to_line_protocol(
            measurement,
            bars,
            tags,
            tm_key="frame",
            formatters={
                "open": "{:.02f}",
                "close": "{:.02f}",
                "high": "{:.02f}",
                "low": "{:.02f}",
                "amount": "{:.02f}",
                "volume": "{:.02f}",
                "factor": "{:.04f}",
            },
        )

        exp = "stock_bars_1d,code=000001.XSHE open=5.10,high=5.20,low=5.00,close=5.15,volume=1000000.00,amount=100000000.00,factor=1.2300 1546300800"
        self.assertEqual(exp, actual)

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
                ),
                (
                    datetime.date(2019, 1, 2),
                    5.1,
                    5.2,
                    5.0,
                    5.15,
                    1000000,
                    100000000,
                    1.23,
                ),
            ],
            dtype=stock_bars_dtype,
        )

        actual = client.nparray_to_line_protocol(
            measurement,
            bars,
            tags,
            tm_key="frame",
            formatters={
                "open": "{:.02f}",
                "close": "{:.02f}",
                "high": "{:.02f}",
                "low": "{:.02f}",
                "amount": "{:.02f}",
                "volume": "{:.02f}",
                "factor": "{:.04f}",
            },
        )

        exp = "stock_bars_1d,code=000001.XSHE open=5.10,high=5.20,low=5.00,close=5.15,volume=1000000.00,amount=100000000.00,factor=1.2300 1546300800\n"
        exp += "stock_bars_1d,code=000001.XSHE open=5.10,high=5.20,low=5.00,close=5.15,volume=1000000.00,amount=100000000.00,factor=1.2300 1546387200"

        self.assertEqual(exp, actual)

        # without tm_key
        actual = client.nparray_to_line_protocol(
            measurement,
            bars,
            tags,
            formatters={
                "open": "{:.02f}",
                "close": "{:.02f}",
                "high": "{:.02f}",
                "low": "{:.02f}",
                "amount": "{:.02f}",
                "volume": "{:.02f}",
                "factor": "{:.04f}",
            },
        )

        exp = "stock_bars_1d,code=000001.XSHE frame=2019-01-01,open=5.10,high=5.20,low=5.00,close=5.15,volume=1000000.00,amount=100000000.00,factor=1.2300 \nstock_bars_1d,code=000001.XSHE frame=2019-01-02,open=5.10,high=5.20,low=5.00,close=5.15,volume=1000000.00,amount=100000000.00,factor=1.2300 "
        self.assertEqual(exp, actual)

    async def test_write(self):
        """
        this also test drop_measurement, query
        """
        measurement = "stock_bars_1d"

        await self.client.drop_measurement("stock_bars_1d")

        bars = np.array(
            [
                (
                    datetime.date(2019, 1, 5),
                    5.1,
                    5.2,
                    5.0,
                    5.15,
                    1000000,
                    100000000,
                    1.23,
                ),
                (
                    datetime.date(2019, 1, 6),
                    5.1,
                    5.2,
                    5.0,
                    5.15,
                    1000000,
                    100000000,
                    1.23,
                ),
            ],
            dtype=stock_bars_dtype,
        )

        data = self.client.nparray_to_line_protocol(
            measurement,
            bars,
            "code=000001.XSHE",
            tm_key="frame",
            formatters={
                "open": "{:.02f}",
                "close": "{:.02f}",
                "high": "{:.02f}",
                "low": "{:.02f}",
                "amount": "{:.02f}",
                "volume": "{:.02f}",
                "factor": "{:.04f}",
            },
        )
        await self.client.write(data)

        data = self.client.nparray_to_line_protocol(
            measurement,
            bars,
            "code=000002.XSHE",
            tm_key="frame",
            formatters={
                "open": "{:.02f}",
                "close": "{:.02f}",
                "high": "{:.02f}",
                "low": "{:.02f}",
                "amount": "{:.02f}",
                "volume": "{:.02f}",
                "factor": "{:.04f}",
            },
        )
        await self.client.write(data)

        query = (
            Flux()
            .bucket(self.client._bucket)
            .measurement(measurement)
            .tags({"code": "000001.XSHE"})
            .range(datetime.date(2019, 1, 5), datetime.date(2019, 1, 6))
            .pivot()
            .keep(
                columns=[
                    "code",
                    "_time",
                    "open",
                    "close",
                    "high",
                    "low",
                    "amount",
                    "volume",
                    "factor",
                ]
            )
        )

        result = await self.client.query(query)
        actual = result.decode("utf-8")

        print(actual)
        exp = ",result,table,_time,code,amount,close,factor,high,low,open,volume\r\n,_result,0,2019-01-05T00:00:00Z,000001.XSHE,100000000,5.15,1.23,5.2,5,5.1,1000000\r\n,_result,0,2019-01-06T00:00:00Z,000001.XSHE,100000000,5.15,1.23,5.2,5,5.1,1000000\r\n\r\n"
        self.assertEqual(exp, actual)

    async def test_query(self):
        measurement = "unitest_test_query"

        mock_data = []
        start = arrow.get("2019-01-01 09:30:00")
        names = ["平安银行", "国联证券", "上海银行", "中国银行", "中国平安"]
        for i in range(10000):
            mock_data.append(
                (
                    start.shift(minutes=i).datetime,
                    0.1,
                    0.2,
                    f"00000{i%5+1}.XSHE",
                    names[i % 5],
                )
            )

        mock_data = np.array(
            mock_data,
            dtype=[
                ("frame", "datetime64[ns]"),
                ("open", "float32"),
                ("close", "float32"),
                ("code", "O"),
                ("name", "O"),
            ],
        )

        lp = self.client.nparray_to_line_protocol(
            measurement,
            mock_data,
            tags={"name", "code"},
            tm_key="frame",
            formatters={"open": "{:.02f}", "close": "{:.02f}"},
        )

        # unitest_test_query,code=000001.XSHE,name=平安银行 close=0.20,open=0.10 1546335000
        await self.client.write(lp)

        # query all from measurement
        flux = (
            Flux()
            .measurement(measurement)
            .range(Flux.EPOCH_START, arrow.now().datetime)
            .bucket(self.client._bucket)
            .pivot()
            .keep(["open", "close", "code", "name"])
        )

        t0 = time.time()
        data = await self.client.query(flux)
        actual = unserialize(
            data, keep_cols=["_time", "open", "code", "name"], sort_by="_time"
        )
        print("query 1万行并反序列化为dataframe cost", time.time() - t0)
        self.assertEqual(actual.loc[0]["name"], "平安银行")
        self.assertEqual(actual.loc[0]["open"], 0.1)
        self.assertEqual(actual.loc[0]["code"], "000001.XSHE")
        self.assertEqual(
            actual.loc[0]["_time"], datetime.datetime(2019, 1, 1, 9, 30, 0)
        )

        # query by code
        flux = (
            Flux()
            .measurement(measurement)
            .tags({"code": "000001.XSHE"})
            .range(Flux.EPOCH_START, datetime.datetime(2019, 1, 1, 9, 30))
            .bucket(self.client._bucket)
            .pivot()
            .keep(["open", "close", "code", "name"])
        )

        partial = functools.partial(
            unserialize,
            keep_cols=["frame", "open", "code", "name"],
            sort_by="_time",
            rename_time_field="frame",
        )
        actual = await self.client.query(flux, partial)
