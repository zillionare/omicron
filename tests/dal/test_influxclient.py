import datetime
import unittest

import cfg4py
import numpy as np
from coretypes import stock_bars_dtype

import omicron
from omicron.dal.flux import Flux
from omicron.dal.influxclient import InfluxClient
from tests import init_test_env

cfg = cfg4py.get_instance()


class InfluxClientTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        await init_test_env()
        await omicron.init()
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
        # this will also test query
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

        url = cfg.influxdb.url
        token = cfg.influxdb.token
        org = cfg.influxdb.org
        bucket_name = cfg.influxdb.bucket_name
        measurement = "stock_bars_1d"

        client = InfluxClient(
            url,
            token,
            bucket=bucket_name,
            measurement="stock_bars_1d",
            org=org,
            debug=True,
        )

        data = client.nparray_to_line_protocol(
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
        await client.write(data)

        data = client.nparray_to_line_protocol(
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
        await client.write(data)

        query = (
            Flux()
            .bucket(bucket_name)
            .measurement(measurement)
            .tags({"code": "000001.XSHE"})
            .range(datetime.date(2019, 1, 1), datetime.date(2019, 1, 1))
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

        result = await client.query(query)
        actual = result.decode("utf-8")

        print(actual)
        exp = ",result,table,_time,code,amount,close,factor,high,low,open,volume\r\n,_result,0,2019-01-01T00:00:00Z,000001.XSHE,100000000,5.15,1.23,5.2,5,5.1,1000000\r\n\r\n"
        self.assertEqual(exp, actual)
