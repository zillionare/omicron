import datetime
import functools
import time
import unittest
from curses import def_shell_mode

import arrow
import cfg4py
import numpy as np
from coretypes import stock_bars_dtype
from sklearn.metrics import mean_squared_error

import omicron
from omicron.dal.influx.flux import Flux
from omicron.dal.influx.influxclient import InfluxClient
from omicron.dal.influx.serialize import DataFrameDeserializer
from tests import init_test_env
from tests.dal.influx import mock_data_for_influx

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

        data = mock_data_for_influx(100)
        lp = self.client.nparray_to_line_protocol(
            "ut_test_query",
            data,
            tags={"name", "code"},
            tm_key="frame",
            formatters={"open": "{:.02f}", "close": "{:.02f}"},
        )

        # ut_test_query,code=000001.XSHE,name=平安银行 close=0.20,open=0.10 1546335000
        await self.client.write(lp)

        return await super().asyncSetUp()

    async def asyncTearDown(self) -> None:
        try:
            await self.client.drop_measurement("ut_test_query")
        except Exception:
            pass

        return await super().asyncTearDown()

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

        exp = "stock_bars_1d,code=000001.XSHE amount=100000000.00,close=5.15,factor=1.2300,high=5.20,low=5.00,open=5.10,volume=1000000.00 1546300800"
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

        exp = "stock_bars_1d,code=000001.XSHE amount=100000000.00,close=5.15,factor=1.2300,high=5.20,low=5.00,open=5.10,volume=1000000.00 1546300800\n"
        exp += "stock_bars_1d,code=000001.XSHE amount=100000000.00,close=5.15,factor=1.2300,high=5.20,low=5.00,open=5.10,volume=1000000.00 1546387200"

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

        exp = "stock_bars_1d,code=000001.XSHE amount=100000000.00,close=5.15,factor=1.2300,frame=2019-01-01,high=5.20,low=5.00,open=5.10,volume=1000000.00 \nstock_bars_1d,code=000001.XSHE amount=100000000.00,close=5.15,factor=1.2300,frame=2019-01-02,high=5.20,low=5.00,open=5.10,volume=1000000.00 "
        self.assertEqual(exp, actual)

    async def test_write(self):
        """
        this also test drop_measurement, query
        """
        measurement = "stock_bars_1d"
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
        measurement = "ut_test_query"
        # query all from measurement
        flux = (
            Flux()
            .measurement(measurement)
            .range(Flux.EPOCH_START, arrow.now().datetime)
            .bucket(self.client._bucket)
            .pivot()
            .keep(["open", "close", "code", "name"])
        )

        data = await self.client.query(flux)
        ds = DataFrameDeserializer(
            sort_values="frame", usecols=["frame", "open", "code", "name"]
        )
        # , keep_cols=["_time", "open", "code", "name"], sort_by="_time"
        actual = ds(data)

        actual = actual.to_records(index=False)

        self.assertEqual(actual[0]["name"], "平安银行")
        self.assertAlmostEqual(actual[0]["open"], 0.1)
        self.assertEqual(actual[0]["code"], "000001.XSHE")
        self.assertEqual(actual[0]["_time"], datetime.datetime(2019, 1, 1, 9, 30, 0))

        # query by code, and test right close range
        flux = (
            Flux()
            .measurement(measurement)
            .tags({"code": "000001.XSHE"})
            .range(Flux.EPOCH_START, datetime.datetime(2019, 1, 1, 9, 30))
            .bucket(self.client._bucket)
            .pivot()
            .keep(["open", "close", "code", "name"])
        )

        # given unserializer
        ds = DataFrameDeserializer()
        actual = await self.client.query(flux, ds)
        self.assertEqual(actual.loc[0]["name"], "平安银行")
        self.assertEqual(1, len(actual))
        self.assertEqual(actual.loc[0]["open"], 0.1)

        # query by two different tags and involve contains operator
        flux = (
            Flux()
            .measurement(measurement)
            .tags({"code": "000001.XSHE", "name": ["平安银行", "中国银行"]})
            .range(Flux.EPOCH_START, datetime.datetime(2019, 1, 1, 9, 35))
            .bucket(self.client._bucket)
            .pivot()
            .keep(["open", "close", "code", "name"])
        )

        ds = DataFrameDeserializer()
        actual = await self.client.query(flux, def_shell_mode())
        self.assertSetEqual(set(["平安银行", "中国银行"]), set(actual["name"]))
        self.assertEqual(3, len(actual))

        # query tags with with array which contains only one value
        flux = (
            Flux()
            .measurement(measurement)
            .tags({"code": ["000001.XSHE"]})
            .bucket(self.client._bucket)
            .range(Flux.EPOCH_START, datetime.datetime(2019, 1, 1, 9, 30))
            .pivot()
            .keep(["open", "close", "code", "name"])
        )

        actual = await self.client.query(flux, ds)
        self.assertEqual(1, len(actual))
        self.assertEqual(actual.loc[0]["name"], "平安银行")
        self.assertEqual(actual.loc[0]["_time"], datetime.datetime(2019, 1, 1, 9, 30))

    async def test_delete(self):
        cols = ["open", "close", "code", "name"]

        q = Flux().drop_measurement("unittest_test_query").bucket(self.client._bucket)
        q = (
            Flux()
            .bucket(self.client._bucket)
            .measurement("ut_test_query")
            .range(Flux.EPOCH_START, arrow.now().datetime)
            .keep(cols)
        )

        ds = DataFrameDeserializer()
        recs = await self.client.query(q, ds)

        self.assertEqual(len(recs), 10)
        self.assertEqual(recs.loc[0]["name"], "平安银行")

        # delete by tags
        await self.client.delete(
            "ut_test_query", arrow.now().naive, {"code": "000001.XSHE"}
        )

        recs = await self.client.query(q, ds)
        self.assertEqual(len(recs), 8)
