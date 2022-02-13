import datetime
import unittest
from unittest import mock

import arrow
import cfg4py
import numpy as np
import pandas as pd
from coretypes import bars_cols, bars_dtype

import omicron
from omicron.core.errors import (
    InfluxDBQueryError,
    InfluxDBWriteError,
    InfluxDeleteError,
)
from omicron.dal.influx.flux import Flux
from omicron.dal.influx.influxclient import InfluxClient
from omicron.dal.influx.serialize import (
    DataframeDeserializer,
    NumpyDeserializer,
    NumpySerializer,
)
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

        self.client = InfluxClient(url, token, bucket=bucket_name, org=org)

        data = mock_data_for_influx(100)
        serializer = NumpySerializer(
            data, "ut_test_query", time_key="frame", tag_keys=["name", "code"]
        )
        # ut_test_query,code=000001.XSHE,name=平安银行 close=0.20,open=0.10 1546335000
        for lines in serializer.serialize(len(data)):
            await self.client.write(lines)

        return await super().asyncSetUp()

    async def asyncTearDown(self) -> None:
        try:
            await self.client.drop_measurement("ut_test_query")
        except Exception:
            pass

        return await super().asyncTearDown()

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
            dtype=bars_dtype,
        )

        serializer = NumpySerializer(
            bars, measurement, global_tags={"code": "000001.XSHE"}, time_key="frame"
        )

        for data in serializer.serialize(len(bars)):
            await self.client.write(data)

        serializer = NumpySerializer(
            bars, measurement, global_tags={"code": "000002.XSHE"}, time_key="frame"
        )

        for data in serializer.serialize(len(bars)):
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

        # reset measurement and test gzip
        await self.client.drop_measurement("stock_bars_1d")

        url = cfg.influxdb.url
        token = cfg.influxdb.token
        org = cfg.influxdb.org
        bucket_name = cfg.influxdb.bucket_name

        client = InfluxClient(
            url, token, bucket=bucket_name, org=org, enable_compress=True
        )
        for data in serializer.serialize(len(bars)):
            await client.write(data)

        query = (
            Flux()
            .bucket(self.client._bucket)
            .measurement(measurement)
            .tags({"code": "000002.XSHE"})
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
        exp = ",result,table,_time,code,amount,close,factor,high,low,open,volume\r\n,_result,0,2019-01-05T00:00:00Z,000002.XSHE,100000000,5.15,1.23,5.2,5,5.1,1000000\r\n,_result,0,2019-01-06T00:00:00Z,000002.XSHE,100000000,5.15,1.23,5.2,5,5.1,1000000\r\n\r\n"
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

        ds = DataframeDeserializer(
            sort_values="_time",
            usecols=["_time", "open", "code", "name"],
            parse_dates="_time",
            engine="c",
        )
        actual = ds(data)

        actual = actual.to_records(index=False)

        self.assertEqual(actual[0]["name"], "平安银行")
        self.assertAlmostEqual(actual[0]["open"], 0.1)
        self.assertEqual(actual[0]["code"], "000001.XSHE")
        self.assertEqual(actual[0]["_time"], np.datetime64("2019-01-01T09:30:00"))

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

        # given deserializer
        ds = DataframeDeserializer()
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

        ds = DataframeDeserializer()
        actual = await self.client.query(flux, ds)
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
        self.assertEqual(str(actual.loc[0]["_time"]), "2019-01-01T09:30:00Z")

        # check exceptions
        with mock.patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.return_value.__aenter__.status = 400
            mock_post.return_value.__aenter__.json = mock.Mock()
            mock_post.return_value.__aenter__.json.return_value = {
                "code": -1,
                "message": "mockerror",
            }
            with self.assertRaises(InfluxDBQueryError):
                await self.client.query(flux, ds)

    async def test_delete(self):
        cols = ["open", "close", "code", "name"]

        q = (
            Flux()
            .bucket(self.client._bucket)
            .measurement("ut_test_query")
            .range(Flux.EPOCH_START, arrow.now().datetime)
            .keep(cols)
        )

        ds = DataframeDeserializer()
        recs = await self.client.query(q, ds)

        self.assertEqual(len(recs), 100)
        self.assertEqual(recs.loc[0]["name"], "平安银行")

        # delete by tags
        await self.client.delete(
            "ut_test_query", arrow.now().naive, {"code": "000001.XSHE"}
        )

        recs = await self.client.query(q, ds)
        self.assertEqual(len(recs), 80)

        # delete by range
        await self.client.delete(
            "ut_test_query", arrow.get("2019-01-01 09:30:00").shift(minutes=90).naive
        )
        recs = await self.client.query(q, ds)
        self.assertEqual(len(recs), 8)

        # test dop measurement
        await self.client.drop_measurement("ut_test_query")

        with mock.patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.return_value.__aenter__.status = 400
            mock_post.return_value.__aenter__.json = mock.Mock()
            mock_post.return_value.__aenter__.json.return_value = {
                "code": -1,
                "message": "mockerror",
            }
            with self.assertRaises(InfluxDeleteError):
                await self.client.delete("ut_test_query", arrow.now().naive)

    async def test_save(self):
        measurement = "ut_test_influxclient_save"
        await self.client.drop_measurement(measurement)
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
            dtype=bars_dtype,
        )

        # save np.array
        await self.client.save(
            bars, measurement, time_key="frame", global_tags={"code": "000001.XSHE"}
        )

        start, end = datetime.datetime(2019, 1, 5), datetime.datetime(2019, 1, 6)
        query = (
            Flux()
            .measurement(measurement)
            .bucket(self.client._bucket)
            .range(start, end)
            .keep(bars_cols)
        )

        actual = await self.client.query(query)
        expected = b",result,table,_time,amount,close,factor,high,low,open,volume\r\n,_result,0,2019-01-05T00:00:00Z,100000000,5.15,1.23,5.2,5,5.1,1000000\r\n,_result,0,2019-01-06T00:00:00Z,100000000,5.15,1.23,5.2,5,5.1,1000000\r\n\r\n"

        self.assertEqual(expected, actual)

        # save np.array with chunk_size == -1
        await self.client.save(
            bars,
            measurement,
            time_key="frame",
            global_tags={"code": "000001.XSHE"},
            chunk_size=-1,
        )

        await self.client.drop_measurement(measurement)
        df = pd.DataFrame(bars, columns=bars_cols)

        ## save pd.DataFrame
        await self.client.save(
            df, measurement, time_key="frame", global_tags={"code": "000001.XSHE"}
        )

        use_cols = bars_cols.copy()
        use_cols[0] = "_time"
        ds = NumpyDeserializer(bars_dtype, "frame", use_cols=use_cols)
        actual = await self.client.query(query, ds)
        for col in ["close", "factor", "high", "low", "open", "volume"]:
            np.testing.assert_array_almost_equal(bars[col], actual[col], decimal=2)

        np.testing.assert_array_equal(
            bars["frame"], [x.date() for x in actual["frame"]]
        )

        ## save pd.DataFrame with chunk_size == -1
        await self.client.save(
            df,
            measurement,
            time_key="frame",
            global_tags={"code": "000001.XSHE"},
            chunk_size=-1,
        )

        with self.assertRaises(InfluxDBWriteError):
            with mock.patch("aiohttp.ClientSession.post") as mock_post:
                mock_post.return_value.__aenter__.status = 400
                mock_post.return_value.__aenter__.json = mock.Mock()
                mock_post.return_value.__aenter__.json.return_value = {
                    "code": -1,
                    "message": "mockerror",
                }

                await self.client.save(
                    df,
                    measurement,
                    time_key="frame",
                    global_tags={"code": "000001.XSHE"},
                )