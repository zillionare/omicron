import datetime
import unittest
import uuid
from unittest import mock

import arrow
import cfg4py
import ciso8601
import numpy as np
import pandas as pd
from coretypes import bars_cols, bars_dtype

import omicron
from omicron.dal.influx.errors import *
from omicron.dal.influx.flux import Flux
from omicron.dal.influx.influxclient import InfluxClient
from omicron.dal.influx.serialize import (
    DataframeDeserializer,
    NumpyDeserializer,
    NumpySerializer,
)
from omicron.models.stock import Stock
from tests import MockException, assert_bars_equal, init_test_env
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

        await omicron.close()
        return await super().asyncTearDown()

    async def test_write(self):
        """
        this also test drop_measurement, query
        """
        measurement = "stock_bars_1d"

        await self.client.drop_measurement(measurement)

        code = "000001.XSHE"
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

        start, end = bars["frame"][0].item(), bars["frame"][1].item()

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

        keep_cols = ["_time"] + bars_cols[1:]
        flux = (
            Flux()
            .bucket(self.client._bucket)
            .measurement(measurement)
            .range(start, end)
            .tags({"code": code})
        )

        des = NumpyDeserializer(
            bars_dtype,
            # sort at server side
            # sort_values="frame",
            encoding="utf-8",
            skip_rows=1,
            use_cols=keep_cols,
            converters={"_time": lambda x: ciso8601.parse_datetime(x).date()},
            parse_date=None,
        )

        actual = await self.client.query(flux, des)
        assert_bars_equal(bars, actual)

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
        )

        actual = await self.client.query(query, des)
        assert_bars_equal(bars, actual)

    async def test_query(self):
        measurement = "ut_test_query"
        # query all from measurement
        flux = (
            Flux()
            .measurement(measurement)
            .range(Flux.EPOCH_START, arrow.now().datetime)
            .bucket(self.client._bucket)
            .pivot()
        )

        data = await self.client.query(flux)

        ds = DataframeDeserializer(
            sort_values="_time",
            usecols=["_time", "open", "code", "name"],
            time_col="_time",
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
        )

        # given deserializer
        actual = await self.client.query(flux, ds)
        self.assertEqual(actual.loc[0]["name"], "平安银行")
        self.assertEqual(1, len(actual))
        self.assertAlmostEqual(actual.loc[0]["open"], 0.1)

        # query by two different tags and involve contains operator
        flux = (
            Flux()
            .measurement(measurement)
            .tags({"code": "000001.XSHE", "name": ["平安银行", "中国银行"]})
            .range(Flux.EPOCH_START, datetime.datetime(2019, 1, 1, 9, 35))
            .bucket(self.client._bucket)
        )

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
        )

        actual = await self.client.query(flux, ds)
        self.assertEqual(1, len(actual))
        self.assertEqual(actual.loc[0]["name"], "平安银行")
        self.assertEqual(str(actual.loc[0]["_time"]), "2019-01-01 09:30:00")

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

        # deserialize error
        with mock.patch(
            "omicron.dal.influx.serialize.DataframeDeserializer.__call__",
            side_effect=MockException,
        ):
            with self.assertRaises(MockException):
                await self.client.query(flux, ds)

    async def test_delete(self):
        q = (
            Flux()
            .bucket(self.client._bucket)
            .measurement("ut_test_query")
            .range(Flux.EPOCH_START, arrow.now().datetime)
        )

        ds = DataframeDeserializer(
            sort_values="_time",
            usecols=["_time", "open", "code", "name"],
            time_col="_time",
            engine="c",
        )

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
        )

        des = NumpyDeserializer(
            bars_dtype,
            parse_date=None,
            converters={"_time": lambda x: ciso8601.parse_datetime(x).date()},
            use_cols=["_time"] + bars_cols[1:],
        )
        actual = await self.client.query(query, des)

        assert_bars_equal(bars, actual)

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
            [x.item().date() for x in bars["frame"]],
            [x.item().date() for x in actual["frame"]],
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

    async def test_bucket_crud(self):
        name = "ut_test_bucket"
        token = cfg.influxdb.token

        org = cfg.influxdb.org
        client = InfluxClient(cfg.influxdb.url, token, name, org)
        orgs = await client.list_organizations()
        self.assertEqual(len(orgs), 1)
        self.assertEqual(orgs[0]["name"], org)

        org_id = await client.query_org_id()

        # 查询bucket
        buckets = await client.list_buckets()
        self.assertTrue(len(buckets) > 0)

        for bucket in buckets:
            if bucket["name"] == "ut_test_bucket":
                await client.delete_bucket(bucket["id"])

        # 创建bucket
        await client.create_bucket("this is ut_test_bucket", org_id=org_id)

        with self.assertRaises(InfluxSchemaError):
            await client.create_bucket(org, "this is ut_test_bucket")

        # 删除bucket by bucket name
        await client.delete_bucket()

        # test error handling
        await client.create_bucket("this is ut_test_bucket", org_id=org_id)
        with self.assertRaises(InfluxSchemaError):
            with mock.patch("aiohttp.ClientSession.delete") as mock_delete:
                mock_delete.return_value.__aenter__.status = 400
                mock_delete.return_value.__aenter__.json = mock.Mock()
                mock_delete.return_value.__aenter__.json.return_value = {
                    "code": -1,
                    "message": "mockerror",
                }
                await client.delete_bucket()
