import datetime
import os
import pickle
import unittest
from unittest import mock

import arrow
import cfg4py
import numpy as np
import pandas as pd
from coretypes import FrameType, SecurityType, bars_dtype, bars_with_limit_dtype
from numpy.testing import assert_array_equal

import omicron
from omicron import tf
from omicron.dal.influx.influxclient import InfluxClient
from omicron.extensions.np import numpy_append_fields
from omicron.models.stock import Stock
from tests import assert_bars_equal, bars_from_csv, init_test_env, test_dir

cfg = cfg4py.get_instance()


class StockTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        await init_test_env()
        await omicron.init()

        url, token, bucket, org = (
            cfg.influxdb.url,
            cfg.influxdb.token,
            cfg.influxdb.bucket_name,
            cfg.influxdb.org,
        )
        self.client = InfluxClient(url, token, bucket, org)

        # 关于数据准备，请参阅: data/README.md
        await Stock.reset_cache()
        await self.client.delete_bucket()
        await self.client.create_bucket()

        # feed 1 min data, last day goes to cache, others go to influxdb
        for i in (1, 2, 4):
            code = f"00000{i}.XSHE"
            bars = bars_from_csv(code, "1m")

            cache_date = bars[-1]["frame"].date()
            persist_date = tf.day_shift(cache_date, -1)
            persist_end = tf.combine_time(persist_date, 15)

            cache_bars = bars[bars["frame"] > persist_end]
            persist_bars = bars[bars["frame"] <= persist_end]

            await Stock.cache_bars(code, FrameType.MIN1, cache_bars)
            await Stock.persist_bars(FrameType.MIN1, {code: persist_bars})

        # fill m30, day, week, month into in influxdb
        # all these bars are closed
        for ft in (FrameType.MIN30, FrameType.DAY, FrameType.WEEK, FrameType.MONTH):
            for i in (1, 2, 4):
                try:
                    code = f"00000{i}.XSHE"
                    bars = bars_from_csv(code, ft.value)

                    await Stock.persist_bars(ft, {code: bars})
                except FileNotFoundError:
                    # no month bars for 000002/000004
                    pass

        # merge test data from backtest, which help find failed to deserialize data with high/low_limits issue
        for ft in (FrameType.MIN1, FrameType.DAY):
            file = os.path.join(test_dir(), f"data/bars_{ft.value}.pkl")
            with open(file, "rb") as f:
                bars = pickle.load(f)
                await Stock.persist_bars(ft, bars)

        df = pd.read_csv(
            os.path.join(test_dir(), "data/limits.csv"), sep="\t", parse_dates=["time"]
        )
        limits = df.to_records(index=False)
        limits.dtype.names = ["frame", "code", "high_limit", "low_limit"]
        await Stock.save_trade_price_limits(limits, False)

        return super().setUp()

    async def asyncTearDown(self) -> None:
        await omicron.close()
        return await super().asyncTearDown()

    def test_fuzzy_match(self):
        exp = set(["600000.XSHG"])
        self.assertSetEqual(exp, set(Stock.fuzzy_match("600").keys()))

        exp = set(["000001.XSHE", "600000.XSHG"])
        self.assertSetEqual(exp, set(Stock.fuzzy_match("P").keys()))

        exp = set(["000001.XSHE"])
        self.assertSetEqual(exp, set(Stock.fuzzy_match("平").keys()))

    async def test_resample(self):
        def totime(tm: str):
            return arrow.get(tm).datetime.replace(tzinfo=None)

        # fields = ["frame", "open", "high", "low", "close", "volume", "amount", "factor"]
        # jq.get_bars("002709.XSHE", count = 30, unit = '1m', fields=fields, fq_ref_date=None, end_dt = end, df=False)
        bars = np.array(
            [
                (
                    datetime.datetime(2021, 4, 27, 9, 31),
                    62.01,
                    62.59,
                    62.01,
                    62.56,
                    777800.0,
                    48370573.0,
                    6.976547,
                ),
                (
                    datetime.datetime(2021, 4, 27, 9, 32),
                    62.56,
                    62.58,
                    62.21,
                    62.46,
                    387300.0,
                    24182909.0,
                    6.976547,
                ),
                (
                    datetime.datetime(2021, 4, 27, 9, 33),
                    62.48,
                    62.79,
                    62.48,
                    62.76,
                    357100.0,
                    22373215.0,
                    6.976547,
                ),
                (
                    datetime.datetime(2021, 4, 27, 9, 34),
                    62.83,
                    63.6,
                    62.83,
                    63.5,
                    471800.0,
                    29798711.0,
                    6.976547,
                ),
                (
                    datetime.datetime(2021, 4, 27, 9, 35),
                    63.57,
                    63.6,
                    63.04,
                    63.04,
                    355900.0,
                    22577178.0,
                    6.976547,
                ),
                (
                    datetime.datetime(2021, 4, 27, 9, 36),
                    63.03,
                    63.26,
                    63.03,
                    63.12,
                    401600.0,
                    25354039.0,
                    6.976547,
                ),
                (
                    datetime.datetime(2021, 4, 27, 9, 37),
                    63.06,
                    63.18,
                    62.93,
                    63.18,
                    330400.0,
                    20832860.0,
                    6.976547,
                ),
                (
                    datetime.datetime(2021, 4, 27, 9, 38),
                    63.2,
                    63.2,
                    62.97,
                    62.97,
                    238600.0,
                    15055131.0,
                    6.976547,
                ),
                (
                    datetime.datetime(2021, 4, 27, 9, 39),
                    62.98,
                    63.73,
                    62.97,
                    63.73,
                    341300.0,
                    21612052.0,
                    6.976547,
                ),
                (
                    datetime.datetime(2021, 4, 27, 9, 40),
                    63.88,
                    64.61,
                    63.88,
                    64.61,
                    694600.0,
                    44473260.0,
                    6.976547,
                ),
                (
                    datetime.datetime(2021, 4, 27, 9, 41),
                    64.54,
                    64.61,
                    64.0,
                    64.09,
                    381600.0,
                    24521706.0,
                    6.976547,
                ),
                (
                    datetime.datetime(2021, 4, 27, 9, 42),
                    64.0,
                    64.01,
                    63.79,
                    63.8,
                    452600.0,
                    28940101.0,
                    6.976547,
                ),
                (
                    datetime.datetime(2021, 4, 27, 9, 43),
                    63.84,
                    63.94,
                    63.58,
                    63.58,
                    254800.0,
                    16266940.0,
                    6.976547,
                ),
                (
                    datetime.datetime(2021, 4, 27, 9, 44),
                    63.65,
                    63.65,
                    63.54,
                    63.58,
                    217000.0,
                    13794439.0,
                    6.976547,
                ),
                (
                    datetime.datetime(2021, 4, 27, 9, 45),
                    63.57,
                    63.9,
                    63.57,
                    63.87,
                    201800.0,
                    12868338.0,
                    6.976547,
                ),
                (
                    datetime.datetime(2021, 4, 27, 9, 46),
                    63.83,
                    63.83,
                    63.29,
                    63.31,
                    289800.0,
                    18402611.0,
                    6.976547,
                ),
            ],
            dtype=[
                ("frame", "O"),
                ("open", "<f4"),
                ("high", "<f4"),
                ("low", "<f4"),
                ("close", "<f4"),
                ("volume", "<f8"),
                ("amount", "<f8"),
                ("factor", "<f4"),
            ],
        )
        actual = Stock.resample(bars, FrameType.MIN1, FrameType.MIN5)
        # end = "2021-04-27 09:46:00"
        # exp = jq.get_bars("002709.XSHE", include_now = True, count = 4, unit = '5m', fields=fields, fq_ref_date=None, end_dt = end, df=False)
        exp = np.array(
            [
                (
                    datetime.datetime(2021, 4, 27, 9, 35),
                    62.01,
                    63.6,
                    62.01,
                    63.04,
                    2349900.0,
                    1.47302586e08,
                    6.976547,
                ),
                (
                    datetime.datetime(2021, 4, 27, 9, 40),
                    63.03,
                    64.61,
                    62.93,
                    64.61,
                    2006500.0,
                    1.27327342e08,
                    6.976547,
                ),
                (
                    datetime.datetime(2021, 4, 27, 9, 45),
                    64.54,
                    64.61,
                    63.54,
                    63.87,
                    1507800.0,
                    9.63915240e07,
                    6.976547,
                ),
                (
                    datetime.datetime(2021, 4, 27, 9, 46),
                    63.83,
                    63.83,
                    63.29,
                    63.31,
                    289800.0,
                    1.84026110e07,
                    6.976547,
                ),
            ],
            dtype=[
                ("frame", "O"),
                ("open", "<f4"),
                ("high", "<f4"),
                ("low", "<f4"),
                ("close", "<f4"),
                ("volume", "<f8"),
                ("amount", "<f8"),
                ("factor", "<f4"),
            ],
        )
        assert_bars_equal(exp, actual)

        # resample to 15m
        exp = np.array(
            [
                (
                    datetime.datetime(2021, 4, 27, 9, 45),
                    62.01,
                    64.61,
                    62.01,
                    63.87,
                    5864200.0,
                    3.71021452e08,
                    6.976547,
                ),
                (
                    datetime.datetime(2021, 4, 27, 9, 46),
                    63.83,
                    63.83,
                    63.29,
                    63.31,
                    289800.0,
                    1.84026110e07,
                    6.976547,
                ),
            ],
            dtype=[
                ("frame", "O"),
                ("open", "<f4"),
                ("high", "<f4"),
                ("low", "<f4"),
                ("close", "<f4"),
                ("volume", "<f8"),
                ("amount", "<f8"),
                ("factor", "<f4"),
            ],
        )

        actual = Stock.resample(bars, FrameType.MIN1, FrameType.MIN15)
        assert_bars_equal(exp, actual)

        # resample to 1d
        exp = np.array(
            [
                (
                    datetime.date(2021, 4, 27),
                    62.01,
                    64.61,
                    62.01,
                    63.31,
                    6154000,
                    389424063.0,
                    6.976547,
                )
            ],
            dtype=bars_dtype,
        )
        actual = Stock.resample(bars, FrameType.MIN1, FrameType.DAY)
        assert_bars_equal(exp, actual)

        # resample when input bars can be evenly divided
        actual = Stock.resample(bars[:-1], FrameType.MIN1, FrameType.MIN15)
        exp = np.array(
            [
                (
                    datetime.datetime(2021, 4, 27, 9, 45),
                    62.01,
                    64.61,
                    62.01,
                    63.87,
                    5864200.0,
                    3.71021452e08,
                    6.976547,
                )
            ],
            dtype=[
                ("frame", "O"),
                ("open", "<f4"),
                ("high", "<f4"),
                ("low", "<f4"),
                ("close", "<f4"),
                ("volume", "<f8"),
                ("amount", "<f8"),
                ("factor", "<f4"),
            ],
        )
        assert_bars_equal(exp, actual)

        # 输入错误检查
        bars[0]["frame"] = datetime.datetime(2021, 4, 27, 9, 35)
        try:
            Stock.resample(bars, FrameType.MIN1, FrameType.MIN5)
        except ValueError as e:
            self.assertEqual(str(e), "resampling from 1min must start from 9:31")

    @mock.patch.object(arrow, "now", return_value=arrow.get("2022-02-09 10:33:00"))
    async def test_get_cached_bars(self, mock_now):
        """cache_bars, cache_unclosed_bars are tested also"""
        # 1. end < ff, 返回空数组
        code = "000001.XSHE"
        ff = arrow.get("2022-02-09 09:31:00")
        lf = arrow.get("2022-02-09 10:33:00")

        tm = ff.shift(minutes=-1).naive
        bars = await Stock._get_cached_bars("000001.XSHE", tm, 10, FrameType.MIN1)
        self.assertEqual(len(bars), 0)

        # 2 end 刚大于 ff, 返回两个bar
        tm = ff.shift(minutes=5).naive
        bars = await Stock._get_cached_bars("000001.XSHE", tm, 10, FrameType.MIN5)
        self.assertEqual(len(bars), 2)
        self.assertEqual(datetime.datetime(2022, 2, 9, 9, 36), bars["frame"][1])

        # 3.1 end > last frame in cache
        tm = lf.shift(minutes=15).naive
        bars = await Stock._get_cached_bars("000001.XSHE", tm, 10, FrameType.MIN5)

        self.assertEqual(10, len(bars))
        self.assertEqual(datetime.datetime(2022, 2, 9, 10, 33), bars["frame"][-1])

        # 3.2 check other fields are equal
        m1_bars = bars_from_csv(code, "1m", 242, 304)
        expected = Stock.resample(m1_bars, FrameType.MIN1, FrameType.MIN5)
        assert_bars_equal(expected[3:], bars)

        # 4. if n < len(cached bars)
        bars = await Stock._get_cached_bars(code, tm, 2, FrameType.MIN5)
        self.assertEqual(2, len(bars))
        assert_bars_equal(expected[-2:], bars)

        # 5. FrameType == min1, no resample
        tm = lf.naive
        bars = await Stock._get_cached_bars(code, tm, 36, FrameType.MIN1)
        self.assertEqual(36, len(bars))

        # 如果为日线，且缓存中不存在日线，此时需要从分钟线中取
        bars = await Stock._get_cached_bars("000001.XSHE", lf.date(), 2, FrameType.DAY)
        exp = np.array(
            [
                (
                    datetime.date(2022, 2, 9),
                    16.92,
                    17.0,
                    16.78,
                    16.84,
                    49839100.0,
                    843350578.0,
                    121.71913,
                )
            ],
            dtype=bars_dtype,
        )
        assert_bars_equal(exp, bars)

        # 6. 当cache为空时，应该返回空数组
        await Stock.reset_cache()

        bars = await Stock._get_cached_bars("000001.XSHE", lf.naive, 10, FrameType.MIN1)
        self.assertEqual(bars.size, 0)

    @mock.patch.object(arrow, "now", return_value=arrow.get("2022-02-09 10:33:00"))
    async def test_get_bars(self, mock_now):
        code = "000001.XSHE"
        ft = FrameType.MIN1

        ff = arrow.get("2022-02-09 09:31:00")
        lf = arrow.get("2022-02-09 10:33:00")

        # 1. end is None, 取当前时间作为end.
        end = None
        n = 1
        bars = await Stock.get_bars(code, n, ft, end, fq=False)
        exp = bars_from_csv(code, "1m", 304, 304)
        assert_bars_equal(exp, bars)

        # 2. end < ff，仅从persistent中取
        tm = ff.shift(minutes=-1).naive
        n = 2

        bars = await Stock.get_bars(code, n, ft, tm, fq=False)
        expected = bars_from_csv(code, "1m", 240, 241)
        assert_bars_equal(expected, bars)

        # 3.1 end > ff，从persistent和cache中取,不包含unclosed
        ft = FrameType.MIN30
        tm = ff.shift(minutes=32).naive
        n = 2
        from_persist = bars_from_csv(code, "30m")[-1:]
        from_cache = bars_from_csv(code, "1m", 242, 274)
        from_cache = Stock.resample(from_cache, FrameType.MIN1, ft)
        bars = await Stock.get_bars(code, n, ft, tm, fq=False, unclosed=False)

        self.assertEqual(2, bars.size)
        assert_bars_equal(from_persist, bars[:1])
        assert_bars_equal(from_cache[:-1], bars[1:])

        # 3.2 end > ff, 从persistent和cache中取,包含unclosed
        n = 3
        bars = await Stock.get_bars(code, n, ft, tm, fq=False, unclosed=True)
        self.assertEqual(n, bars.size)
        assert_bars_equal(from_persist[-1:], bars[:1])
        assert_bars_equal(from_cache, bars[1:])

        # 4. check fq = True
        with mock.patch.object(Stock, "qfq") as mocked_qfq:
            end = None
            n = 1
            bars = await Stock.get_bars(code, n, ft, end, fq=True)
            mocked_qfq.assert_called()

        # 5. ft == DAY, in trade time
        ft = FrameType.DAY
        n = 2
        bars = await Stock.get_bars(code, n, ft, end=lf.date(), fq=False)
        exp = np.array(
            [
                (
                    datetime.date(2022, 2, 8),
                    16.3,
                    16.97,
                    16.26,
                    16.83,
                    1.75469528e08,
                    2.95030901e09,
                    121.71913,
                ),
                (
                    datetime.date(2022, 2, 9),
                    16.92,
                    17.0,
                    16.78,
                    16.84,
                    4.98391000e07,
                    8.43350578e08,
                    121.71913,
                ),
            ],
            dtype=bars_dtype,
        )

        assert_bars_equal(exp, bars)

        # error handling
        with self.assertRaises(ValueError):
            with mock.patch(
                "omicron.models.stock.Stock._get_cached_bars",
                side_effect=ValueError("test"),
            ):
                await Stock.get_bars(code, n, ft, end=lf.date(), fq=False)

    @mock.patch.object(arrow, "now", return_value=datetime.datetime(2022, 1, 10, 9, 34))
    async def test_batch_cache_bars(self, mock_now):
        data = {
            "000001.XSHE": np.array(
                [
                    (
                        datetime.datetime(2022, 1, 10, 9, 31),
                        17.29,
                        17.32,
                        17.26,
                        17.27,
                        3828400.0,
                        66221789.0,
                        121.71913,
                    ),
                    (
                        datetime.datetime(2022, 1, 10, 9, 32),
                        17.27,
                        17.36,
                        17.26,
                        17.36,
                        1380000.0,
                        23888716.0,
                        121.71913,
                    ),
                    (
                        datetime.datetime(2022, 1, 10, 9, 33),
                        17.36,
                        17.42,
                        17.36,
                        17.41,
                        2411400.0,
                        41931080.0,
                        121.71913,
                    ),
                    (
                        datetime.datetime(2022, 1, 10, 9, 34),
                        17.42,
                        17.42,
                        17.38,
                        17.38,
                        1597500.0,
                        27808995.0,
                        121.71913,
                    ),
                ],
                dtype=bars_dtype,
            ),
            "000004.XSHE": np.array(
                [
                    (
                        datetime.datetime(2022, 1, 10, 9, 34),
                        20.74,
                        20.89,
                        20.72,
                        20.76,
                        70800.0,
                        1470015.0,
                        7.446,
                    ),
                ],
                dtype=bars_dtype,
            ),
        }

        await Stock.reset_cache()
        await Stock.batch_cache_bars(FrameType.MIN1, data)

        bars = await Stock._get_cached_bars(
            "000001.XSHE", datetime.datetime(2022, 1, 10, 9, 34), 4, FrameType.MIN1
        )
        exp = np.array(
            [
                (
                    datetime.datetime(2022, 1, 10, 9, 34),
                    17.42,
                    17.42,
                    17.38,
                    17.38,
                    1597500.0,
                    27808995.0,
                    121.71913,
                )
            ],
            dtype=bars_dtype,
        )

        assert_bars_equal(exp, bars[-1:])
        self.assertEqual(4, len(bars))

        bars = await Stock._get_cached_bars(
            "000004.XSHE", datetime.datetime(2022, 1, 10, 9, 34), 1, FrameType.MIN1
        )
        exp = np.array(
            [
                (
                    datetime.datetime(2022, 1, 10, 9, 34),
                    20.74,
                    20.89,
                    20.72,
                    20.76,
                    70800.0,
                    1470015.0,
                    7.446,
                )
            ],
            dtype=bars_dtype,
        )
        assert_bars_equal(exp, bars)

    async def test_stock_ctor(self):
        payh = Stock("000001.XSHE")
        self.assertEqual(payh.code, "000001.XSHE")
        self.assertEqual(payh.name, "PAYH")
        self.assertEqual(payh.display_name, "平安银行")
        self.assertEqual(payh.ipo_date, datetime.date(1991, 4, 3))
        self.assertEqual(payh.security_type, SecurityType.STOCK)
        self.assertEqual(payh.end_date, datetime.date(2200, 1, 1))

        with mock.patch.object(
            arrow, "now", return_value=datetime.datetime(2022, 1, 15)
        ):
            self.assertEqual(payh.days_since_ipo(), 4141)

        self.assertEqual("平安银行[000001.XSHE]", str(payh))
        self.assertEqual("000001", payh.sim_code)
        self.assertEqual("000001", Stock.simplify_code("000001.XSHE"))

    async def test_get_persisted_bars(self):
        code = "000001.XSHE"
        start = arrow.get("2022-02-08 10:00:00").naive
        end = arrow.get("2022-02-08 15:00:00")

        bars = await Stock._get_persisted_bars(code, FrameType.MIN30, begin=start)

        self.assertEqual(bars.size, 8)
        expected = bars_from_csv(code, "30m")
        assert_bars_equal(expected, bars)

        # test with end
        bars = await Stock._get_persisted_bars(
            "000001.XSHE", FrameType.MIN30, n=8, begin=start, end=end
        )

        self.assertEqual(len(bars), 8)
        expected = bars_from_csv(code, "30m")
        assert_bars_equal(expected, bars)

        # test with FrameType.DAY, to see if it can handle date correctly
        ft = FrameType.DAY
        start = arrow.get("2022-02-08").date()
        end = start

        actual = await Stock._get_persisted_bars(code, ft, start, end)
        expected = bars_from_csv(code, "1d")
        assert_bars_equal(expected[1:], actual)

    async def test_batch_get_persisted_bars(self):
        await self.client.drop_measurement("stock_bars_1d")
        # test with multiple codes
        data = {
            code: bars_from_csv(code, "1d") for code in ["000001.XSHE", "000002.XSHE"]
        }

        await Stock.persist_bars(FrameType.DAY, data)

        start = arrow.get("2021-09-03").date()
        end = arrow.get("2022-02-08").date()
        actual = await Stock._batch_get_persisted_bars(
            list(data.keys()), FrameType.DAY, start, end=end
        )
        for code in data.keys():
            assert_bars_equal(data[code], actual[code])

        await self.client.drop_measurement("stock_bars_30m")
        data = {
            code: bars_from_csv(code, "30m") for code in ["000001.XSHE", "000002.XSHE"]
        }
        await Stock.persist_bars(FrameType.MIN30, data)

        start = arrow.get("2022-01-17 13:30:00").naive
        end = arrow.get("2022-02-09 15:00:00").naive
        actual = await Stock._batch_get_persisted_bars(
            list(data.keys()), FrameType.MIN30, start, end=end
        )
        for code in data.keys():
            assert_bars_equal(data[code], actual[code])

    async def test_get_trade_price_limits(self):
        measurement = "stock_bars_1d"

        # fill in data
        start = datetime.date(2022, 1, 10)
        dtype = [
            ("frame", "O"),
            ("high_limit", "<f4"),
            ("low_limit", "<f4"),
            ("factor", "<f4"),
        ]

        trade_limits = np.array(
            [
                (datetime.date(2022, 1, 10), 18.92, 15.48, 1.0),
                (datetime.date(2022, 1, 11), 18.91, 15.47, 1.1),
                (datetime.date(2022, 1, 12), 19.15, 15.67, 1.11),
                (datetime.date(2022, 1, 13), 18.7, 15.3, 1.12),
                (datetime.date(2022, 1, 14), 18.68, 15.28, 1.13),
                (datetime.date(2022, 1, 17), 17.96, 14.7, 1.14),
                (datetime.date(2022, 1, 18), 17.84, 14.6, 1.15),
                (datetime.date(2022, 1, 19), 18.17, 14.87, 1.15),
                (datetime.date(2022, 1, 20), 18.15, 14.85, 1.15),
                (datetime.date(2022, 1, 21), 19.06, 15.6, 1.15),
                (datetime.date(2022, 1, 24), 19.09, 15.62, 1.15),
                (datetime.date(2022, 1, 25), 18.92, 15.48, 1.15),
                (datetime.date(2022, 1, 26), 18.54, 15.17, 1.16),
                (datetime.date(2022, 1, 27), 18.32, 14.99, 1.17),
                (datetime.date(2022, 1, 28), 17.93, 14.67, 1.18),
                (datetime.date(2022, 2, 7), 17.41, 14.25, 1.19),
                (datetime.date(2022, 2, 8), 18.03, 14.75, 1.2),
            ],
            dtype=dtype,
        )

        await self.client.save(
            trade_limits,
            measurement,
            time_key="frame",
            global_tags={"code": "000001.XSHE"},
        )

        end = arrow.get("2022-02-08").date()
        result = await Stock.get_trade_price_limits("000001.XSHE", start, end)

        for col in ["high_limit", "low_limit"]:
            np.testing.assert_array_almost_equal(trade_limits[col], result[col])

        np.testing.assert_array_equal(trade_limits["frame"], result["frame"])

    async def test_get_limits_in_range(self):
        measurement = "stock_bars_1d"
        code = "000001.XSHE"
        await self.client.drop_measurement(measurement)
        bars = np.array(
            [
                (
                    datetime.date(2022, 1, 7),
                    17.1,
                    17.28,
                    17.06,
                    17.8,
                    1.1266307e08,
                    1.93771096e09,
                    1.0,
                    18.83,
                    15.41,
                ),
                (
                    datetime.date(2022, 1, 6),
                    17.1,
                    17.28,
                    17.06,
                    17.2,
                    1.1266307e08,
                    1.93771096e09,
                    1.0,
                    18.83,
                    15.41,
                ),
            ],
            dtype=bars_with_limit_dtype,
        )

        await self.client.save(
            bars, measurement, time_key="frame", global_tags={"code": code}
        )

        start = datetime.date(2022, 1, 6)
        end = datetime.date(2022, 1, 7)
        actual = await Stock.get_trade_price_limits(code, start, end)

        expected = np.array(
            [
                (datetime.date(2022, 1, 6), 18.83, 15.41, 1.0),
                (datetime.date(2022, 1, 7), 18.83, 15.41, 1.0),
            ],
            dtype=[
                ("frame", "O"),
                ("high_limit", "<f4"),
                ("low_limit", "<f4"),
                ("factor", "<f4"),
            ],
        )

        for col in ["high_limit", "low_limit"]:
            np.testing.assert_array_almost_equal(expected[col], actual[col])

        np.testing.assert_array_equal(expected["frame"], actual["frame"])

    async def test_batch_get_cached_bars(self):
        codes = ["000001.XSHE", "000002.XSHE", "000004.XSHE"]

        ft = FrameType.MIN1
        unclosed = True

        end = arrow.get("2022-02-09 10:33:00").naive
        with mock.patch.object(arrow, "now", return_value=end):
            for unclosed in (True, False):
                # doesn't matter for 1min
                result = await Stock._batch_get_cached_bars(codes, end, 10, ft)
                self.assertEqual(10, result["000001.XSHE"].size)
                self.assertEqual(10, result["000004.XSHE"].size)
                self.assertEqual(end, result["000001.XSHE"][-1]["frame"])
                exp_start = arrow.get(end).shift(minutes=-9).naive
                self.assertEqual(exp_start, result["000001.XSHE"][0]["frame"])

            # if some code contains no bars
            codes = ["000001.XSHE", "000002.XSHE", "000003.XSHE"]
            result = await Stock._batch_get_cached_bars(codes, end, 10, ft)
            self.assertEqual(3, len(result))
            self.assertEqual(0, result["000003.XSHE"].size)

    def test_qfq(self):
        # 000001.XSHE, 2021-5-14, week bars
        # 注意通过jq.get_bars获取的前复权数据与expected有差异。其一，最后一个bar的数据，在前复权情况下，应该等于未复权数据，即现价。jq给出的数据是不一致的；其二，它的成交量复权，与我们的计算结果不一致。但同一公式用来计算其它字段复权则是对的。
        bars = np.array(
            [
                (
                    datetime.date(2021, 4, 30),
                    23.87,
                    24.23,
                    22.78,
                    23.29,
                    3.11350520e08,
                    7.24329575e09,
                    120.769436,
                ),
                (
                    datetime.date(2021, 5, 7),
                    23.1,
                    24.3,
                    23.1,
                    24.05,
                    1.30250943e08,
                    3.10329398e09,
                    120.769436,
                ),
                (
                    datetime.date(2021, 5, 14),
                    24.0,
                    24.04,
                    22.6,
                    23.32,
                    2.80536547e08,
                    6.54642212e09,
                    121.71913,
                ),
            ],
            dtype=bars_dtype,
        )

        expected = np.array(
            [
                (
                    datetime.date(2021, 4, 30),
                    23.68,
                    24.04,
                    22.6,
                    23.11,
                    3.08921267e08,
                    7.24329575e09,
                    120.769436,
                ),
                (
                    datetime.date(2021, 5, 7),
                    22.92,
                    24.11,
                    22.92,
                    23.86,
                    1.29234685e08,
                    3.10329398e09,
                    120.769436,
                ),
                (
                    datetime.date(2021, 5, 14),
                    24.0,
                    24.04,
                    22.6,
                    23.32,
                    2.80536547e08,
                    6.54642212e09,
                    121.71913,
                ),
            ],
            dtype=bars_dtype,
        )

        actual = Stock.qfq(bars)
        assert_bars_equal(expected, actual)

    @mock.patch.object(arrow, "now", return_value=arrow.get("2022-02-09 10:33:00"))
    async def test_batch_get_bars(self, mocked_now):
        codes = ["000001.XSHE", "000002.XSHE", "000003.XSHE"]

        # 30分钟线
        end = arrow.get("2022-02-09 10:33:00").naive
        ft = FrameType.MIN30
        bars = await Stock.batch_get_bars(codes, 11, ft, end, fq=False)

        self.assertEqual(3, len(bars))
        self.assertEqual(11, bars["000001.XSHE"].size)
        self.assertEqual(11, bars["000002.XSHE"].size)
        self.assertEqual(0, bars["000003.XSHE"].size)

        code = "000001.XSHE"
        from_persist = bars_from_csv(code, "30m", 2, 9)
        from_cache = bars_from_csv(code, "1m", 242, 304)
        from_cache = Stock.resample(from_cache, FrameType.MIN1, ft)

        assert_bars_equal(from_persist, bars[code][:-3])
        assert_bars_equal(from_cache, bars[code][8:])

        # all in cache
        bars = await Stock.batch_get_bars(codes, 3, ft, end, fq=False)
        self.assertEqual(3, len(bars))
        assert_bars_equal(from_cache, bars[code])

        # 日线
        ft = FrameType.DAY
        bars = await Stock.batch_get_bars(codes, 2, ft, end, fq=True)
        self.assertEqual(3, len(bars))
        self.assertEqual(2, bars[code].size)
        self.assertEqual(datetime.date(2022, 2, 8), bars[code][0]["frame"])
        self.assertEqual(datetime.date(2022, 2, 9), bars[code][1]["frame"])

        # 周线
        end = arrow.now().date()
        bars = await Stock.batch_get_bars(
            codes, 2, FrameType.WEEK, end, fq=False, unclosed=True
        )
        from_persist = bars_from_csv(code, "1w")
        self.assertEqual(3, len(bars))
        self.assertEqual(2, bars[code].size)
        self.assertEqual(from_persist[0]["frame"], bars[code][0]["frame"])

    async def test_save_trade_price_limits(self):
        limits = np.array(
            [
                (
                    datetime.date(2022, 3, 22),
                    3.0,
                    3.14,
                    2.7,
                    3.14,
                    1.64808427e08,
                    4.86487509e08,
                    1.0,
                    3.14,
                    2.57,
                ),
                (
                    datetime.date(2022, 3, 23),
                    3.45,
                    3.45,
                    3.45,
                    3.45,
                    3.83227050e07,
                    1.32213332e08,
                    1.0,
                    3.45,
                    2.83,
                ),
                (
                    datetime.date(2022, 3, 24),
                    3.45,
                    3.8,
                    3.27,
                    3.8,
                    2.78554700e08,
                    9.86430823e08,
                    1.0,
                    3.8,
                    3.11,
                ),
                (
                    datetime.date(2022, 3, 25),
                    3.76,
                    4.11,
                    3.52,
                    3.76,
                    2.68746669e08,
                    1.02163013e09,
                    1.0,
                    4.18,
                    3.42,
                ),
                (
                    datetime.date(2022, 3, 28),
                    3.38,
                    4.0,
                    3.38,
                    3.76,
                    2.36926943e08,
                    8.49636557e08,
                    1.0,
                    4.14,
                    3.38,
                ),
                (
                    datetime.date(2022, 3, 29),
                    3.58,
                    3.68,
                    3.38,
                    3.38,
                    1.29861905e08,
                    4.46620979e08,
                    1.0,
                    4.14,
                    3.38,
                ),
                (
                    datetime.date(2022, 3, 30),
                    3.1,
                    3.35,
                    3.05,
                    3.07,
                    1.71553784e08,
                    5.43805962e08,
                    1.0,
                    3.72,
                    3.04,
                ),
                (
                    datetime.date(2022, 3, 31),
                    3.07,
                    3.19,
                    2.9,
                    3.14,
                    1.74144017e08,
                    5.28782426e08,
                    1.0,
                    3.38,
                    2.76,
                ),
                (
                    datetime.date(2022, 4, 1),
                    3.01,
                    3.06,
                    2.91,
                    2.94,
                    1.03947342e08,
                    3.09315863e08,
                    1.0,
                    3.45,
                    2.83,
                ),
                (
                    datetime.date(2022, 4, 6),
                    2.92,
                    3.05,
                    2.91,
                    3.01,
                    8.73678140e07,
                    2.60495520e08,
                    1.0,
                    3.23,
                    2.65,
                ),
            ],
            dtype=[
                ("frame", "O"),
                ("open", "<f4"),
                ("high", "<f4"),
                ("low", "<f4"),
                ("close", "<f4"),
                ("volume", "<f8"),
                ("amount", "<f8"),
                ("factor", "<f4"),
                ("high_limit", "<f4"),
                ("low_limit", "<f4"),
            ],
        )

        code = "002482.XSHE"
        limits = numpy_append_fields(
            limits, "code", [code] * len(limits), [("code", "O")]
        )

        start = datetime.date(2022, 3, 23)
        end = datetime.date(2022, 4, 6)

        await Stock.save_trade_price_limits(limits, False)
        actual = await Stock.get_trade_price_limits(code, start, end)
        self.assertAlmostEqual(3.45, actual[0]["high_limit"])

        # save it to cache
        await Stock.save_trade_price_limits(limits, True)

    @mock.patch.object(arrow, "now", return_value=arrow.get("2022-02-09 10:33:00"))
    async def test_get_bars_in_range(self, mocked_now):
        code = "000001.XSHE"
        start = datetime.datetime(2022, 2, 8, 10)
        end = mocked_now.return_value.naive

        bars = await Stock.get_bars_in_range(code, FrameType.MIN30, start, end)
        self.assertEqual(11, len(bars))
        self.assertEqual(datetime.datetime(2022, 2, 8, 10), bars[0]["frame"])
        self.assertEqual(datetime.datetime(2022, 2, 9, 10, 33), bars[-1]["frame"])

        bars = await Stock.get_bars_in_range(
            code, FrameType.MIN30, start, end, unclosed=False
        )
        self.assertEqual(10, len(bars))
        self.assertEqual(datetime.datetime(2022, 2, 8, 10), bars[0]["frame"])
        self.assertEqual(datetime.datetime(2022, 2, 9, 10, 30), bars[-1]["frame"])

    @mock.patch.object(arrow, "now", return_value=arrow.get("2022-02-09 10:33:00"))
    async def test_batch_get_bars_in_range(self, mocked_now):
        codes = ["000001.XSHE", "000002.XSHE", "000003.XSHE"]
        start = datetime.datetime(2022, 2, 8, 10)
        end = mocked_now.return_value.naive

        actual = await Stock.batch_get_bars_in_range(
            codes, FrameType.MIN30, start, end, fq=False
        )
        self.assertEqual(3, len(actual))
        self.assertEqual(11, len(actual["000001.XSHE"]))
        self.assertEqual(0, len(actual["000003.XSHE"]))

        self.assertEqual(
            datetime.datetime(2022, 2, 8, 10), actual["000001.XSHE"][0]["frame"]
        )
        self.assertEqual(
            datetime.datetime(2022, 2, 9, 10, 33), actual["000001.XSHE"][-1]["frame"]
        )

    async def test_batch_cache_unclosed_bars(self):
        data = {
            "000001.XSHE": bars_from_csv("000001.XSHE", "1d"),
        }
        await Stock.batch_cache_unclosed_bars(FrameType.DAY, data)
        data = await Stock.batch_get_bars(
            ["000001.XSHE"], 1, FrameType.DAY, unclosed=True
        )
        self.assertEqual(1, len(data))
        self.assertEqual(datetime.date(2022, 2, 7), data["000001.XSHE"][0]["frame"])

    def test_choose_by_date(self):
        # 测试获取股票，过滤掉还没上市的股票
        codes = Stock.choose_listed(
            arrow.get("2000-01-01").date(), [SecurityType.STOCK.value]
        )
        self.assertEqual(
            codes,
            ["000001.XSHE", "000406.XSHE", "000005.XSHE", "600000.XSHG", "000007.XSHE"],
        )
        # 测试获取股票，过滤掉没上市和已经退市的股票
        codes = Stock.choose_listed(
            arrow.get("2007-01-01").date(), [SecurityType.STOCK.value]
        )
        self.assertEqual(
            codes, ["000001.XSHE", "000005.XSHE", "600000.XSHG", "000007.XSHE"]
        )

    @mock.patch.object(arrow, "now", return_value=arrow.get("2022-02-09 10:33:00"))
    async def test_get_day_level_bars(self, mocked_now):
        # 假设今天是2022-2-10日。cache中有到2/10日的分钟线数据，持久化中有2/7~2/9的日线数据和到1月28日的周线数据。
        code = "000001.XSHE"

        # 1. 周线
        expected = np.array(
            [
                (
                    datetime.date(2022, 1, 28),
                    17.34,
                    17.38,
                    15.82,
                    15.83,
                    5.65323684e08,
                    9.37250597e09,
                    121.71913,
                ),
                (
                    datetime.date(2022, 2, 9),
                    16.02,
                    17.0,
                    15.89,
                    16.84,
                    3.76856258e08,
                    6.24545808e09,
                    121.71913,
                ),
            ],
            dtype=bars_dtype,
        )
        actual = await Stock._get_day_level_bars(
            code, 2, FrameType.WEEK, datetime.date(2022, 2, 9)
        )
        assert_bars_equal(expected, actual)

        # 2. 日线
        actual = await Stock._get_day_level_bars(
            code, 3, FrameType.DAY, datetime.date(2022, 2, 9)
        )
        expected = np.array(
            [
                (
                    datetime.date(2022, 2, 7),
                    16.02,
                    16.41,
                    15.89,
                    16.39,
                    1.51547630e08,
                    2.45179848e09,
                    121.71913,
                ),
                (
                    datetime.date(2022, 2, 8),
                    16.3,
                    16.97,
                    16.26,
                    16.83,
                    1.75469528e08,
                    2.95030901e09,
                    121.71913,
                ),
                (
                    datetime.date(2022, 2, 9),
                    16.92,
                    17.0,
                    16.78,
                    16.84,
                    4.98391000e07,
                    8.43350578e08,
                    121.71913,
                ),
            ],
            dtype=bars_dtype,
        )
        assert_bars_equal(expected, actual)

        # 3. 日线，unclosed已缓存
        bars = bars_from_csv("000001.XSHE", "1d")[1:]
        bars["frame"] = datetime.date(2022, 2, 9)

        # 让cache中的数据更新为实际为2月8日数据，以证明当unclosed数据存在时，优先使用unclosed
        expected[2] = bars

        await Stock.cache_unclosed_bars("000001.XSHE", FrameType.DAY, bars)
        actual = await Stock._get_day_level_bars(
            code, 3, FrameType.DAY, datetime.date(2022, 2, 9)
        )

        assert_bars_equal(expected, actual)

    def test_resample_from_day(self):
        day_bars = bars_from_csv("000004.XSHE", "1d")

        actual = Stock._resample_from_day(day_bars, FrameType.WEEK)
        expected_week_bars = np.array(
            [
                (
                    datetime.date(2021, 9, 10),
                    18.56,
                    19.58,
                    18.25,
                    18.86,
                    15651442.0,
                    2.98069614e08,
                    7.446,
                ),
                (
                    datetime.date(2021, 9, 17),
                    18.86,
                    20.49,
                    18.5,
                    19.16,
                    32686734.0,
                    6.48175352e08,
                    7.446,
                ),
                (
                    datetime.date(2021, 9, 24),
                    18.78,
                    20.67,
                    18.65,
                    20.0,
                    14061718.0,
                    2.78599598e08,
                    7.446,
                ),
            ],
            dtype=bars_dtype,
        )
        print(actual)
        assert_bars_equal(expected_week_bars, actual[1:4])

        expected_month_bars = np.array(
            [
                (
                    datetime.date(2021, 10, 29),
                    19.05,
                    20.5,
                    16.2,
                    16.79,
                    48524039.0,
                    9.14937381e08,
                    7.446,
                ),
                (
                    datetime.date(2021, 11, 30),
                    16.78,
                    20.58,
                    16.28,
                    18.82,
                    84562100.0,
                    1.57764758e09,
                    7.446,
                ),
                (
                    datetime.date(2021, 12, 31),
                    19.0,
                    19.99,
                    18.02,
                    19.42,
                    67253473.0,
                    1.28188833e09,
                    7.446,
                ),
            ],
            dtype=bars_dtype,
        )

        actual = Stock._resample_from_day(day_bars, FrameType.MONTH)
        assert_bars_equal(expected_month_bars, actual[1:4])

    async def test_trade_price_limit_flags(self):
        limits = np.array(
            [
                (
                    datetime.date(2022, 3, 22),
                    3.0,
                    3.14,
                    2.7,
                    3.14,
                    1.64808427e08,
                    4.86487509e08,
                    1.0,
                    3.14,
                    2.57,
                ),
                (
                    datetime.date(2022, 3, 23),
                    3.45,
                    3.45,
                    3.45,
                    3.45,
                    3.83227050e07,
                    1.32213332e08,
                    1.0,
                    3.45,
                    2.83,
                ),
                (
                    datetime.date(2022, 3, 24),
                    3.45,
                    3.8,
                    3.27,
                    3.8,
                    2.78554700e08,
                    9.86430823e08,
                    1.0,
                    3.8,
                    3.11,
                ),
                (
                    datetime.date(2022, 3, 25),
                    3.76,
                    4.11,
                    3.52,
                    3.76,
                    2.68746669e08,
                    1.02163013e09,
                    1.0,
                    4.18,
                    3.42,
                ),
                (
                    datetime.date(2022, 3, 28),
                    3.38,
                    4.0,
                    3.38,
                    3.76,
                    2.36926943e08,
                    8.49636557e08,
                    1.0,
                    4.14,
                    3.38,
                ),
                (
                    datetime.date(2022, 3, 29),
                    3.58,
                    3.68,
                    3.38,
                    3.38,
                    1.29861905e08,
                    4.46620979e08,
                    1.0,
                    4.14,
                    3.38,
                ),
                (
                    datetime.date(2022, 3, 30),
                    3.1,
                    3.35,
                    3.05,
                    3.07,
                    1.71553784e08,
                    5.43805962e08,
                    1.0,
                    3.72,
                    3.04,
                ),
                (
                    datetime.date(2022, 3, 31),
                    3.07,
                    3.19,
                    2.9,
                    3.14,
                    1.74144017e08,
                    5.28782426e08,
                    1.0,
                    3.38,
                    2.76,
                ),
                (
                    datetime.date(2022, 4, 1),
                    3.01,
                    3.06,
                    2.91,
                    2.94,
                    1.03947342e08,
                    3.09315863e08,
                    1.0,
                    3.45,
                    2.83,
                ),
                (
                    datetime.date(2022, 4, 6),
                    2.92,
                    3.05,
                    2.91,
                    3.01,
                    8.73678140e07,
                    2.60495520e08,
                    1.0,
                    3.23,
                    2.65,
                ),
            ],
            dtype=[
                ("frame", "O"),
                ("open", "<f4"),
                ("high", "<f4"),
                ("low", "<f4"),
                ("close", "<f4"),
                ("volume", "<f8"),
                ("amount", "<f8"),
                ("factor", "<f4"),
                ("high_limit", "<f4"),
                ("low_limit", "<f4"),
            ],
        )

        code = "002482.XSHE"
        limits = numpy_append_fields(
            limits, "code", [code] * len(limits), [("code", "O")]
        )

        start = datetime.date(2022, 3, 23)
        end = datetime.date(2022, 4, 6)

        await Stock.save_trade_price_limits(limits, False)

        buy_limit, sell_limit = await Stock.trade_price_limit_flags(code, start, end)

        assert_array_equal(
            [True, True, False, False, False, False, False, False, False], buy_limit
        )
        assert_array_equal(
            [False, False, False, False, True, False, False, False, False], sell_limit
        )
