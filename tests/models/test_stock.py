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

import omicron
from omicron.core.constants import TRADE_PRICE_LIMITS
from omicron.dal import cache
from omicron.dal.influx.influxclient import InfluxClient
from omicron.extensions import numpy_append_fields
from omicron.models.stock import Stock
from omicron.models.timeframe import TimeFrame
from tests import assert_bars_equal, bars_from_csv, init_test_env, test_dir

cfg = cfg4py.get_instance()

ranges_1m = {"cache_start": None, "cache_stop": None, "db_start": None, "db_stop": None}

ranges_30m = ranges_1m.copy()
ranges_1d = ranges_1m.copy()
ranges_1w = ranges_1m.copy()


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

        # fill in cache
        await Stock.reset_cache()
        for i in (1, 2, 4):
            code = f"00000{i}.XSHE"
            bars = bars_from_csv(code, "1m")
            lf = bars[-1]["frame"]
            ff = TimeFrame.first_min_frame(lf, FrameType.MIN1)
            ranges_1m["cache_start"] = ff
            ranges_1m["cache_stop"] = lf

            bars_ = bars[bars["frame"] >= ff]
            await Stock.cache_bars(code, FrameType.MIN1, bars_)

        # fill in influxdb
        await self.client.drop_measurement("stock_bars_1d")
        await self.client.drop_measurement("stock_bars_1m")
        await self.client.drop_measurement("stock_bars_30m")
        await self.client.drop_measurement("stock_bars_1w")

        for ranges, ft in zip(
            (ranges_1m, ranges_30m, ranges_1d, ranges_1w),
            (FrameType.MIN1, FrameType.MIN30, FrameType.DAY, FrameType.WEEK),
        ):
            for code in (1, 2, 4):
                code = f"00000{code}.XSHE"
                bars = bars_from_csv(code, ft.value)
                lf = bars[-1]["frame"]
                ff = bars[0]["frame"]
                ranges["db_start"] = ff
                ranges["db_stop"] = lf

                bars = numpy_append_fields(
                    bars, "code", [code] * len(bars), [("code", "O")]
                )
                await Stock.persist_bars(ft, bars)

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

    def test_choose(self):
        codes = set(Stock.choose())
        exp = {"000001.XSHE", "300001.XSHE", "600000.XSHG", "000001.XSHG"}
        self.assertSetEqual(exp, codes)

        # 允许ST
        codes = set(Stock.choose(exclude_st=False))
        exp = {
            "000001.XSHE",
            "000005.XSHE",
            "300001.XSHE",
            "600000.XSHG",
            "000007.XSHE",
            "000001.XSHG",
        }
        self.assertSetEqual(exp, codes)

        # 允许科创板
        codes = set(Stock.choose(exclude_688=False))
        exp = {
            "000001.XSHE",
            "300001.XSHE",
            "600000.XSHG",
            "688001.XSHG",
            "000001.XSHG",
        }
        self.assertSetEqual(exp, codes)

        # stock only
        codes = set(Stock.choose(types=["stock"]))
        exp = {"000001.XSHE", "300001.XSHE", "600000.XSHG"}
        self.assertSetEqual(exp, codes)

        # index only
        codes = set(Stock.choose(types=["index"]))
        exp = {"000001.XSHG"}
        self.assertSetEqual(exp, codes)

        # 排除创业板
        codes = set(Stock.choose(exclude_300=True))
        exp = {
            "000001.XSHE",
            "000001.XSHG",
            "600000.XSHG",
        }
        self.assertSetEqual(exp, codes)

    async def test_choose_cyb(self):
        self.assertListEqual(["300001.XSHE"], Stock.choose_cyb())

    def test_choose_kcb(self):
        self.assertListEqual(["688001.XSHG"], Stock.choose_kcb())

    def test_fuzzy_match(self):
        exp = set(["600000.XSHG"])
        self.assertSetEqual(exp, set(Stock.fuzzy_match("600").keys()))

        exp = set(["000001.XSHE", "600000.XSHG"])
        self.assertSetEqual(exp, set(Stock.fuzzy_match("P").keys()))

        exp = set(["000001.XSHE"])
        self.assertSetEqual(exp, set(Stock.fuzzy_match("平").keys()))

    async def test_save_securities(self):
        stocks = [
            ("000001.XSHE", "平安银行", "PAYH", "1991-04-03", "2200-01-01", "stock"),
            ("000001.XSHG", "上证指数", "SZZS", "1991-07-15", "2200-01-01", "index"),
            ("000406.XSHE", "石油大明", "SYDM", "1996-06-28", "2006-04-20", "stock"),
            ("000005.XSHE", "ST星源", "STXY", "1990-12-10", "2200-01-01", "stock"),
            ("300001.XSHE", "特锐德", "TRD", "2009-10-30", "2200-01-01", "stock"),
            ("600000.XSHG", "浦发银行", "PFYH", "1999-11-10", "2200-01-01", "stock"),
            ("688001.XSHG", "华兴源创", "HXYC", "2019-07-22", "2200-01-01", "stock"),
            ("000007.XSHE", "*ST全新", "*STQX", "1992-04-13", "2200-01-01", "stock"),
        ]

        await Stock.save_securities(stocks)

        # make sure no duplicate
        await Stock.save_securities(stocks)
        stocks = await Stock.load_securities()
        self.assertEqual(len(stocks), 8)
        self.assertEqual(stocks[0]["ipo"], datetime.date(1991, 4, 3))

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

    async def test_get_cached_bars(self):
        """cache_bars, cache_unclosed_bars are tested also"""
        # 1. end < ff, 返回空数组
        code = "000001.XSHE"
        ff = arrow.get(ranges_1m["cache_start"])
        lf = arrow.get(ranges_1m["cache_stop"])
        tm = ff.shift(minutes=-1).naive
        bars = await Stock._get_cached_bars(
            "000001.XSHE", tm, 10, FrameType.MIN1, unclosed=True
        )
        self.assertEqual(len(bars), 0)

        # 2.1 end 刚大于 ff, unclosed为 False 只返回第一个bar
        tm = ff.shift(minutes=15).naive
        bars = await Stock._get_cached_bars(
            "000001.XSHE", tm, 10, FrameType.MIN15, unclosed=False
        )
        self.assertEqual(len(bars), 1)
        self.assertEqual(datetime.datetime(2022, 2, 10, 9, 45), bars["frame"][0])

        # 2.2 end 刚大于 ff, unclosed为 True 返回两个bar
        bars = await Stock._get_cached_bars("000001.XSHE", tm, 10, FrameType.MIN15)
        self.assertEqual(len(bars), 2)
        self.assertEqual(datetime.datetime(2022, 2, 10, 9, 46), bars["frame"][1])

        # 3.1 end > last frame in cache, unclosed = True
        tm = lf.shift(minutes=15).naive
        bars = await Stock._get_cached_bars("000001.XSHE", tm, 10, FrameType.MIN15)

        self.assertEqual(3, len(bars))
        self.assertEqual(datetime.datetime(2022, 2, 10, 10, 6), bars["frame"][-1])

        # 3.2 check other fields are equal
        m1_bars = bars_from_csv(code, "1m", 66, 101)
        expected = Stock.resample(m1_bars, FrameType.MIN1, FrameType.MIN15)
        assert_bars_equal(expected, bars)

        # 3.3 let unclosed = False
        bars = await Stock._get_cached_bars(
            "000001.XSHE", tm, 10, FrameType.MIN15, unclosed=False
        )
        self.assertEqual(2, len(bars))
        assert_bars_equal(expected[:-1], bars)

        # 4. if n < len(cached bars)
        ## 4.1 included unclosed
        bars = await Stock._get_cached_bars(code, tm, 2, FrameType.MIN15)
        self.assertEqual(2, len(bars))
        assert_bars_equal(expected[1:], bars)

        ## 4.2 not include unclosed
        bars = await Stock._get_cached_bars(
            code, tm, 2, FrameType.MIN15, unclosed=False
        )
        self.assertEqual(2, len(bars))
        assert_bars_equal(expected[:-1], bars)

        # 5. FrameType == min1, no resample
        tm = lf.naive
        bars = await Stock._get_cached_bars(code, tm, 36, FrameType.MIN1)
        self.assertEqual(36, len(bars))

        # 6. 当cache为空时，应该返回空数组
        await Stock.reset_cache()
        await cache._sys_.delete("second_data_source")

        bars = await Stock._get_cached_bars(
            "000001.XSHE", lf.naive, 10, FrameType.MIN1, unclosed=True
        )
        self.assertEqual(bars.size, 0)

        bars = await Stock._get_cached_bars(
            "000001.XSHE", lf.naive, 10, FrameType.DAY, unclosed=False
        )
        self.assertEqual(len(bars), 0)

    async def test_get_bars(self):
        code = "000001.XSHE"
        ft = FrameType.MIN1

        lf = arrow.get(ranges_1m["cache_stop"])
        ff = arrow.get(ranges_1m["cache_start"])

        # 1. end is None, 取当前时间作为end.
        with mock.patch.object(
            arrow, "now", return_value=arrow.get(lf.shift(minutes=1))
        ):
            end = None
            n = 1
            bars = await Stock.get_bars(code, n, ft, end, fq=False)
            exp = bars_from_csv(code, "1m", 101, 101)
            assert_bars_equal(exp, bars)

        # 2. end < ff，仅从persistent中取
        tm = ff.shift(minutes=-1).naive
        n = 2

        bars = await Stock.get_bars(code, n, ft, tm, fq=False)
        expected = bars_from_csv(code, "1m", 64, 65)
        assert_bars_equal(expected, bars)

        # 3.1 end > ff，从persistent和cache中取,不包含unclosed
        ft = FrameType.MIN30
        tm = ff.shift(minutes=30).naive
        n = 3
        from_persist = bars_from_csv(code, "30m", 100, 101)
        from_cache = bars_from_csv(code, "1m", 66, 96)
        from_cache = Stock.resample(from_cache, FrameType.MIN1, ft)
        bars = await Stock.get_bars(code, n, ft, tm, fq=False, unclosed=False)

        self.assertEqual(n, bars.size)
        assert_bars_equal(from_persist, bars[:2])
        assert_bars_equal(from_cache[:-1], bars[2:])

        # 3.2 end > ff, 从persistent和cache中取,包含unclosed
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

    async def test_batch_cache_bars(self):
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
        start = ranges_30m["db_start"]
        end = ranges_30m["db_stop"]

        bars = await Stock._get_persisted_bars(code, FrameType.MIN30, begin=start)

        self.assertEqual(bars.size, 100)
        expected = bars_from_csv(code, "30m")
        assert_bars_equal(expected, bars)

        # test with end
        bars = await Stock._get_persisted_bars(
            "000001.XSHE", FrameType.MIN30, n=10, begin=start, end=end
        )

        self.assertEqual(len(bars), 10)
        expected = bars_from_csv(code, "30m")[-10:]
        assert_bars_equal(expected, bars)

        # test with FrameType.DAY, to see if it can handle date correctly
        ft = FrameType.DAY
        start = ranges_1d["db_start"]
        end = ranges_1d["db_stop"]

        actual = await Stock._get_persisted_bars(code, ft, start, end)
        expected = bars_from_csv(code, "1d")
        assert_bars_equal(expected, actual)

        await self.client.drop_measurement("stock_bars_1d")
        # test with multiple codes
        data = {
            code: bars_from_csv(code, "1d", 90)
            for code in ["000001.XSHE", "000002.XSHE"]
        }

        start = data["000001.XSHE"][0][0]
        await Stock.persist_bars(FrameType.DAY, data)
        actual = await Stock._batch_get_persisted_bars(
            list(data.keys()), FrameType.DAY, start, end=end
        )
        for code in data.keys():
            assert_bars_equal(data[code], actual[code])

    async def test_batch_get_persisted_bars(self):
        codes = []
        start = ranges_30m["db_start"]
        end = ranges_30m["db_stop"]
        ft = FrameType.MIN30

        data = await Stock._batch_get_persisted_bars(codes, ft, begin=start)
        self.assertTrue(isinstance(data, dict))
        self.assertEqual(3, len(data))

        bars = data["000001.XSHE"]
        self.assertEqual(bars.size, 100)
        expected = bars_from_csv("000001.XSHE", "30m")
        assert_bars_equal(expected, bars)

        codes = ["000001.XSHE", "000002.XSHE"]
        data = await Stock._batch_get_persisted_bars(codes, ft, begin=start)

        self.assertEqual(2, len(data))
        expected = bars_from_csv("000002.XSHE", "30m")
        assert_bars_equal(expected, data["000002.XSHE"])

        # test with end
        data = await Stock._batch_get_persisted_bars(codes, ft, begin=start, end=end)
        expected = bars_from_csv("000001.XSHE", "30m")
        assert_bars_equal(expected, data["000001.XSHE"])

        # test with n
        data = await Stock._batch_get_persisted_bars(
            codes, ft, begin=start, end=end, n=10
        )
        expected = bars_from_csv("000001.XSHE", "30m")[-10:]
        assert_bars_equal(expected, data["000001.XSHE"])

    async def test_get_persisted_trade_price_limits(self):
        measurement = "stock_bars_1d"

        # fill in data
        start = datetime.date(2022, 1, 10)
        end = ranges_1d["db_stop"]

        trade_limits = np.array(
            [
                (datetime.date(2022, 1, 10), 18.92, 15.48),
                (datetime.date(2022, 1, 11), 18.91, 15.47),
                (datetime.date(2022, 1, 12), 19.15, 15.67),
                (datetime.date(2022, 1, 13), 18.7, 15.3),
                (datetime.date(2022, 1, 14), 18.68, 15.28),
                (datetime.date(2022, 1, 17), 17.96, 14.7),
                (datetime.date(2022, 1, 18), 17.84, 14.6),
                (datetime.date(2022, 1, 19), 18.17, 14.87),
                (datetime.date(2022, 1, 20), 18.15, 14.85),
                (datetime.date(2022, 1, 21), 19.06, 15.6),
                (datetime.date(2022, 1, 24), 19.09, 15.62),
                (datetime.date(2022, 1, 25), 18.92, 15.48),
                (datetime.date(2022, 1, 26), 18.54, 15.17),
                (datetime.date(2022, 1, 27), 18.32, 14.99),
                (datetime.date(2022, 1, 28), 17.93, 14.67),
                (datetime.date(2022, 2, 7), 17.41, 14.25),
                (datetime.date(2022, 2, 8), 18.03, 14.75),
            ],
            dtype=[("frame", "O"), ("high_limit", "<f4"), ("low_limit", "<f4")],
        )

        await self.client.save(
            trade_limits,
            measurement,
            time_key="frame",
            global_tags={"code": "000001.XSHE"},
        )

        result = await Stock._get_persisted_trade_price_limits(
            "000001.XSHE", start, end
        )

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
                (datetime.date(2022, 1, 6), 18.83, 15.41),
                (datetime.date(2022, 1, 7), 18.83, 15.41),
            ],
            dtype=[
                ("frame", "O"),
                ("high_limit", "<f8"),
                ("low_limit", "<f8"),
            ],
        )

        for col in ["high_limit", "low_limit"]:
            np.testing.assert_array_almost_equal(expected[col], actual[col])

        np.testing.assert_array_equal(expected["frame"], actual["frame"])

        # 取当天的限价
        dt = arrow.get("2022-01-10").date()
        field_high = f"{code}.high_limit"
        field_low = f"{code}.low_limit"

        await cache._security_.hmset(
            TRADE_PRICE_LIMITS, field_high, 19.83, field_low, 17.32
        )
        with mock.patch("arrow.now", return_value=dt):
            actual = await Stock.get_trade_price_limits(code, start, dt)
            expected = np.array(
                [
                    (datetime.date(2022, 1, 6), 18.83, 15.41),
                    (datetime.date(2022, 1, 7), 18.83, 15.41),
                    (dt, 19.83, 17.32),
                ],
                dtype=[
                    ("frame", "O"),
                    ("high_limit", "<f8"),
                    ("low_limit", "<f8"),
                ],
            )
            for col in ["high_limit", "low_limit"]:
                np.testing.assert_array_almost_equal(expected[col], actual[col])

    async def test_batch_get_cached_bars(self):
        codes = ["000001.XSHE", "000002.XSHE", "000004.XSHE"]
        stop = ranges_1m["cache_stop"]

        ft = FrameType.MIN1
        unclosed = True

        for unclosed in (True, False):
            # doesn't matter for 1min
            result = await Stock._batch_get_cached_bars(codes, stop, 10, ft, unclosed)
            self.assertEqual(10, result["000001.XSHE"].size)
            self.assertEqual(10, result["000004.XSHE"].size)
            self.assertEqual(stop, result["000001.XSHE"][-1]["frame"])
            exp_start = arrow.get(stop).shift(minutes=-9).naive
            self.assertEqual(exp_start, result["000001.XSHE"][0]["frame"])

        # if some code contains no bars
        codes = ["000001.XSHE", "000002.XSHE", "000003.XSHE"]
        result = await Stock._batch_get_cached_bars(codes, stop, 10, ft)
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

    async def test_batch_get_bars(self):
        codes = ["000001.XSHE", "000002.XSHE", "000003.XSHE"]

        end = arrow.get("2022-02-10 10:06:00").naive
        ft = FrameType.MIN30
        bars = await Stock.batch_get_bars(codes, 10, ft, end, fq=False, unclosed=True)

        self.assertEqual(3, len(bars))
        self.assertEqual(10, bars["000001.XSHE"].size)
        self.assertEqual(10, bars["000002.XSHE"].size)
        self.assertEqual(0, bars["000003.XSHE"].size)

        code = "000001.XSHE"
        from_persist = bars_from_csv(code, "30m", 94, 101)
        from_cache = bars_from_csv(code, "1m", 66, 101)
        from_cache = Stock.resample(from_cache, FrameType.MIN1, ft)

        assert_bars_equal(from_persist, bars[code][:-2])
        assert_bars_equal(from_cache, bars[code][8:])

        # all in cache
        bars = await Stock.batch_get_bars(codes, 1, ft, end, fq=False)
        self.assertEqual(3, len(bars))
        assert_bars_equal(from_cache[:1], bars[code])

        # fq = true, unclosed = true
        bars = await Stock.batch_get_bars(codes, 10, ft, end, fq=True)
        self.assertEqual(3, len(bars))
        self.assertEqual(10, bars[code].size)

        # 周线
        end = arrow.get("2021-12-24").date()
        bars = await Stock.batch_get_bars(codes, 8, FrameType.WEEK, end, fq=False)
        from_persist = bars_from_csv(code, "1w", 94, 101)
        self.assertEqual(3, len(bars))
        self.assertEqual(8, bars[code].size)

    async def test_save_trade_price_limits(self):
        limits = np.array(
            [
                (datetime.date(2022, 1, 6), "000001.XSHE", 18.83, 15.41),
                (datetime.date(2022, 1, 7), "000002.XSHE", 18.83, 15.41),
            ],
            dtype=[
                ("frame", "O"),
                ("code", "O"),
                ("high_limit", "<f8"),
                ("low_limit", "<f8"),
            ],
        )

        await Stock.save_trade_price_limits(limits, True)

        start = datetime.date(2022, 1, 6)
        end = datetime.date(2022, 1, 6)

        with mock.patch("arrow.now", return_value=start):
            actual = await Stock.get_trade_price_limits("000001.XSHE", start, end)
            self.assertAlmostEqual(18.83, actual[0]["high_limit"])

        await Stock.save_trade_price_limits(limits, False)
        with mock.patch("arrow.now", return_value=datetime.date(1900, 1, 1)):
            actual = await Stock.get_trade_price_limits("000001.XSHE", start, end)
            self.assertAlmostEqual(18.83, actual[0]["high_limit"])

    async def test_get_bars_in_range(self):
        code = "000001.XSHE"
        start = datetime.datetime(2022, 2, 9, 10)
        end = datetime.datetime(2022, 2, 10, 10, 6)

        bars = await Stock.get_bars_in_range(code, FrameType.MIN30, start, end)
        self.assertEqual(10, len(bars))
        self.assertEqual(datetime.datetime(2022, 2, 9, 10), bars[0]["frame"])
        self.assertEqual(datetime.datetime(2022, 2, 10, 10, 6), bars[-1]["frame"])

        bars = await Stock.get_bars_in_range(
            code, FrameType.MIN30, start, end, unclosed=False
        )
        self.assertEqual(9, len(bars))
        self.assertEqual(datetime.datetime(2022, 2, 9, 10), bars[0]["frame"])
        self.assertEqual(datetime.datetime(2022, 2, 10, 10), bars[-1]["frame"])

    async def test_batch_get_bars_in_range(self):
        codes = ["000001.XSHE", "000002.XSHE", "000003.XSHE"]
        start = datetime.datetime(2022, 2, 9, 10)
        end = datetime.datetime(2022, 2, 10, 10, 6)

        actual = await Stock.batch_get_bars_in_range(
            codes, FrameType.MIN30, start, end, fq=False
        )
        self.assertEqual(3, len(actual))
        self.assertEqual(10, len(actual["000001.XSHE"]))
        self.assertEqual(0, len(actual["000003.XSHE"]))

        self.assertEqual(
            datetime.datetime(2022, 2, 9, 10), actual["000001.XSHE"][0]["frame"]
        )
        self.assertEqual(
            datetime.datetime(2022, 2, 10, 10, 6), actual["000001.XSHE"][-1]["frame"]
        )

        actual = await Stock.batch_get_bars_in_range(
            codes, FrameType.MIN30, start, end, unclosed=False, fq=False
        )

        self.assertEqual(3, len(actual))
        self.assertEqual(9, len(actual["000001.XSHE"]))
        self.assertEqual(0, len(actual["000003.XSHE"]))
        self.assertEqual(
            datetime.datetime(2022, 2, 9, 10), actual["000001.XSHE"][0]["frame"]
        )
        self.assertEqual(
            datetime.datetime(2022, 2, 10, 10), actual["000001.XSHE"][-1]["frame"]
        )

    async def test_batch_cache_unclosed_bars(self):
        data = {
            "000001.XSHE": bars_from_csv("000001.XSHE", "1d", 101, 101),
        }
        await Stock.batch_cache_unclosed_bars(FrameType.DAY, data)
        data = await Stock.batch_get_bars(
            ["000001.XSHE"], 1, FrameType.DAY, unclosed=True
        )
        self.assertEqual(1, len(data))
        self.assertEqual(datetime.date(2022, 2, 8), data["000001.XSHE"][0]["frame"])

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
