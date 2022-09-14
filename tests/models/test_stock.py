import datetime
import os
import pickle
import unittest
from turtle import end_fill
from unittest import mock

import arrow
import cfg4py
import numpy as np
import pandas as pd
from coretypes import (
    FrameType,
    SecurityType,
    bars_cols,
    bars_dtype,
    bars_with_limit_dtype,
)
from freezegun import freeze_time
from numpy.testing import assert_array_equal

import omicron
from omicron import tf
from omicron.core.constants import TRADE_LATEST_PRICE, TRADE_PRICE_LIMITS_DATE
from omicron.core.errors import BadParameterError
from omicron.dal import cache
from omicron.dal.influx.influxclient import InfluxClient
from omicron.extensions.np import numpy_append_fields
from omicron.models.stock import Stock
from tests import assert_bars_equal, bars_from_csv, init_test_env, read_csv, test_dir

cfg = cfg4py.get_instance()
feb8_0931 = datetime.datetime(2022, 2, 8, 9, 31, 0)
feb8_1500 = datetime.datetime(2022, 2, 8, 15, 0, 0)
feb9_0931 = datetime.datetime(2022, 2, 9, 9, 31)
feb9_1033 = datetime.datetime(2022, 2, 9, 10, 33)


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

            cache_date = bars[-1]["frame"].item().date()
            persist_date = tf.day_shift(cache_date, -1)
            persist_end = tf.combine_time(persist_date, 15)

            cache_bars = bars[bars["frame"] > persist_end]
            persist_bars = bars[bars["frame"] <= persist_end]

            for ft in tf.minute_level_frames[1:]:
                resampled = Stock.resample(cache_bars, FrameType.MIN1, ft)
                lf = resampled["frame"][-1].item()
                if lf == tf.floor(lf, ft):
                    await Stock.cache_bars(code, ft, resampled)
                else:
                    await Stock.cache_bars(code, ft, resampled[:-1])
                    await Stock.cache_unclosed_bars(code, ft, resampled[-1:])

            day_resampled = Stock.resample(cache_bars, FrameType.MIN1, FrameType.DAY)
            await Stock.cache_unclosed_bars(code, FrameType.DAY, day_resampled)

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

        # for issue 23
        code = "600590.XSHG"
        bars = {
            code: np.array(
                [
                    (
                        datetime.date(2014, 9, 9),
                        8.99,
                        9.09,
                        8.8,
                        8.98,
                        10216680.0,
                        9.14967520e07,
                        3.278,
                    ),
                    (
                        datetime.date(2014, 9, 10),
                        8.94,
                        9.16,
                        8.89,
                        8.97,
                        12416833.0,
                        1.11841024e08,
                        3.278,
                    ),
                ],
                dtype=bars_dtype,
            )
        }

        await Stock.persist_bars(FrameType.DAY, bars)
        # end issue 23

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

        return await super().asyncSetUp()

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
        bars = bars_from_csv("000001.XSHE", "1m", feb8_0931, feb8_1500)

        for ft in [
            FrameType.MIN5,
            FrameType.MIN15,
            FrameType.MIN30,
            FrameType.MIN60,
            FrameType.DAY,
        ]:
            actual = Stock.resample(bars, FrameType.MIN1, ft)
            exp = bars_from_csv(
                "000001.XSHE",
                ft.value,
                datetime.datetime(2022, 2, 8),
                datetime.datetime(2022, 2, 9),
            )
            assert_bars_equal(exp, actual)

        # resample when input bars can not be evenly divided
        actual = Stock.resample(bars[:-1], FrameType.MIN1, FrameType.MIN15)
        exp = bars_from_csv(
            "000001.XSHE",
            "15m",
            datetime.datetime(2022, 2, 8),
            datetime.datetime(2022, 2, 9),
        )

        assert_bars_equal(exp[:-1], actual[:-1])
        self.assertEqual(
            datetime.datetime(2022, 2, 8, 14, 59), actual[-1]["frame"].item()
        )

        # 输入错误检查
        bars[0]["frame"] = datetime.datetime(2021, 4, 27, 9, 35)
        try:
            Stock.resample(bars, FrameType.MIN1, FrameType.MIN5)
        except ValueError as e:
            self.assertEqual(str(e), "resampling from 1min must start from 9:31")

    @freeze_time("2022-02-09 10:33:00")
    async def test_get_cached_bars_n(self):
        """cache_bars, cache_unclosed_bars are tested also"""
        now = arrow.get("2022-02-09 10:33:00").naive
        # 1. end < ff, 返回空数组
        code = "000001.XSHE"
        ff = arrow.get("2022-02-09 09:31:00").naive
        lf = arrow.get("2022-02-09 10:33:00").naive

        tm = datetime.datetime(2022, 2, 9, 9, 30)
        bars = await Stock._get_cached_bars_n("000001.XSHE", 10, FrameType.MIN1, end=tm)
        self.assertEqual(len(bars), 0)

        # 2 end < lf
        tm = tf.floor(now, FrameType.MIN5)
        bars = await Stock._get_cached_bars_n("000001.XSHE", 15, FrameType.MIN5, end=tm)
        self.assertEqual(len(bars), 12)
        self.assertEqual(
            datetime.datetime(2022, 2, 9, 10, 30), bars["frame"][-1].item()
        )

        # 2.0 end < lf 且end不在边界上,取到上一个边界为止
        tm = arrow.get("2022-02-09 10:29:00").naive
        bars = await Stock._get_cached_bars_n("000001.XSHE", 15, FrameType.MIN5, end=tm)
        self.assertEqual(
            datetime.datetime(2022, 2, 9, 10, 25), bars["frame"][-1].item()
        )

        # 3.0 end == lf 并且结果的起始时间不早于lf
        bars = await Stock._get_cached_bars_n("000001.XSHE", 12, FrameType.MIN5, end=lf)
        self.assertEqual(12, len(bars))
        self.assertEqual(lf, bars["frame"][-1].item())
        self.assertEqual(datetime.datetime(2022, 2, 9, 9, 40), bars["frame"][0].item())

        # 3.1 end == lf 并且结果的起始时间早于lf
        bars = await Stock._get_cached_bars_n("000001.XSHE", 20, FrameType.MIN5, end=lf)
        self.assertEqual(13, len(bars))
        self.assertEqual(lf, bars["frame"][-1].item())
        self.assertEqual(datetime.datetime(2022, 2, 9, 9, 35), bars["frame"][0].item())

        # 3.1 end > lf in cache
        tm = datetime.datetime(2022, 2, 9, 10, 48)
        bars = await Stock._get_cached_bars_n("000001.XSHE", 10, FrameType.MIN5, end=tm)

        self.assertEqual(8, len(bars))
        self.assertEqual(
            datetime.datetime(2022, 2, 9, 10, 33), bars["frame"][-1].item()
        )
        self.assertEqual(datetime.datetime(2022, 2, 9, 10), bars["frame"][0].item())

        # 4 check 1m bars
        bars = await Stock._get_cached_bars_n(code, 100, FrameType.MIN1, end=lf)
        self.assertEqual(63, len(bars))
        self.assertEqual(datetime.datetime(2022, 2, 9, 9, 31), bars["frame"][0].item())
        self.assertEqual(
            datetime.datetime(2022, 2, 9, 10, 33), bars["frame"][-1].item()
        )
        # check deserialize is ok
        np.testing.assert_array_equal(feb9_0931, bars["frame"][0].item())

        # 5 check 1d bars
        bars = await Stock._get_cached_bars_n(
            "000001.XSHE", 2, FrameType.DAY, end=lf.date()
        )
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
        np.testing.assert_array_equal(exp, bars)

        # 6. check end is None
        with freeze_time("2022-02-09 10:33:00"):
            bars = await Stock._get_cached_bars_n("000001.XSHE", 10, FrameType.MIN1)
            self.assertEqual(bars.size, 10)

        # 7. 当cache为空时，应该返回空数组
        await Stock.reset_cache()

        bars = await Stock._get_cached_bars_n("000001.XSHE", 10, FrameType.MIN1, end=lf)
        self.assertEqual(bars.size, 0)

    @freeze_time("2022-02-09 10:33:00")
    async def test_get_bars(self):
        code = "000001.XSHE"
        ft = FrameType.MIN1

        # 1. end is None, 取当前时间作为end.
        end = None
        n = 1
        bars = await Stock.get_bars(code, n, ft, end, fq=False)
        exp = bars_from_csv(code, "1m", feb9_1033)
        assert_bars_equal(exp, bars)

        # 2. end < ff，仅从persistent中取
        tm = arrow.get(feb9_0931).shift(minutes=-1).naive
        n = 2

        bars = await Stock.get_bars(code, n, ft, tm, fq=False)
        expected = bars_from_csv(code, "1m", end=feb8_1500)[-2:]
        assert_bars_equal(expected, bars)

        # 3.1 end > ff，从persistent和cache中取,不包含unclosed
        ft = FrameType.MIN30
        n = 2
        from_persist = bars_from_csv(code, "30m")[-1:]
        end = arrow.get("2022-02-09 10:00:00").naive
        from_cache = bars_from_csv(code, "1m", feb9_0931, end)
        from_cache = Stock.resample(from_cache, FrameType.MIN1, ft)
        bars = await Stock.get_bars(code, n, ft, end, fq=False, unclosed=False)

        self.assertEqual(2, bars.size)
        assert_bars_equal(from_persist, bars[:1])
        assert_bars_equal(from_cache, bars[1:])

        # 3.2 end < lf, 从persistent和cache中取，要求包含unclosed，但实际不应该返回unclosed
        n = 3
        bars = await Stock.get_bars(code, n, ft, end, unclosed=True)
        self.assertEqual(n, bars.size)

        from_persist = bars_from_csv(code, "30m")[-2:]
        assert_bars_equal(from_persist, bars[:2])
        assert_bars_equal(from_cache, bars[2:])

        # 3.3 end == lf, 从persistent和cache中取,不包含unclosed
        n = 3
        tm = datetime.datetime(2022, 2, 9, 10, 33)
        bars = await Stock.get_bars(code, n, ft, tm, unclosed=False)
        from_persist = bars_from_csv(code, "30m", feb8_1500)
        from_cache = bars_from_csv(code, "1m", feb9_0931, feb9_1033)
        from_cache = Stock.resample(from_cache, FrameType.MIN1, ft)

        self.assertEqual(n, bars.size)
        assert_bars_equal(from_persist[-1:], bars[:1])
        assert_bars_equal(from_cache[:-1], bars[1:])

        # 3.3 end == lf, 从persistent和cache中取,包含unclosed
        n = 4
        bars = await Stock.get_bars(code, n, ft, feb9_1033, fq=False, unclosed=True)
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
        bars = await Stock.get_bars(code, n, ft, end=feb9_1033.date(), fq=False)
        exp = np.array(
            [
                (
                    datetime.date(2022, 2, 8),
                    16.3,
                    16.97,
                    16.28,
                    16.83,
                    1.754697e08,
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
                "omicron.models.stock.Stock._get_cached_bars_n",
                side_effect=ValueError("test"),
            ):
                await Stock.get_bars(code, n, ft, end=feb9_1033.date(), fq=False)

        # with empty cache
        with mock.patch(
            "omicron.models.stock.Stock._get_cached_bars_n",
            return_value=np.array([], dtype=bars_dtype),
        ):
            for ft, n in zip((FrameType.MIN30, FrameType.DAY), (8, 2)):
                bars = await Stock.get_bars(code, n, ft, end=feb9_1033)
                self.assertEqual(bars.size, n)

    @freeze_time("2022-01-10 09:34:00")
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
                    )
                ],
                dtype=bars_dtype,
            ),
        }

        await Stock.reset_cache()
        await Stock.batch_cache_bars(FrameType.MIN1, data)

        bars = await Stock._get_cached_bars_n(
            "000001.XSHE", 4, FrameType.MIN1, datetime.datetime(2022, 1, 10, 9, 34)
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

        bars = await Stock._get_cached_bars_n(
            "000004.XSHE", 1, FrameType.MIN1, datetime.datetime(2022, 1, 10, 9, 34)
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

    async def test_get_persisted_bars_in_range(self):
        code = "000001.XSHE"
        start = arrow.get("2022-02-08 10:00:00").naive
        end = arrow.get("2022-02-08 15:00:00")

        # 2/8 10:00 ~ 2/8 15:00
        expected = bars_from_csv(code, "30m")

        bars = await Stock._get_persisted_bars_in_range(
            code, FrameType.MIN30, start, end
        )
        assert_bars_equal(expected, bars)

        bars = await Stock._get_persisted_bars_in_range(
            code, FrameType.MIN30, start, end.shift(minutes=-30)
        )
        assert_bars_equal(expected[:-1], bars)

        # 停牌区间，没有数据
        bars = await Stock._get_persisted_bars_in_range(
            code, FrameType.DAY, datetime.date(2022, 3, 8), datetime.date(2022, 3, 20)
        )
        self.assertEqual(bars.size, 0)

    async def test_get_persisted_bars_n(self):
        code = "000001.XSHE"
        start = arrow.get("2022-02-08 10:00:00").naive
        end = arrow.get("2022-02-08 15:00:00")

        with freeze_time("2022-02-08 15:00:00"):
            bars = await Stock._get_persisted_bars_n(code, FrameType.MIN30, 10)

            self.assertEqual(bars.size, 8)
            expected = bars_from_csv(code, "30m")
            assert_bars_equal(expected, bars)

        # test with end
        bars = await Stock._get_persisted_bars_n(
            "000001.XSHE", FrameType.MIN30, 8, end=end.shift(minutes=-5)
        )

        # db conttains only 7
        self.assertEqual(len(bars), 7)
        expected = bars_from_csv(code, "30m")
        assert_bars_equal(expected[:-1], bars)

        # test with FrameType.DAY, to see if it can handle date correctly
        ft = FrameType.DAY
        end = arrow.get("2022-02-08").date()

        actual = await Stock._get_persisted_bars_n(code, ft, 2, end)
        expected = bars_from_csv(code, "1d")
        assert_bars_equal(expected, actual)

        # 如果某段时间停牌，则向前取数据，直到取到n条
        code = "000002.XSHE"
        actual = await Stock._get_persisted_bars_n(
            code, ft, 40, datetime.date(2022, 3, 8)
        )
        expected = bars_from_csv(code, "1d")
        assert_bars_equal(expected[-2:], actual[-2:])
        self.assertEqual(len(actual), 40)

    async def test_batch_get_persisted_bars_in_range(self):
        start = datetime.date(2022, 2, 7)
        end = datetime.date(2022, 2, 8)

        codes = ["000001.XSHE", "000002.XSHE", "000004.XSHE"]
        # test codes is None
        barss = await Stock._batch_get_persisted_bars_in_range(
            codes, FrameType.DAY, start, end
        )
        np.testing.assert_array_equal(barss.code.unique(), codes)
        np.testing.assert_array_equal(
            barss.frame.values,
            np.array(
                [
                    "2022-02-07T00:00:00.000000000",
                    "2022-02-08T00:00:00.000000000",
                    "2022-02-07T00:00:00.000000000",
                    "2022-02-08T00:00:00.000000000",
                    "2022-02-07T00:00:00.000000000",
                    "2022-02-08T00:00:00.000000000",
                ],
                dtype="datetime64[ns]",
            ),
        )

        np.testing.assert_array_almost_equal(
            barss.close,
            np.array(
                [
                    16.38999939,
                    16.82999992,
                    20.90999985,
                    20.87999916,
                    21.52000046,
                    22.45000076,
                ]
            ),
            2,
        )

        start = datetime.datetime(2022, 2, 8, 10)
        end = datetime.datetime(2022, 2, 9, 14, 30)
        barss = await Stock._batch_get_persisted_bars_in_range(
            codes[:2], FrameType.MIN30, start, end
        )

        bars1 = barss[barss.code == "000001.XSHE"]
        np.testing.assert_array_equal(
            bars1.frame.values,
            np.array(
                [
                    "2022-02-08T10:00:00.000000000",
                    "2022-02-08T10:30:00.000000000",
                    "2022-02-08T11:00:00.000000000",
                    "2022-02-08T11:30:00.000000000",
                    "2022-02-08T13:30:00.000000000",
                    "2022-02-08T14:00:00.000000000",
                    "2022-02-08T14:30:00.000000000",
                    "2022-02-08T15:00:00.000000000",
                ],
                dtype="datetime64[ns]",
            ),
        )
        np.testing.assert_array_almost_equal(
            bars1.close,
            np.array(
                [
                    16.82999992,
                    16.89999962,
                    16.93000031,
                    16.73999977,
                    16.73999977,
                    16.72999954,
                    16.82999992,
                    16.82999992,
                ]
            ),
            2,
        )

        # check size limit
        with mock.patch.object(cfg.influxdb, "max_query_size", 5):
            with self.assertRaises(BadParameterError):
                barss = await Stock._batch_get_persisted_bars_in_range(
                    codes, FrameType.DAY, start, end
                )

    async def test_batch_get_persisted_bars_n(self):
        await self.client.drop_measurement("stock_bars_1d")
        # test with multiple codes
        data = {
            code: bars_from_csv(code, "1d") for code in ["000001.XSHE", "000002.XSHE"]
        }

        await Stock.persist_bars(FrameType.DAY, data)

        end = arrow.get("2022-02-08").date()
        actual = await Stock._batch_get_persisted_bars_n(
            list(data.keys()), FrameType.DAY, 2, end=end
        )
        for code, group in actual.groupby("code"):
            assert_bars_equal(
                data[code][-2:],
                group.to_records(index=False)[list(bars_dtype.names)].astype(
                    bars_dtype
                ),
            )

        # 查询数量限制
        with mock.patch.object(cfg.influxdb, "max_query_size", 5):
            with self.assertRaises(BadParameterError):
                actual = await Stock._batch_get_persisted_bars_n(
                    list(data.keys()), FrameType.MIN30, 10, end=end
                )

        # 模拟其中有停牌的情况，以及不存在的code
        end = arrow.get("2022-02-09 15:00:00").naive
        actual = await Stock._batch_get_persisted_bars_n(
            list(data.keys()) + ["000008.XSHE"], FrameType.DAY, 10, end=end
        )
        for code, group in actual.groupby("code"):
            assert_bars_equal(
                data[code][-10:],
                group.to_records(index=False)[list(bars_dtype.names)].astype(
                    bars_dtype
                ),
            )

        await self.client.drop_measurement("stock_bars_30m")
        data = {
            code: bars_from_csv(code, "30m") for code in ["000001.XSHE", "000002.XSHE"]
        }
        await Stock.persist_bars(FrameType.MIN30, data)

        # 30分钟
        end = arrow.get("2022-02-09 15:00:00").naive
        actual = await Stock._batch_get_persisted_bars_n(
            list(data.keys()) + ["000008.XSHE"], FrameType.MIN30, 10, end=end
        )
        for code, group in actual.groupby("code"):
            assert_bars_equal(
                data[code][-10:],
                group.to_records(index=False)[list(bars_dtype.names)].astype(
                    bars_dtype
                ),
            )

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

    @freeze_time("2022-02-09 15:00:00")
    async def test_batch_get_day_level_bars_in_range(self):
        codes = ["000001.XSHE", "000002.XSHE", "000004.XSHE"]

        async for code, bars in Stock.batch_get_day_level_bars_in_range(
            codes,
            FrameType.DAY,
            arrow.get("2022-02-08").date(),
            arrow.get("2022-02-09").date(),
        ):
            min_bars = bars_from_csv(code, "1m")
            min_bars = min_bars[min_bars["frame"] >= datetime.datetime(2022, 2, 9)]
            cached = Stock.resample(min_bars, FrameType.MIN1, FrameType.DAY)
            persisted = bars_from_csv(code, "1d")
            persisted = persisted[persisted["frame"] >= datetime.datetime(2022, 2, 8)]

            exp = np.concatenate((persisted, cached))
            assert_bars_equal(exp, bars)

        # when cache contains no data
        await Stock.reset_cache()
        async for code, bars in Stock.batch_get_day_level_bars_in_range(
            codes,
            FrameType.DAY,
            arrow.get("2022-02-08").date(),
            arrow.get("2022-02-09").date(),
        ):
            min_bars = bars_from_csv(code, "1m")
            min_bars = min_bars[min_bars["frame"] >= datetime.datetime(2022, 2, 9)]
            cached = Stock.resample(min_bars, FrameType.MIN1, FrameType.DAY)
            persisted = bars_from_csv(code, "1d")
            persisted = persisted[persisted["frame"] >= datetime.datetime(2022, 2, 8)]

            assert_bars_equal(persisted, bars)

    @freeze_time("2022-02-09 10:33:00")
    async def test_batch_get_min_level_bars_in_range(self):
        codes = ["000001.XSHE", "000002.XSHE", "000004.XSHE"]

        start = datetime.datetime(2022, 2, 8, 10)
        feb9_0930 = datetime.datetime(2022, 2, 9, 9, 30)
        feb9_1030 = datetime.datetime(2022, 2, 9, 10, 30)

        # end < cache_start and now.date() > end.date()
        end_ = datetime.datetime(2022, 1, 28, 15)
        start_ = datetime.datetime(2022, 1, 18, 15)
        codes_ = ["000002.XSHE"]
        with freeze_time("2022-08-09 09:31:00"):
            async for code, bars in Stock.batch_get_min_level_bars_in_range(
                codes_, FrameType.MIN30, start_, end_
            ):
                exp = bars_from_csv(code, "30m", start_, end_)
                assert_bars_equal(exp, bars)

        # end < cache_start
        async for code, bars in Stock.batch_get_min_level_bars_in_range(
            codes, FrameType.MIN30, start, feb9_1030
        ):
            exp = bars_from_csv(code, "30m", start, feb9_1030)
            m1_bars = bars_from_csv(
                code, "1m", datetime.datetime(2022, 2, 9, 9, 31), feb9_1030
            )
            m30 = Stock.resample(m1_bars, FrameType.MIN1, FrameType.MIN30)
            exp = np.concatenate((exp, m30))
            assert_bars_equal(exp, bars)

            # check fq called
            with mock.patch("omicron.models.stock.Stock.qfq") as mock_qfq:
                async for code, bars in Stock.batch_get_min_level_bars_in_range(
                    codes, FrameType.MIN30, start, feb9_1033
                ):
                    pass

                self.assertEqual(len(codes), mock_qfq.call_count)

                # fq = False，这样mock对象仍然只被调用一次
                async for code, bars in Stock.batch_get_min_level_bars_in_range(
                    codes, FrameType.MIN30, start, feb9_1033, fq=False
                ):
                    pass

                self.assertEqual(len(codes), mock_qfq.call_count)

        # test batch size
        with mock.patch.object(cfg.influxdb, "max_query_size", 212):
            async for code, bars in Stock.batch_get_min_level_bars_in_range(
                codes, FrameType.MIN1, feb9_0931, feb9_1033
            ):
                pass

            self.assertEqual(len(codes), mock_qfq.call_count)

        # end > cache_start & end < cache_end
        end = datetime.datetime(2022, 2, 9, 10, 31)
        async for code, bars in Stock.batch_get_min_level_bars_in_range(
            codes, FrameType.MIN30, start, end
        ):
            persisted = bars_from_csv(code, "30m", start, feb8_1500)
            min_bars = bars_from_csv(code, "1m", feb9_0931, feb9_1030)

            cached = Stock.resample(min_bars, FrameType.MIN1, FrameType.MIN30)
            exp = np.concatenate((persisted, cached))
            assert_bars_equal(exp, bars)

        # end == cache_end
        end = datetime.datetime(2022, 2, 9, 10, 33)
        async for code, bars in Stock.batch_get_min_level_bars_in_range(
            codes, FrameType.MIN30, start, end
        ):
            persisted = bars_from_csv(code, "30m", start, feb8_1500)
            min_bars = bars_from_csv(code, "1m", feb9_0931, end)

            cached = Stock.resample(min_bars, FrameType.MIN1, FrameType.MIN30)
            exp = np.concatenate((persisted, cached))
            assert_bars_equal(exp, bars)

        # start > cache_start
        start = datetime.datetime(2022, 2, 9, 9, 35)
        end = datetime.datetime(2022, 2, 9, 10, 33)
        async for code, bars in Stock.batch_get_min_level_bars_in_range(
            codes, FrameType.MIN5, start, end
        ):
            min_bars = bars_from_csv(
                code, "1m", datetime.datetime(2022, 2, 9, 9, 31), end
            )

            exp = Stock.resample(min_bars, FrameType.MIN1, FrameType.MIN5)
            assert_bars_equal(exp, bars)

    async def test_batch_get_cached_bars_n(self):
        codes = ["000002.XSHE", "000001.XSHE", "000004.XSHE"]

        # day level
        barss = await Stock._batch_get_cached_bars_n(
            FrameType.DAY, 1, None, codes=codes
        )
        m1 = bars_from_csv("000001.XSHE", "1m")
        cached = m1[m1["frame"] > datetime.datetime(2022, 2, 9)]
        exp = Stock.resample(cached, FrameType.MIN1, FrameType.DAY)
        exp["frame"][0] = datetime.datetime(2022, 2, 9)

        actual_bar = barss[barss["code"] == "000001.XSHE"][bars_cols].astype(bars_dtype)
        assert_bars_equal(actual_bar, exp)

        # issue 39
        codes_ = ["000003.XSHE", "000001.XSHE", "000002.XSHE", "000004.XSHE"]
        barss = await Stock._batch_get_cached_bars_n(
            FrameType.DAY, 1, None, codes=codes_
        )
        np.testing.assert_array_equal(
            barss["code"], ["000001.XSHE", "000002.XSHE", "000004.XSHE"]
        )
        m1_4 = bars_from_csv("000004.XSHE", "1m")
        cached_4 = m1_4[m1_4["frame"] > datetime.datetime(2022, 2, 9)]
        exp_4 = Stock.resample(cached_4, FrameType.MIN1, FrameType.DAY)
        exp_4["frame"][0] = datetime.datetime(2022, 2, 9)
        actual_bar = barss[barss["code"] == "000004.XSHE"][bars_cols].astype(bars_dtype)
        assert_bars_equal(actual_bar, exp_4)

        # codes is None for day level
        barss = await Stock._batch_get_cached_bars_n(FrameType.DAY, 1, codes=None)
        self.assertEqual(len(barss), 3)
        actual_bar = barss[barss["code"] == "000001.XSHE"][bars_cols].astype(bars_dtype)
        assert_bars_equal(actual_bar, exp)
        self.assertSetEqual(set(barss["code"]), set(codes))

        # 5m, end < lf, so unclosed is excluded
        end = datetime.datetime(2022, 2, 9, 9, 48)
        barss = await Stock._batch_get_cached_bars_n(
            FrameType.MIN5, 3, end=end, codes=codes
        )
        exp = Stock.resample(cached, FrameType.MIN1, FrameType.MIN5)[:3]
        actual_bar = barss[barss["code"] == "000001.XSHE"][bars_cols].astype(bars_dtype)
        assert_bars_equal(exp, actual_bar)

        # 5m, codes is none
        end = datetime.datetime(2022, 2, 9, 9, 48)
        barss = await Stock._batch_get_cached_bars_n(FrameType.MIN5, 3, end=end)
        exp = Stock.resample(cached, FrameType.MIN1, FrameType.MIN5)[:3]
        actual_bar = barss[barss["code"] == "000001.XSHE"][bars_cols].astype(bars_dtype)
        assert_bars_equal(exp, actual_bar)
        self.assertSetEqual(set(barss["code"]), set(codes))

        # 5m, end >= lf, so unclosed is included
        end = datetime.datetime(2022, 2, 9, 10, 33)
        barss = await Stock._batch_get_cached_bars_n(
            FrameType.MIN5, 3, end=end, codes=codes
        )
        exp = Stock.resample(cached, FrameType.MIN1, FrameType.MIN5)[-3:]
        actual_bar = barss[barss["code"] == "000001.XSHE"][bars_cols].astype(bars_dtype)
        assert_bars_equal(exp, actual_bar)

        # 5m, end is none, so unclosed is included
        with freeze_time("2022-02-09 10:33:00"):
            barss = await Stock._batch_get_cached_bars_n(
                FrameType.MIN5, 3, end=None, codes=codes
            )
            exp = Stock.resample(cached, FrameType.MIN1, FrameType.MIN5)[-3:]
            actual_bar = barss[barss["code"] == "000001.XSHE"][bars_cols].astype(
                bars_dtype
            )
            assert_bars_equal(exp, actual_bar)

        # issue 39, 5m, 某支票当天停牌
        codes = ["000003.XSHE", "000002.XSHE", "000001.XSHE", "000004.XSHE"]
        with freeze_time("2022-02-09 10:33:00"):
            barss = await Stock._batch_get_cached_bars_n(
                FrameType.MIN5, 3, end=None, codes=codes
            )
            np.testing.assert_array_equal(
                barss["code"],
                [
                    "000002.XSHE",
                    "000002.XSHE",
                    "000002.XSHE",
                    "000001.XSHE",
                    "000001.XSHE",
                    "000001.XSHE",
                    "000004.XSHE",
                    "000004.XSHE",
                    "000004.XSHE",
                ],
            )
            exp = Stock.resample(cached, FrameType.MIN1, FrameType.MIN5)[-3:]
            actual_bar = barss[barss["code"] == "000001.XSHE"][bars_cols].astype(
                bars_dtype
            )
            assert_bars_equal(exp, actual_bar)

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

        await Stock.save_trade_price_limits(limits, to_cache=False)
        actual = await Stock.get_trade_price_limits(code, start, end)
        self.assertAlmostEqual(3.45, actual[0]["high_limit"])

    async def test_save_trade_price_limits2(self):
        # 清除之前的UT数据残留
        await self.client.drop_measurement("stock_bars_1d")
        await Stock.reset_price_limits_cache(True, None)

        limits = np.array(
            [
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
        await cache.security.set(TRADE_PRICE_LIMITS_DATE, "2022-04-01")
        await Stock.save_trade_price_limits(limits, False)
        start = datetime.date(2022, 3, 31)
        end = datetime.date(2022, 4, 6)
        actual = await Stock.get_trade_price_limits(code, start, end)
        self.assertAlmostEqual(3.45, actual[1]["high_limit"])

        limits = np.array(
            [
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
                ("high_limit", "<f4"),
                ("low_limit", "<f4"),
            ],
        )
        code = "002482.XSHE"
        limits = numpy_append_fields(
            limits, "code", [code] * len(limits), [("code", "O")]
        )
        # save it to cache
        await Stock.save_trade_price_limits(limits, True)

        start = datetime.date(2022, 3, 23)
        end = datetime.date(2022, 4, 6)
        actual = await Stock.get_trade_price_limits(code, start, end)
        self.assertAlmostEqual(3.23, actual[2]["high_limit"])

        await Stock.reset_price_limits_cache(False, datetime.date(2022, 4, 6))

    async def test_reset_price_limit_cache(self):
        limits = np.array(
            [
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
                ("high_limit", "<f4"),
                ("low_limit", "<f4"),
            ],
        )
        code = "002482.XSHE"
        limits = numpy_append_fields(
            limits, "code", [code] * len(limits), [("code", "O")]
        )
        # save it to cache
        await Stock.save_trade_price_limits(limits, True)
        date_str = await cache.security.get(TRADE_PRICE_LIMITS_DATE)
        self.assertEqual(date_str, "2022-04-06")

        await Stock.reset_price_limits_cache(False, datetime.date(2022, 4, 6))
        date_str = await cache.security.get(TRADE_PRICE_LIMITS_DATE)
        self.assertFalse(date_str)

    @freeze_time("2022-02-09 10:33:00")
    async def test_get_bars_in_range(self):
        code = "000001.XSHE"
        start = datetime.datetime(2022, 2, 8, 10)
        now = datetime.datetime(2022, 2, 9, 10, 33)
        end = now

        # 1. include unclosed, end = lf
        bars = await Stock.get_bars_in_range(code, FrameType.MIN30, start, end)
        self.assertEqual(11, len(bars))
        self.assertEqual(datetime.datetime(2022, 2, 8, 10), bars[0]["frame"])
        self.assertEqual(datetime.datetime(2022, 2, 9, 10, 33), bars[-1]["frame"])

        # 2. unclosed is false
        bars = await Stock.get_bars_in_range(
            code, FrameType.MIN30, start, end, unclosed=False
        )
        self.assertEqual(10, len(bars))
        self.assertEqual(datetime.datetime(2022, 2, 8, 10), bars[0]["frame"])
        self.assertEqual(datetime.datetime(2022, 2, 9, 10, 30), bars[-1]["frame"])

        # end < cache_start and now.date() > end.date()
        start_ = datetime.datetime(2022, 2, 8, 10)
        end_ = datetime.datetime(2022, 2, 8, 15)
        code_ = "000001.XSHE"
        with freeze_time("2022-09-09 15:00:00"):
            bars = await Stock.get_bars_in_range(code, FrameType.MIN30, start, end)
            exp = bars_from_csv(code_, "30m", start_, end_)
            assert_bars_equal(exp, bars)

        # end < cache_start
        start_ = datetime.datetime(2022, 2, 8, 10)
        end_ = datetime.datetime(2022, 2, 8, 15)
        code_ = "000001.XSHE"
        bars = await Stock.get_bars_in_range(code, FrameType.MIN30, start_, end_)
        exp = bars_from_csv(code_, "30m", start_, end_)
        assert_bars_equal(exp, bars)

        # 3. test ff < end < last frame
        bars = await Stock.get_bars_in_range(
            code, FrameType.MIN30, start, datetime.datetime(2022, 2, 9, 9, 31)
        )
        self.assertEqual(8, len(bars))
        self.assertEqual(datetime.datetime(2022, 2, 8, 10), bars[0]["frame"])
        self.assertEqual(datetime.datetime(2022, 2, 8, 15), bars[-1]["frame"])

        # test start >= ff
        start_ = datetime.datetime(2022, 2, 9, 9, 40)
        end_ = datetime.datetime(2022, 2, 9, 10, 30)
        code_ = "000001.XSHE"
        bars = await Stock.get_bars_in_range(code, FrameType.MIN5, start_, end_)
        exp = bars_from_csv(code_, "1m", datetime.datetime(2022, 2, 9, 9, 31), end_)
        exp = Stock.resample(exp, FrameType.MIN1, FrameType.MIN5)[1:]
        assert_bars_equal(exp, bars)

        # test end is None
        bars = await Stock.get_bars_in_range(code, FrameType.MIN30, start)
        self.assertEqual(11, len(bars))
        self.assertEqual(datetime.datetime(2022, 2, 8, 10), bars[0]["frame"])
        self.assertEqual(datetime.datetime(2022, 2, 9, 10, 33), bars[-1]["frame"])

        # test end < lf
        end = datetime.datetime(2022, 2, 8, 15, 31)
        bars = await Stock.get_bars_in_range(code, FrameType.MIN30, start, end=end)
        self.assertEqual(8, len(bars))
        self.assertEqual(datetime.datetime(2022, 2, 8, 10), bars[0]["frame"])
        self.assertEqual(datetime.datetime(2022, 2, 8, 15), bars[-1]["frame"])

        # test day, include unclosed
        bars = await Stock.get_bars_in_range(
            code, FrameType.DAY, datetime.date(2022, 2, 7)
        )
        self.assertEqual(3, len(bars))
        self.assertEqual(datetime.date(2022, 2, 9), bars[-1]["frame"].item().date())
        self.assertEqual(datetime.date(2022, 2, 7), bars[0]["frame"].item().date())

        # test day, exclude unclosed
        bars = await Stock.get_bars_in_range(
            code, FrameType.DAY, datetime.date(2022, 2, 7), unclosed=False
        )
        self.assertEqual(2, len(bars))
        self.assertEqual(datetime.date(2022, 2, 8), bars[-1]["frame"].item().date())
        self.assertEqual(datetime.date(2022, 2, 7), bars[0]["frame"].item().date())

        # test qfq
        with mock.patch("omicron.models.stock.Stock.qfq") as mocked_qfq:
            bars = await Stock.get_bars_in_range(
                code, FrameType.DAY, datetime.date(2022, 2, 7)
            )
            mocked_qfq.assert_called_once()

        with mock.patch("omicron.models.stock.Stock.qfq") as mocked_qfq:
            bars = await Stock.get_bars_in_range(
                code, FrameType.DAY, datetime.date(2022, 2, 7), fq=False
            )
            mocked_qfq.assert_not_called()

        with mock.patch("omicron.models.stock.Stock.qfq") as mocked_qfq:
            bars = await Stock.get_bars_in_range(
                code, FrameType.MIN30, datetime.datetime(2022, 2, 8, 10)
            )
            mocked_qfq.assert_called_once()

        with mock.patch("omicron.models.stock.Stock.qfq") as mocked_qfq:
            bars = await Stock.get_bars_in_range(
                code, FrameType.MIN30, datetime.datetime(2022, 2, 8, 10), fq=False
            )
            mocked_qfq.assert_not_called()

    async def test_batch_cache_unclosed_bars(self):
        data = {"000001.XSHE": bars_from_csv("000001.XSHE", "1d")}
        await Stock.batch_cache_unclosed_bars(FrameType.DAY, data)
        barss = await Stock._batch_get_cached_bars_n(
            FrameType.DAY, 1, codes=["000001.XSHE"]
        )
        self.assertEqual(1, len(data))
        self.assertEqual(
            datetime.date(2022, 2, 7),
            barss[barss["code"] == "000001.XSHE"]["frame"].item().date(),
        )

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

    async def test_get_latest_price(self):
        cache.feature.hset(TRADE_LATEST_PRICE, "000001", 10.2)
        cache.feature.hset(TRADE_LATEST_PRICE, "002227", 12.1)
        cache.feature.hset(TRADE_LATEST_PRICE, "601398", 5)

        codes = ["000001.XSHE"]
        rc = await Stock.get_latest_price(codes)
        assert_array_equal([10.2], rc)

        codes = ["002227.XSHE", "601398.XSHG"]
        rc = await Stock.get_latest_price(codes)
        assert_array_equal([12.1, 5], rc)

        codes = ["000002.XSHE"]
        rc = await Stock.get_latest_price(codes)
        assert_array_equal([None], rc)
