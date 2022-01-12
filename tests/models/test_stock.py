import datetime
import ftplib
import unittest
from unittest import mock

import arrow
import numpy as np

import omicron
from omicron.core.types import FrameType, bars_with_limit_dtype, stock_bars_dtype
from omicron.models.stock import Stock
from tests import init_test_env


class StockTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        await init_test_env()
        await omicron.init()
        return super().setUp()

    async def asyncTearDown(self) -> None:
        await omicron.close()
        return await super().asyncTearDown()

    def test_choose(self):
        codes = Stock.choose()
        exp = ["000001.XSHE", "300001.XSHE", "600000.XSHG"]
        self.assertListEqual(exp, codes)

        # 允许ST
        codes = Stock.choose(exclude_st=False)
        exp = [
            "000001.XSHE",
            "000005.XSHE",
            "300001.XSHE",
            "600000.XSHG",
            "000007.XSHE",
        ]
        self.assertListEqual(exp, codes)

        # 允许科创板
        codes = Stock.choose(exclude_688=False)
        exp = ["000001.XSHE", "300001.XSHE", "600000.XSHG", "688001.XSHG"]
        self.assertListEqual(exp, codes)

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

    async def test_cache_unclosed_bars(self):
        bars = np.array(
            [
                (
                    datetime.date(2022, 1, 7),
                    17.1,
                    17.28,
                    17.06,
                    17.2,
                    1.1266307e08,
                    1.93771096e09,
                    1.0,
                ),
            ],
            dtype=stock_bars_dtype,
        )

        await Stock.cache_unclosed_bars("000001.XSHE", FrameType.DAY, bars)
        actual = await Stock._get_cached_bars("000001.XSHE", FrameType.DAY)
        np.testing.assert_array_equal(bars, actual)

        bars = np.array(
            [
                (
                    datetime.date(2022, 1, 7),
                    17.1,
                    17.28,
                    17.06,
                    17.2,
                    1.1266307e08,
                    1.93771096e09,
                    1.0,
                ),
                (
                    datetime.date(2022, 1, 8),
                    17.11,
                    17.27,
                    17.0,
                    17.12,
                    1.10788519e08,
                    1.89653584e09,
                    1.0,
                ),
            ],
            dtype=stock_bars_dtype,
        )

        try:
            await Stock.cache_unclosed_bars("000001.XSHE", FrameType.DAY, bars)
        except AssertionError as e:
            self.assertEqual(str(e), "unclosed bars should only have one record")

    async def test_cache_bars(self):
        await Stock.reset_cache()
        bars = np.array(
            [
                (
                    datetime.datetime(2022, 1, 7, 10, 0),
                    17.1,
                    17.28,
                    17.06,
                    17.2,
                    1.1266307e08,
                    1.93771096e09,
                    1.0,
                )
            ],
            dtype=stock_bars_dtype,
        )

        unclosed = np.array(
            [
                (
                    datetime.datetime(2022, 1, 7, 10, 30),
                    17.11,
                    17.27,
                    17.0,
                    17.12,
                    1.10788519e08,
                    1.89653584e09,
                    1.0,
                ),
            ],
            dtype=stock_bars_dtype,
        )

        await Stock.cache_bars("000001.XSHE", FrameType.MIN30, bars)
        await Stock.cache_unclosed_bars("000001.XSHE", FrameType.MIN30, unclosed)
        cached = await Stock._get_cached_bars("000001.XSHE", FrameType.MIN30)

        np.testing.assert_array_equal(bars[0], cached[0])
        np.testing.assert_array_equal(unclosed[0], cached[1])

    async def test_resample(self):
        def totime(tm: str):
            return arrow.get(tm).datetime.replace(tzinfo=None)

        # fiels = ["frame", "open", "high", "low", "close", "volume", "money", "factor"]
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
                ("money", "<f8"),
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
                ("money", "<f8"),
                ("factor", "<f4"),
            ],
        )
        fields = ["open", "high", "low", "close", "volume", "money", "factor"]
        np.testing.assert_array_equal(actual["frame"], exp["frame"])
        for field in fields:
            np.testing.assert_array_almost_equal(actual[field], exp[field], decimal=2)

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
                ("money", "<f8"),
                ("factor", "<f4"),
            ],
        )

        actual = Stock.resample(bars, FrameType.MIN1, FrameType.MIN15)
        np.testing.assert_array_equal(actual["frame"], exp["frame"])
        for field in fields:
            np.testing.assert_array_almost_equal(actual[field], exp[field], decimal=2)

        bars[0]["frame"] = datetime.datetime(2021, 4, 27, 9, 35)
        try:
            Stock.resample(bars, FrameType.MIN1, FrameType.MIN5)
        except ValueError as e:
            self.assertEqual(str(e), "resampling from 1min must start from 9:31")

    async def test_persist_bars(self):
        bars = np.array(
            [
                (
                    datetime.datetime(2022, 1, 7, 10, 0),
                    17.1,
                    17.28,
                    17.06,
                    17.2,
                    1.1266307e08,
                    1.93771096e09,
                    18.83,
                    15.41,
                    1.0,
                    "000001.XSHE"
                )
            ],
            dtype=bars_with_limit_dtype,
        )
        await Stock.persist_bars(FrameType.MIN30, bars)

    async def test_get_bars(self):
        await Stock.reset_cache()

        code = "000001.XSHE"
        cache_bars = np.array(
            [
                (
                    datetime.datetime(2022, 1, 10, 9, 45),
                    17.29,
                    17.42,
                    17.16,
                    17.18,
                    23069200.0,
                    3.99426730e08,
                    121.71913,
                ),
                (
                    datetime.datetime(2022, 1, 10, 10, 0),
                    17.17,
                    17.27,
                    17.08,
                    17.13,
                    12219500.0,
                    2.09659075e08,
                    121.71913,
                ),
                (
                    datetime.datetime(2022, 1, 10, 10, 15),
                    17.14,
                    17.15,
                    17.03,
                    17.04,
                    11106800.0,
                    1.89643093e08,
                    121.71913,
                ),
                (
                    datetime.datetime(2022, 1, 10, 10, 30),
                    17.05,
                    17.12,
                    17.05,
                    17.09,
                    3352900.0,
                    5.72981150e07,
                    121.71913,
                ),
                (
                    datetime.datetime(2022, 1, 10, 10, 45),
                    17.08,
                    17.19,
                    17.07,
                    17.14,
                    3150400.0,
                    5.39887270e07,
                    121.71913,
                ),
            ],
            dtype=stock_bars_dtype,
        )

        cache_unclosed_bars = np.array(
            [
                (
                    datetime.datetime(2022, 1, 10, 10, 47),
                    17.14,
                    17.17,
                    17.14,
                    17.15,
                    376200.0,
                    6.45355700e06,
                    121.71913,
                )
            ],
            dtype=stock_bars_dtype,
        )

        await Stock.cache_bars(code, FrameType.MIN15, cache_bars)
        await Stock.cache_unclosed_bars(code, FrameType.MIN15, cache_unclosed_bars)

        persist_bars = np.array(
            [
                (
                    datetime.datetime(2022, 1, 7, 13, 15),
                    17.24,
                    17.28,
                    17.24,
                    17.26,
                    7509700.0,
                    1.29655754e08,
                    121.71913,
                ),
                (
                    datetime.datetime(2022, 1, 7, 13, 30),
                    17.25,
                    17.26,
                    17.17,
                    17.19,
                    6348400.0,
                    1.09249162e08,
                    121.71913,
                ),
                (
                    datetime.datetime(2022, 1, 7, 13, 45),
                    17.18,
                    17.22,
                    17.16,
                    17.2,
                    2748100.0,
                    4.72354460e07,
                    121.71913,
                ),
                (
                    datetime.datetime(2022, 1, 7, 14, 0),
                    17.19,
                    17.2,
                    17.17,
                    17.2,
                    3153700.0,
                    5.42049370e07,
                    121.71913,
                ),
                (
                    datetime.datetime(2022, 1, 7, 14, 15),
                    17.19,
                    17.23,
                    17.19,
                    17.22,
                    5729200.0,
                    9.86084270e07,
                    121.71913,
                ),
                (
                    datetime.datetime(2022, 1, 7, 14, 30),
                    17.22,
                    17.25,
                    17.2,
                    17.22,
                    6729300.0,
                    1.15946564e08,
                    121.71913,
                ),
                (
                    datetime.datetime(2022, 1, 7, 14, 45),
                    17.21,
                    17.23,
                    17.17,
                    17.18,
                    7203100.0,
                    1.23899524e08,
                    121.71913,
                ),
                (
                    datetime.datetime(2022, 1, 7, 15, 0),
                    17.16,
                    17.2,
                    17.15,
                    17.2,
                    8683800.0,
                    1.49181676e08,
                    121.71913,
                ),
            ],
            dtype=stock_bars_dtype,
        )

        with mock.patch.object(Stock, "_get_persited_bars", return_value=persist_bars):
            ft = FrameType.MIN15
            end = datetime.datetime(2022, 1, 10, 10, 47)
            n = 1
            bars = await Stock.get_bars("000001.XSHE", n, ft, end)
            self.assertEqual(len(bars), n)
