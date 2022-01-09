from re import A
import unittest
from omicron.models.stock import Stock
import numpy as np
from unittest import mock
import omicron
from tests import init_test_env
import datetime
from omicron.core.types import FrameType, stock_bars_dtype


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
                    17.2,
                    18.83,
                    15.41,
                    17.12,
                    1.0,
                ),
            ],
            dtype=stock_bars_dtype,
        )

        await Stock.cache_unclosed_bars("000001.XSHE", FrameType.DAY, bars)
        actual = await Stock._get_cached_bars("000001.XSHE", FrameType.DAY)
        np.testing.assert_array_equal(bars, bars)

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
                    17.2,
                    18.83,
                    15.41,
                    17.12,
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
                    17.12,
                    18.87,
                    15.44,
                    17.15,
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
                    17.2,
                    18.83,
                    15.41,
                    17.12,
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
                    17.12,
                    18.87,
                    15.44,
                    17.15,
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
