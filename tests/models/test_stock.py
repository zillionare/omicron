import unittest
from omicron.models.stock import Stock
import numpy as np
from unittest import mock
import arrow


class StockTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        start = arrow.get("1990-01-01").date()
        alive = arrow.get("2200-01-01").date()
        exited = arrow.get("2020-01-01").date()

        self.stocks = np.array(
            [
                ("000001.XSHE", "平安银行", "PAYH", start, alive, "stock"),
                ("000001.XSHG", "上证指数", "SHZS", start, alive, "index"),
                ("000002.XSHE", "万科A", "WKA", start, exited, "stock"),
                ("300000.XSHE", "招商轮船", "ZSLC", start, alive, "stock"),
                ("600000.XSHG", "浦发银行", "PFYH", start, alive, "stock"),
                ("600005.XSHG", "ST建工", "BYJG", start, alive, "stock"),
                ("600006.XSHG", "*ST白银", "BYBJ", start, alive, "stock"),
                ("688010.XSHG", "湖北银行", "YHBY", start, alive, "stock"),
            ],
            dtype=Stock.fileds_type,
        )

        return super().setUp()

    def test_choose(self):
        with mock.patch.object(Stock, "_stocks", self.stocks):
            codes = Stock.choose()
            exp = ["000001.XSHE", "300000.XSHE", "600000.XSHG"]
            self.assertListEqual(exp, codes)

            # 允许ST
            codes = Stock.choose(exclude_st=False)
            exp = [
                "000001.XSHE",
                "300000.XSHE",
                "600000.XSHG",
                "600005.XSHG",
                "600006.XSHG",
            ]
            self.assertListEqual(exp, codes)

            # 允许科创板
            codes = Stock.choose(exclude_688=False)
            exp = ["000001.XSHE", "300000.XSHE", "600000.XSHG", "688010.XSHG"]
            self.assertListEqual(exp, codes)

    def test_choose_cyb(self):

        with mock.patch.object(Stock, "_stocks", self.stocks):
            self.assertListEqual(["300000.XSHE"], Stock.choose_cyb())

    def test_choose_kcb(self):
        with mock.patch.object(Stock, "_stocks", self.stocks):
            self.assertListEqual(["688010.XSHG"], Stock.choose_kcb())

    def test_fuzzy_match(self):
        with mock.patch.object(Stock, "_stocks", self.stocks):
            exp = set(["600000.XSHG", "600005.XSHG", "600006.XSHG"])
            self.assertSetEqual(exp, set(Stock.fuzzy_match("600").keys()))

            exp = set(["000001.XSHE", "600000.XSHG"])
            self.assertSetEqual(exp, set(Stock.fuzzy_match("P").keys()))

            exp = set(["000001.XSHE"])
            self.assertSetEqual(exp, set(Stock.fuzzy_match("平").keys()))
