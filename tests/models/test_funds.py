import datetime
import logging
import unittest
from random import randint
from unittest.mock import patch

import arrow
import numpy as np

import omicron
from omicron.models.funds import FundNetValue, FundPortfolioStock, Funds, FundShareDaily
from tests import init_test_env

logger = logging.getLogger(__name__)


class FundNetValueTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.cfg = await init_test_env()

        self.cfg.postgres.enabled = True
        await omicron.init()
        await FundNetValue.delete.gino.status()
        await Funds.delete.gino.status()

    async def asyncTearDown(self) -> None:
        await FundNetValue.delete.gino.status()
        await Funds.delete.gino.status()
        await omicron.close()

    async def test_crud(self) -> None:

        dtype = [
            ("code", "O"),
            ("name", "O"),
            ("advisor", "O"),
            ("trustee", "O"),
            ("operate_mode_id", "f4"),
            ("operate_mode", "O"),
            ("underlying_asset_type_id", "f4"),
            ("underlying_asset_type", "O"),
            ("start_date", "O"),
            ("end_date", "O"),
            ("total_tna", "f4"),
            ("net_value", "f4"),
            ("quote_change_weekly", "f4"),
            ("quote_change_monthly", "f4"),
        ]
        funds = np.array(
            [
                (
                    "999999",
                    "华夏成长先锋一年持有混合",
                    "999999",
                    "999999",
                    "401001",
                    "开放式基金",
                    "402003",
                    "债券型",
                    arrow.get("2020-01-01").date(),
                    arrow.get("2020-01-01").date(),
                    0,
                    0,
                    "0.95",
                    "1.1",
                )
            ],
            dtype=dtype,
        )

        fund_net_vals_dtype = [
            ("code", "O"),
            ("net_value", "O"),
            ("sum_value", "O"),
            ("factor", "O"),
            ("acc_factor", "O"),
            ("refactor_net_value", "O"),
            ("day", "O"),
        ]
        fund_net_vals = np.array(
            [
                (
                    "999999",
                    1.5,
                    1.5,
                    1.5,
                    1.5,
                    1.5,
                    arrow.get("2020-01-01").date(),
                )
            ],
            dtype=fund_net_vals_dtype,
        )
        result = await Funds.save(funds)
        self.assertEqual(len(result), len(funds))

        result = await FundNetValue.save(fund_net_vals)
        self.assertEqual(len(result), len(fund_net_vals))

        result = await Funds.get(code=["999999"])
        self.assertEqual(result["items"][0]["net_value"], 1.5)


class FundsTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.cfg = await init_test_env()

        self.cfg.postgres.enabled = True
        await omicron.init()
        await Funds.delete.gino.status()
        await FundPortfolioStock.delete.gino.status()

    async def asyncTearDown(self) -> None:
        await Funds.delete.gino.status()
        await FundPortfolioStock.delete.gino.status()
        await omicron.close()

    async def test_crud(self) -> None:
        dtype = [
            ("code", "O"),
            ("name", "O"),
            ("advisor", "O"),
            ("trustee", "O"),
            ("operate_mode_id", "f4"),
            ("operate_mode", "O"),
            ("underlying_asset_type_id", "f4"),
            ("underlying_asset_type", "O"),
            ("start_date", "O"),
            ("end_date", "O"),
            ("total_tna", "f4"),
            ("net_value", "f4"),
            ("quote_change_weekly", "f4"),
            ("quote_change_monthly", "f4"),
        ]
        total_tna = randint(1, 100)
        funds = np.array(
            [
                (
                    "999999",
                    "华夏成长先锋一年持有混合",
                    "999999",
                    "999999",
                    "401001",
                    "开放式基金",
                    "402003",
                    "债券型",
                    arrow.get("2020-01-01").date(),
                    arrow.get("2020-01-01").date(),
                    total_tna,
                    "1.5",
                    "0.95",
                    "1.1",
                )
            ],
            dtype=dtype,
        )
        result = await Funds.save(funds)
        self.assertEqual(len(result), len(funds))

        code = "999999"
        recs = await Funds.get(code=code)
        self.assertEqual(recs["items"][0]["code"], code)
        self.assertEqual(recs["count"], 1)

        code = ["999999"]
        recs = await Funds.get(code=code)
        self.assertEqual(recs["items"][0]["code"], code[0])
        self.assertEqual(recs["count"], len(code))

        name = "华夏成长先锋一年持有混合"
        recs = await Funds.get(name=name)
        self.assertEqual(recs["items"][0]["name"], name)

        operate_mode_ids = [401001]
        recs = await Funds.get(operate_mode_ids=operate_mode_ids)
        self.assertEqual(recs["items"][0]["operate_mode_id"], operate_mode_ids[0])
        self.assertEqual(recs["items"][0]["operate_mode"], "开放式基金")

        underlying_asset_type = 402003
        recs = await Funds.get(underlying_asset_type=underlying_asset_type)
        self.assertEqual(
            recs["items"][0]["underlying_asset_type_id"], underlying_asset_type
        )
        self.assertEqual(recs["items"][0]["underlying_asset_type"], "债券型")

        total_tna_min = total_tna
        total_tna_max = total_tna
        recs = await Funds.get(total_tna_max=total_tna_max, total_tna_min=total_tna_min)
        self.assertEqual(recs["count"], 1)

        fund_portfolio_stocks_dtypes = [
            ("code", "O"),
            ("period_start", "O"),
            ("period_end", "O"),
            ("pub_date", "O"),
            ("report_type_id", "O"),
            ("report_type", "O"),
            ("rank", "f4"),
            ("symbol", "O"),
            ("name", "O"),
            ("shares", "f4"),
            ("market_cap", "f4"),
            ("proportion", "f4"),
            ("deadline", "O"),
        ]
        fund_portfolio_stocks = np.array(
            [
                (
                    "999999",
                    arrow.get("2021-01-01").date(),
                    arrow.get("2021-03-31").date(),
                    arrow.get("2021-04-12").date(),
                    403004,
                    "第四季度",
                    5,
                    "000001",
                    "平安银行",
                    348200,
                    6734188,
                    0.73,
                    arrow.get("2021-03-31").date(),
                )
            ],
            dtype=fund_portfolio_stocks_dtypes,
        )
        result = await FundPortfolioStock.save(fund_portfolio_stocks)
        self.assertEqual(len(result), len(fund_portfolio_stocks))

        position_stock = "平安银行"
        recs = await Funds.get(position_stock=position_stock)
        self.assertEqual(recs["count"], 1)

        position_symbol = "000001"
        recs = await Funds.get(position_symbol=position_symbol)
        self.assertEqual(recs["count"], 1)

        total_tna_min = total_tna
        fund_range = 1
        recs = await Funds.get(fund_range=fund_range)
        self.assertEqual(recs["count"], 1)

        position_stock_percent = 0.73
        recs = await Funds.get(position_stock_percent=position_stock_percent)
        self.assertEqual(recs["count"], 1)

        position_stock_percent = 0.73
        recs = await Funds.get(
            position_stock_percent=position_stock_percent, code="999999"
        )
        self.assertEqual(recs["count"], 1)

        orders = [{"order": "desc", "field": "total_tna"}]
        recs = await Funds.get(orders=orders, code="999999")
        self.assertEqual(recs["count"], 1)


class FundPortfolioStockTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.cfg = await init_test_env()

        self.cfg.postgres.enabled = True
        await omicron.init()

    async def asyncTearDown(self) -> None:
        await omicron.close()

    async def test_crud(self) -> None:

        fund_portfolio_stocks_dtypes = [
            ("code", "O"),
            ("period_start", "O"),
            ("period_end", "O"),
            ("pub_date", "O"),
            ("report_type_id", "O"),
            ("report_type", "O"),
            ("rank", "f4"),
            ("symbol", "O"),
            ("name", "O"),
            ("shares", "f4"),
            ("market_cap", "f4"),
            ("proportion", "f4"),
            ("deadline", "O"),
        ]
        fund_portfolio_stocks = np.array(
            [
                (
                    "999999",
                    arrow.get("2021-01-01").date(),
                    arrow.get("2021-03-31").date(),
                    arrow.get("2021-04-12").date(),
                    403004,
                    "第四季度",
                    5,
                    "000001",
                    "平安银行",
                    348200,
                    6734188,
                    0.73,
                    arrow.get("2021-03-31").date(),
                )
            ],
            dtype=fund_portfolio_stocks_dtypes,
        )
        result = await FundPortfolioStock.save(fund_portfolio_stocks)
        self.assertEqual(len(result), len(fund_portfolio_stocks))

        codes = ["999999"]
        recs = await FundPortfolioStock.get(codes=codes)
        self.assertEqual(len(recs), 1)

        codes = "999999"
        symbol = "000001"
        recs = await FundPortfolioStock.get(codes=codes, symbol=symbol)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["symbol"], symbol)

        symbol = ["000001"]
        recs = await FundPortfolioStock.get(codes=codes, symbol=symbol)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["symbol"], symbol[0])

        deadline = arrow.get("2021-03-31").date()
        recs = await FundPortfolioStock.get(codes=codes, deadline=deadline)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["deadline"], deadline.strftime("%Y-%m-%d"))

        stock_symbols = ["000001"]
        recs = await FundPortfolioStock.get(
            codes=codes, stock_symbols=stock_symbols, symbol=symbol
        )
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["is_single_stock"], True)

        stock_symbols = ["000002"]
        recs = await FundPortfolioStock.get(
            codes=codes, stock_symbols=stock_symbols, symbol=symbol
        )
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["is_single_stock"], False)


class FundShareDailyTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.cfg = await init_test_env()

        self.cfg.postgres.enabled = True
        await omicron.init()
        await FundShareDaily.delete.gino.status()
        await Funds.delete.gino.status()

    async def asyncTearDown(self) -> None:
        await FundShareDaily.delete.gino.status()
        await Funds.delete.gino.status()
        await omicron.close()

    async def test_crud(self) -> None:

        dtype = [
            ("code", "O"),
            ("name", "O"),
            ("advisor", "O"),
            ("trustee", "O"),
            ("operate_mode_id", "f4"),
            ("operate_mode", "O"),
            ("underlying_asset_type_id", "f4"),
            ("underlying_asset_type", "O"),
            ("start_date", "O"),
            ("end_date", "O"),
            ("total_tna", "f4"),
            ("net_value", "f4"),
            ("quote_change_weekly", "f4"),
            ("quote_change_monthly", "f4"),
        ]
        total_tna = randint(1, 100)
        funds = np.array(
            [
                (
                    "999999",
                    "华夏成长先锋一年持有混合",
                    "999999",
                    "999999",
                    "401001",
                    "开放式基金",
                    "402003",
                    "债券型",
                    arrow.get("2020-01-01").date(),
                    arrow.get("2020-01-01").date(),
                    total_tna,
                    "1.5",
                    "0.95",
                    "1.1",
                )
            ],
            dtype=dtype,
        )
        print("save data to db", funds)
        result = await Funds.save(funds)
        fund_share_daily_dtypes = [
            ("code", "O"),
            ("name", "O"),
            ("date", "O"),
            ("total_tna", "f4"),
        ]
        fund_share_daily = np.array(
            [
                (
                    "999999",
                    "test",
                    arrow.get("2021-03-31").date(),
                    6734188,
                )
            ],
            dtype=fund_share_daily_dtypes,
        )
        result = await FundShareDaily.save(fund_share_daily)
        self.assertEqual(len(result), len(fund_share_daily))

        result = await Funds.get(code=["999999"])
        self.assertEqual(result["items"][0]["total_tna"], 6734188)
