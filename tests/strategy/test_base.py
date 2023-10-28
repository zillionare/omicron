import datetime
import pickle
import unittest
from functools import partial
from os import path
from unittest import mock

import cfg4py
import numpy as np
from coretypes import FrameType

import omicron
from omicron.strategy.base import BaseStrategy
from tests import init_test_env, test_dir

cfg = cfg4py.get_instance()


class BaseStrategyTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        await init_test_env()
        await omicron.init()

        return await super().asyncSetUp()

    async def asyncTearDown(self) -> None:
        await omicron.close()
        return await super().asyncTearDown()

    @mock.patch("omicron.strategy.base.TraderClient")
    @mock.patch("omicron.strategy.base.BaseStrategy.sell")
    @mock.patch("omicron.strategy.base.BaseStrategy.buy")
    async def test_strategy(self, mc1, mc2, mc3):
        start = datetime.date(2021, 9, 9)
        end = datetime.date(2022, 1, 26)

        s = BaseStrategy(
            "", is_backtest=True, start=start, end=end, frame_type=FrameType.DAY
        )

        async def dummy_predict(*args, **kwargs):
            self = args[0]
            bars = await self.peek("000002.XSHE", 5)
            if bars is not None:
                print(
                    f"peek returns {len(bars)}, since {bars[0]['frame']} to {bars[-1]['frame']}"
                )

        async def mock_cache_bars_for_backtest(*args):
            self, portfolio, n = args
            with open(path.join(test_dir(), "data/test_strategy.pkl"), "rb") as f:
                barss = pickle.load(f)
                self.bs.barss = barss

        s.predict = partial(dummy_predict, s)

        with mock.patch.object(
            s, "_cache_bars_for_backtest", partial(mock_cache_bars_for_backtest, s)
        ):
            await s.backtest(portfolio=["000002.XSHE", "000004.XSHE"], required_bars=2)
