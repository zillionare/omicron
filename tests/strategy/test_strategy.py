import datetime
import unittest
from unittest import mock

import cfg4py
import numpy as np
from coretypes import FrameType

import omicron
from omicron import tf
from omicron.strategy.sma import SMAStrategy
from tests import bars_from_csv, init_test_env

cfg = cfg4py.get_instance()


class SMAStrategyTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        await init_test_env()
        await omicron.init()

        self.bars = bars_from_csv("600000.XSHG", FrameType.DAY)
        tf.day_frames = np.array([tf.date2int(f.item()) for f in self.bars["frame"]])
        return await super().asyncSetUp()

    async def get_bars(self, sec, n, frame_type, end):
        return self.bars[self.bars["frame"] < tf.combine_time(end, 0)][-n:]

    async def asyncTearDown(self) -> None:
        await omicron.close()
        return await super().asyncTearDown()

    @mock.patch("omicron.strategy.base.TraderClient")
    @mock.patch("omicron.strategy.base.BaseStrategy.sell")
    @mock.patch("omicron.strategy.base.BaseStrategy.buy")
    async def test_strategy(self, mc1, mc2, mc3):
        sma = SMAStrategy(
            "600000.XSHG",
            url="",
            is_backtest=True,
            start=datetime.date(2023, 2, 3),
            end=datetime.date(2023, 4, 28),
            frame_type=FrameType.DAY,
        )

        # no exception is ok
        with mock.patch.object(omicron.models.stock.Stock, "get_bars", self.get_bars):
            await sma.backtest(stop_on_error=True)
