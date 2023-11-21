import datetime
import logging
import unittest
from logging import Formatter
from unittest import mock

import cfg4py
import numpy as np
from coretypes import Frame, FrameType

import omicron
from omicron import tf
from omicron.core.backtestlog import BacktestLogger
from omicron.strategy.base import BaseStrategy
from omicron.strategy.sma import SMAStrategy
from tests import bars_from_csv, init_test_env

cfg = cfg4py.get_instance()
logger = BacktestLogger.getLogger("dummy_strategy")


def mock_available_shares(*args, **kwargs):
    return 500


def mock_position(*args, **kwargs):
    pass


class DummyStrategy(BaseStrategy):
    def __init__(self, url, **kwargs):
        super().__init__(url, **kwargs)

    async def predict(self, frame: Frame, frame_type: FrameType, i: int, **kwargs):
        logger.info("%sth frame", i, date=frame)


class SMAStrategyTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        await init_test_env()
        await omicron.init()

        root = logging.getLogger()
        root.handlers.clear()

        ch = logging.StreamHandler()
        fmt = "[回测] %(bt_date)s | %(message)s"
        formatter = Formatter(fmt)
        ch.setFormatter(formatter)
        ch.setLevel(logging.DEBUG)

        logger.addHandler(ch)

        self.bars = bars_from_csv("600000.XSHG", FrameType.DAY)
        # tf.day_frames = np.array([tf.date2int(f.item()) for f in self.bars["frame"]])
        return await super().asyncSetUp()

    async def get_bars(self, sec, n, frame_type, end, fq=None):
        return self.bars[self.bars["frame"] < tf.combine_time(end, 0)][-n:]

    async def asyncTearDown(self) -> None:
        await omicron.close()
        return await super().asyncTearDown()

    @mock.patch("omicron.strategy.base.TraderClient")
    @mock.patch("omicron.strategy.base.BaseStrategy.sell")
    @mock.patch("omicron.strategy.base.BaseStrategy.buy")
    async def test_base_strategy(self, mc1, mc2, mc3) -> None:
        s = DummyStrategy(
            url="",
            is_backtest=True,
            start=datetime.date(2023, 2, 3),
            end=datetime.date(2023, 4, 28),
            frame_type=FrameType.DAY,
        )

        with mock.patch.object(omicron.models.stock.Stock, "get_bars", self.get_bars):
            await s.backtest(stop_on_error=True)

        s = DummyStrategy(
            url="",
            is_backtest=True,
            start=datetime.datetime(2023, 2, 3, 10),
            end=datetime.datetime(2023, 4, 28, 15),
            frame_type=FrameType.MIN30,
        )

        with mock.patch.object(omicron.models.stock.Stock, "get_bars", self.get_bars):
            await s.backtest(stop_on_error=True)

    @mock.patch("omicron.strategy.base.TraderClient")
    @mock.patch("omicron.strategy.base.BaseStrategy.sell")
    @mock.patch("omicron.strategy.base.BaseStrategy.buy")
    async def test_sma_strategy(self, mc1, mc2, mc3):
        code = "600000.XSHG"
        sma = SMAStrategy(
            code,
            url="",
            is_backtest=True,
            start=datetime.date(2023, 2, 3),
            end=datetime.date(2023, 4, 28),
            frame_type=FrameType.DAY,
            warmup_period = 20
        )

        # setup the mock
        mc3.return_value.available_shares.return_value = 500
        mc3.return_value.available_money = 1_000_000
        mc3.return_value.positions.return_value = np.array(
            [(code, 500)], dtype=[("code", object), ("shares", int)]
        )
        mc3.buy = mock.AsyncMock()
        mc3.sell = mock.AsyncMock()

        # no exception is ok
        with mock.patch.object(omicron.models.stock.Stock, "get_bars", self.get_bars):
            await sma.backtest(stop_on_error=True, portfolio=[code])
