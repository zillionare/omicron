import datetime
from typing import Union

import numpy as np
from coretypes import BarsArray, Frame, FrameType

from omicron import tf
from omicron.core.backtestlog import BacktestLogger
from omicron.models.stock import Stock
from omicron.strategy.base import BaseStrategy

logger = BacktestLogger.getLogger(__name__)


class SMAStrategy(BaseStrategy):
    def __init__(self, sec: str, n_short: int = 5, n_long: int = 10, *args, **kwargs):
        self._sec = sec
        self._n_short = n_short
        self._n_long = n_long

        self.indicators = []

        super().__init__(*args, **kwargs)

    async def before_start(self):
        date = self.bs.end if self.bs is not None else None
        logger.info("before_start, cash is %s", self.cash, date=date)

    async def before_trade(self, date: datetime.date):
        logger.info(
            "before_trade, cash is %s, portfolio is %s",
            self.cash,
            self.positions(date),
            date=date,
        )

    async def after_trade(self, date: datetime.date):
        logger.info(
            "after_trade, cash is %s, portfolio is %s",
            self.cash,
            self.positions(date),
            date=date,
        )

    async def after_stop(self):
        date = self.bs.end if self.bs is not None else None
        logger.info(
            "after_stop, cash is %s, portfolio is %s",
            self.cash,
            self.positions,
            date=date,
        )

    async def predict(
        self, frame: Frame, frame_type: FrameType, i: int, barss, **kwargs
    ):
        if barss is None:
            raise ValueError("please specify `prefetch_stocks`")

        bars: Union[BarsArray, None] = barss.get(self._sec)
        if bars is None:
            raise ValueError(f"{self._sec} not found in `prefetch_stocks`")

        ma_short = np.mean(bars["close"][-self._n_short :])
        ma_long = np.mean(bars["close"][-self._n_long :])

        if ma_short > ma_long:
            self.indicators.append((frame, 1))
            if self.cash >= 100 * bars["close"][-1]:
                await self.buy(
                    self._sec,
                    money=self.cash,
                    order_time=tf.combine_time(frame, 14, 55),
                )
        elif ma_short < ma_long:
            self.indicators.append((frame, -1))
            if self.available_shares(self._sec, frame) > 0:
                await self.sell(
                    self._sec, percent=1.0, order_time=tf.combine_time(frame, 14, 55)
                )
