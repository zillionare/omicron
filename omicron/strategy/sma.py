import numpy as np
from coretypes import Frame, FrameType

from omicron import tf
from omicron.models.stock import Stock
from omicron.strategy.base import BaseStrategy


class SMAStrategy(BaseStrategy):
    def __init__(self, sec: str, n_short: int = 5, n_long: int = 10, *args, **kwargs):
        self._sec = sec
        self._n_short = n_short
        self._n_long = n_long

        super().__init__(*args, **kwargs)

    async def predict(self, frame: Frame, frame_type: FrameType, i: int):
        n = max(self._n_short, self._n_long) - 1
        bars = await Stock.get_bars(self._sec, n, frame_type, end=frame)

        if len(bars) < n:
            return

        ma_short = np.mean(bars["close"][-self._n_short :])
        ma_long = np.mean(bars["close"][-self._n_long :])

        if ma_short > ma_long:
            await self.buy(
                self._sec, money=self.cash, order_time=tf.combine_time(frame, 14, 55)
            )
        elif ma_short < ma_long:
            await self.sell(
                self._sec, percent=1.0, order_time=tf.combine_time(frame, 14, 55)
            )
