import datetime
import logging
import uuid
from asyncio import gather
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import jqdatasdk as jq
import numpy as np
import pandas as pd
from coretypes import BarsArray, Frame, FrameType
from numpy.typing import NDArray
from traderclient import TraderClient

from omicron import tf
from omicron.core.backtestlog import BacktestLogger
from omicron.models.security import Security
from omicron.models.stock import Stock
from omicron.plotting.metrics import MetricsGraph

logger = BacktestLogger.getLogger(__name__)


@dataclass
class BacktestState(object):
    start: Frame
    end: Frame
    barss: Union[None, Dict[str, BarsArray]]
    cursor: int


class BaseStrategy:
    def __init__(
        self,
        url: str,
        account: Optional[str] = None,
        token: Optional[str] = None,
        name: Optional[str] = None,
        ver: Optional[str] = None,
        is_backtest: bool = True,
        start: Optional[Frame] = None,
        end: Optional[Frame] = None,
        frame_type: Optional[FrameType] = None,
        baseline: Optional[str] = "399300.XSHE",
    ):
        """构造函数

        Args:
            url: 实盘/回测服务器的地址。
            start: 回测起始日期。回测模式下必须传入。
            end: 回测结束日期。回测模式下必须传入。
            account: 实盘/回测账号。实盘模式下必须传入。在回测模式下，如果未传入，将以策略名+随机字符构建账号。
            token: 实盘/回测时用的token。实盘模式下必须传入。在回测模式下，如果未传入，将自动生成。
            is_backtest: 是否为回测模式。
            name: 策略名。如果不传入，则使用类名字小写
            ver: 策略版本号。如果不传入，则默认为0.1.
            start: 如果是回测模式，则需要提供回测起始时间
            end: 如果是回测模式，则需要提供回测结束时间
            frame_type: 如果是回测模式，则需要提供回测时使用的主周期
            baseline: 如果是回测模式，则可以提供此参数作为回测基准
        """
        self.ver = ver or "0.1"
        self.name = name or self.__class__.__name__.lower() + f"_v{self.ver}"

        self.token = token or uuid.uuid4().hex
        self.account = account or f"smallcap-{self.token[-4:]}"

        self.url = url

        if is_backtest:
            if start is None or end is None or frame_type is None:
                raise ValueError("start, end and frame_type must be presented.")

            self.bs = BacktestState(start, end, None, 0)
            self.bills = None
            self.metrics = None
            self._frame_type = frame_type
            self.broker = TraderClient(
                url,
                self.account,
                self.token,
                is_backtest=True,
                start=self.bs.start,
                end=self.bs.end,
            )
            self._baseline = baseline
        else:
            if account is None or token is None:
                raise ValueError("account and token must be presented.")

            self.broker = TraderClient(url, self.account, self.token, is_backtest=False)

    async def _cache_bars_for_backtest(self, portfolio: List[str], n: int):
        if portfolio is None or len(portfolio) == 0:
            return

        count = tf.count_frames(self.bs.start, self.bs.end, self._frame_type)
        tasks = [
            Stock.get_bars(code, count + n, self._frame_type, self.bs.end, fq=False)
            for code in portfolio
        ]

        results = await gather(*tasks)
        self.bs.barss = {k: v for (k, v) in zip(portfolio, results)}

    def _next(self):
        if self.bs.barss is None:
            return None

        self.bs.cursor += 1
        return {k: Stock.qfq(v[: self.bs.cursor]) for (k, v) in self.bs.barss.items()}

    async def peek(self, code: str, n: int):
        """允许策略偷看未来数据

        可用以因子检验场景。要求数据本身已缓存。否则请用Stock.get_bars等方法获取。
        """
        if self.bs is None or self.bs.barss is None:
            raise ValueError("data is not cached")

        if code in self.bs.barss:
            if self.bs.cursor + n + 1 < len(self.bs.barss[code]):
                return Stock.qfq(
                    self.bs.barss[code][self.bs.cursor : self.bs.cursor + n]
                )

        else:
            raise ValueError("data is not cached")

    async def backtest(self, stop_on_error: bool = True, **kwargs):
        """执行回测

        Args:
            stop_on_error: 如果为True，则发生异常时，将停止回测。否则忽略错误，继续执行。
        Keyword Args:
            portfolio Dict[str, BarsArray]: 代码列表。在该列表中的品种，将在回测之前自动预取行情数据，并在调用predict时，传入截止到当前frame的，长度为n的行情数据。行情周期由构造时的frame_type指定
            min_bars int: 回测时必要的bars的最小值
        """
        portfolio: List[str] = kwargs.get("portfolio")  # type: ignore
        n = kwargs.get("min_bars", 0)
        await self._cache_bars_for_backtest(portfolio, n)
        self.bs.cursor = n

        converter = (
            tf.int2date if self._frame_type in tf.day_level_frames else tf.int2time
        )

        for i, frame in enumerate(
            tf.get_frames(self.bs.start, self.bs.end, self._frame_type)  # type: ignore
        ):
            barss = self._next()
            logger.debug("%sth iteration", i, date=converter(frame))
            try:
                await self.predict(
                    converter(frame), self._frame_type, i, barss=barss, **kwargs  # type: ignore
                )
            except Exception as e:
                logger.exception(e)
                if stop_on_error:
                    raise e

        self.broker.stop_backtest()
        self.bills = self.broker.bills()
        self.metrics = self.broker.metrics(baseline=self._baseline)

    @property
    def cash(self):
        """返回当前可用现金"""
        return self.broker.available_money

    def positions(self, dt: Optional[datetime.date] = None):
        """返回当前持仓"""
        return self.broker.positions(dt)

    def available_shares(self, sec: str, dt: Optional[Frame] = None):
        """返回给定股票在`dt`日的可售股数

        Args:
            sec: 证券代码
            dt: 日期，在实盘中无意义，只能返回最新数据；在回测时，必须指定日期，且返回指定日期下的持仓。
        """
        return self.broker.available_shares(sec, dt)

    async def buy(
        self,
        sec: str,
        price: Optional[float] = None,
        vol: Optional[int] = None,
        money: Optional[float] = None,
        order_time: Optional[datetime.datetime] = None,
    ) -> Dict:
        """买入股票

        Args:
            sec: 证券代码
            price: 委买价。如果为None，则自动转市价买入。
            vol: 委买股数。请自行保证为100的整数。如果为None, 则money必须传入。
            money: 委买金额。如果同时传入了vol，则此参数自动忽略
            order_time: 仅在回测模式下需要提供。实盘模式下，此参数自动被忽略
        Returns:
            见traderclient中的`buy`方法。
        """
        logger.info(
            "buy order: %s, %s, %s, %s",
            sec,
            f"{price:.2f}" if price is not None else None,
            f"{vol:.0f}" if vol is not None else None,
            f"{money:.0f}" if money is not None else None,
            date=order_time,
        )
        if vol is None:
            if money is None:
                raise ValueError("parameter `mnoey` must be presented!")

            return await self.broker.buy_by_money(
                sec, money, price, order_time=order_time
            )
        elif price is None:
            return self.broker.market_buy(sec, vol, order_time=order_time)
        else:
            return self.broker.buy(sec, price, vol, order_time=order_time)

    async def sell(
        self,
        sec: str,
        price: Optional[float] = None,
        vol: Optional[float] = None,
        percent: Optional[float] = None,
        order_time: Optional[datetime.datetime] = None,
    ) -> Union[List, Dict]:
        """卖出股票

        Args:
            sec: 证券代码
            price: 委卖价，如果未提供，则转为市价单
            vol: 委卖股数。如果为None，则percent必须传入
            percent: 卖出一定比例的持仓，取值介于0与1之间。如果与vol同时提供，此参数将被忽略。请自行保证按比例换算后的卖出数据是符合要求的（比如不为100的倍数，但有些情况下这是允许的，所以程序这里无法帮你判断）
            order_time: 仅在回测模式下需要提供。实盘模式下，此参数自动被忽略

        Returns:
            Union[List, Dict]: 成交返回，详见traderclient中的`buy`方法，trade server只返回一个委托单信息
        """
        logger.info(
            "sell order: %s, %s, %s, %s",
            sec,
            f"{price:.2f}" if price is not None else None,
            f"{vol:.0f}" if vol is not None else None,
            f"{percent:.2%}" if percent is not None else None,
            date=order_time,
        )

        if vol is None and percent is None:
            raise ValueError("either vol or percent must be presented")

        if vol is None:
            if price is None:
                price = await self.broker._get_market_sell_price(
                    sec, order_time=order_time
                )
            # there's no market_sell_percent API in traderclient
            return self.broker.sell_percent(sec, price, percent, order_time=order_time)  # type: ignore
        else:
            if price is None:
                return self.broker.market_sell(sec, vol, order_time=order_time)
            else:
                return self.broker.sell(sec, price, vol, order_time=order_time)

    async def filter_paused_stock(self, buylist: List[str], dt: datetime.date):
        secs = await Security.select(dt).eval()
        in_trading = jq.get_price(
            secs, fields=["paused"], start_date=dt, end_date=dt, skip_paused=True
        )["code"].to_numpy()

        return np.intersect1d(buylist, in_trading)

    async def predict(
        self,
        frame: Frame,
        frame_type: FrameType,
        i: int,
        barss: Dict[str, BarsArray],
        **kwargs,
    ):
        """策略评估函数。在此函数中实现交易信号检测和处理。

        Args:
            frame: 当前时间帧
            frame_type: 处理的数据主周期
            i: 当前时间离回测起始的单位数
            barss: 如果调用`backtest`时传入了`portfolio`及`min_bars`参数，则`backtest`将会在回测之前，预取从[start - min_bars * frame_type, end]间的portfolio行情数据，并在每次调用`predict`方法时，通过`barss`参数，将[start - min_bars * frame_type, start + i * frame_type]间的数据传给`predict`方法。传入的数据已进行前复权。

        Keyword Args: 在`backtest`方法中的传入的kwargs参数将被透传到此方法中。
        """
        raise NotImplementedError

    async def plot_metrics(
        self, indicator: Union[pd.DataFrame, List[Tuple], None] = None
    ):
        """策略回测报告

        Args:
            indicator: 回测时使用的指标。如果存在，将叠加到策略回测图上。它应该是一个以日期为索引，指标列名为"value"的DataFrame
        """
        if self.bills is None or self.metrics is None:
            raise ValueError("Please run `start_backtest` first.")

        if isinstance(indicator, list):
            assert len(indicator[0]) == 2
            indicator = pd.DataFrame(indicator, columns=["date", "value"])
            indicator.set_index("date", inplace=True)

        if self._baseline is not None:
            mg = MetricsGraph(
                self.bills,
                self.metrics,
                baseline_code=self._baseline,
                indicator=indicator,
            )
        else:
            mg = MetricsGraph(self.bills, self.metrics, indicator=indicator)
        await mg.plot()
