import datetime
import uuid
from asyncio import gather
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import jqdatasdk as jq
import numpy as np
import pandas as pd
from coretypes import BarsArray, Frame, FrameType
from coretypes.errors.trade import TradeError
from deprecation import deprecated
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
    cursor: Frame
    warmup_peroid: int
    baseline: str = "399300.XSHE"


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
        warmup_period: int = 0,
        principal: float = 1_000_000,
        commission: float = 1.5e-4
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
            warmup_period: 策略执行时需要的最小bar数（以frame_type）计。
            principal: 回测时初始资金。默认为100万。实盘时会自动忽略此参数
            commission: 回测时的手续费率。默认为0.015%。实盘时会自动忽略此参数
        """
        self.ver = ver or "0.1"
        self.name = name or self.__class__.__name__.lower() + f"_v{self.ver}"

        self.token = token or uuid.uuid4().hex
        self.account = account or f"smallcap-{self.token[-4:]}"

        self.url = url
        self.bills = None
        self.metrics = None

        # used by both live and backtest
        self.warmup_period = warmup_period
        self.is_backtest = is_backtest
        if is_backtest:
            if start is None or end is None or frame_type is None:
                raise ValueError("start, end and frame_type must be presented.")

            start = tf.floor(start, frame_type)
            end = tf.floor(end, frame_type)

            self.bs = BacktestState(start, end, None, 0, warmup_period)
            self._frame_type = frame_type
            self.broker = TraderClient(
                url,
                self.account,
                self.token,
                is_backtest=True,
                start=self.bs.start,
                end=self.bs.end,
                principal=principal,
                commission=commission
            )
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

        barss = {}

        for k, v in self.bs.barss.items():
            iend = np.argwhere(v['frame'] == np.datetime64(self.bs.cursor)).flatten()
            if len(iend) == 0:
                barss[k] = None
                continue
            else:
                iend = iend[0] + 1
                istart = max(0, iend - self.bs.warmup_peroid)
                barss[k] = v[istart: iend]

        self.bs.cursor = tf.shift(self.bs.cursor, 1, self._frame_type)
        return barss

    async def peek(self, code: str, n: int):
        """允许策略偷看未来数据

        可用以因子检验场景。要求数据本身已缓存。否则请用Stock.get_bars等方法获取。
        """
        if self.bs is None or self.bs.barss is None:
            raise ValueError("data is not cached")

        if code in self.bs.barss:
            bars = self.bs.barss[code]
            istart = np.argwhere(bars["frame" == self.bs.cursor]).flatten()
            if len(istart) == 0: # 如果当前周期处于停牌中，则不允许任何操作
                raise ValueError("无数据或者停牌中")
            
            istart = istart[0]
            return Stock.qfq(
                self.bs.barss[code][istart : istart + n + 1]
            )

        else:
            raise ValueError("data is not cached")

    async def backtest(self, stop_on_error: bool = True, **kwargs):
        """执行回测

        Args:
            stop_on_error: 如果为True，则发生异常时，将停止回测。否则忽略错误，继续执行。
        Keyword Args:
            prefetch_stocks Dict[str, BarsArray]: 代码列表。在该列表中的品种，将在回测之前自动预取行情数据，并在调用predict时，传入截止到当前frame的，长度为n的行情数据。行情周期由构造时的frame_type指定。预取数据长度由`self.warmup_period`决定
        """
        prefetch_stocks: List[str] = kwargs.get("prefetch_stocks")  # type: ignore
        await self._cache_bars_for_backtest(prefetch_stocks, self.warmup_period)
        self.bs.cursor = self.bs.start

        intra_day = self._frame_type in tf.minute_level_frames
        converter = tf.int2time if intra_day else tf.int2date

        await self.before_start()

        # 最后一周期不做预测，留出来执行上一周期的信号
        end_ = tf.shift(self.bs.end, -1, self._frame_type)
        for i, frame in enumerate(
            tf.get_frames(self.bs.start, end_, self._frame_type)  # type: ignore
        ):
            barss = self._next()
            day_barss = barss if self._frame_type == FrameType.DAY else None
            frame_ = converter(frame)

            prev_frame = tf.shift(frame_, -1, self._frame_type)
            next_frame = tf.shift(frame_, 1, self._frame_type)

            # new trading day start
            if (not intra_day and prev_frame < frame_) or (
                intra_day and prev_frame.date() < frame_.date()
            ):
                await self.before_trade(frame_, day_barss)

            logger.debug("%sth iteration", i, date=frame_)
            try:
                await self.predict(
                    frame_, self._frame_type, i, barss=barss, **kwargs  # type: ignore
                )
            except Exception as e:
                if isinstance(e, TradeError):
                    logger.warning("call stack is:\n%s", e.stack)
                else:
                    logger.exception(e)
                if stop_on_error:
                    raise e

            # trading day ends
            if (not intra_day and next_frame > frame_) or (
                intra_day and next_frame.date() > frame_.date()
            ):
                await self.after_trade(frame_, day_barss)

        self.broker.stop_backtest()

        await self.after_stop()
        self.bills = self.broker.bills()
        baseline = kwargs.get("baseline", "399300.XSHE")
        self.metrics = self.broker.metrics(baseline=baseline)
        self.bs.baseline = baseline

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
        logger.debug(
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
        logger.debug(
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

    async def before_start(self):
        """策略启动前的准备工作。

        在一次回测中，它会在backtest中、进入循环之前调用。如果策略需要根据过去的数据来计算一些自适应参数，可以在此方法中实现。
        """
        if self.bs is not None:
            logger.info(
                "BEFORE_START: %s<%s - %s>",
                self.name,
                self.bs.start,
                self.bs.end,
                date=self.bs.start,
            )
        else:
            logger.info("BEFORE_START: %s", self.name)

    async def before_trade(self, date: datetime.date, barss: Optional[Dict[str, BarsArray]]=None):
        """每日开盘前的准备工作

        Args:
            date: 日期。在回测中为回测当日日期，在实盘中为系统日期
            barss: 如果主周期为日线，且支持预取，则会将预取的barss传入
        """
        logger.debug("BEFORE_TRADE: %s", self.name, date=date)

    async def after_trade(self, date: Frame, barss: Optional[Dict[str, BarsArray]]=None):
        """每日收盘后的收尾工作

        Args:
            date: 日期。在回测中为回测当日日期，在实盘中为系统日期
            barss: 如果主周期为日线，且支持预取，则会将预取的barss传入
        """
        logger.debug("AFTER_TRADE: %s", self.name, date=date)

    async def after_stop(self):
        if self.bs is not None:
            logger.info(
                "STOP %s<%s - %s>",
                self.name,
                self.bs.start,
                self.bs.end,
                date=self.bs.end,
            )
        else:
            logger.info("STOP %s", self.name)

    async def predict(
        self,
        frame: Frame,
        frame_type: FrameType,
        i: int,
        barss: Optional[Dict[str, BarsArray]] = None,
        **kwargs,
    ):
        """策略评估函数。在此函数中实现交易信号检测和处理。

        Args:
            frame: 当前时间帧
            frame_type: 处理的数据主周期
            i: 当前时间离回测起始的单位数
            barss: 如果调用`backtest`时传入了`portfolio`及参数，则`backtest`将会在回测之前，预取从[start - warmup_period * frame_type, end]间的portfolio行情数据，并在每次调用`predict`方法时，通过`barss`参数，将[start - warmup_period * frame_type, start + i * frame_type]间的数据传给`predict`方法。传入的数据已进行前复权。

        Keyword Args: 在`backtest`方法中的传入的kwargs参数将被透传到此方法中。
        """
        raise NotImplementedError

    @deprecated("2.0.0", details="use `make_report` instead")
    async def plot_metrics(
        self, indicator: Union[pd.DataFrame, List[Tuple], None] = None
    ):
        return await self.make_report(indicator)

    async def make_report(
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

        mg = MetricsGraph(
            self.bills,
            self.metrics,
            indicator=indicator,
            baseline_code=self.bs.baseline,
        )
        await mg.plot()
