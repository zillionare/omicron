"""策略基类

# 子类派生

1. 从此基类派生出一个策略子类，比如sma.py
2. 子类需要重载`predict`方法，根据当前传入的时间帧和帧类型参数，获取数据并进行处理，评估出交易信号
3. 子类根据交易信号，在`predict`方法里，调用基类的`buy`和`sell`方法来进行交易
4. 子类调用`backtest`方法来进行回测，该方法将根据策略构建时指定的回测起始时间、终止时间、帧类型，逐帧生成各个时间帧，并调用子类的`predict`方法
4. 在交易结束时，调用`plot_metrics`方法来获取如下所示的回测指标图
![](https://images.jieyu.ai/images/2023/05/20230508160012.png)

如何派生子类，可以参考[sma][omicron.strategy.sma.SMAStrategy]源代码。

# 回测
```python
from omicron.strategy.sma import SMAStrategy
sma = SMAStrategy(
    "600000.XSHG",
    url="", # the url of either backtest server, or trade server
    is_backtest=True,
    start=datetime.date(2023, 2, 3),
    end=datetime.date(2023, 4, 28),
    frame_type=FrameType.DAY,
)

await sma.backtest(stop_on_error=True)
```
# 实盘
在实盘环境下，你还需要在子类中加入周期性任务(比如每分钟执行一次），在该任务中调用`predict`方法来完成交易。
"""
import datetime
import logging
import uuid
from typing import Dict, List, Optional, Union

import jqdatasdk as jq
import numpy as np
from coretypes import Frame, FrameType
from traderclient import TraderClient

from omicron import tf
from omicron.models.security import Security
from omicron.models.stock import Stock
from omicron.plotting.metrics import MetricsGraph

logger = logging.getLogger(__name__)


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
        frame_type: Optional[Frame] = None,
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
            self.bills = None
            self.metrics = None
            self._bt_start = start
            self._bt_end = end
            self._frame_type = frame_type
            self.broker = TraderClient(
                url,
                self.account,
                self.token,
                is_backtest=True,
                start=self._bt_start,
                end=self._bt_end,
            )
            self._baseline = baseline
        else:
            if account is None or token is None:
                raise ValueError("account and token must be presented.")

            self.broker = TraderClient(url, self.account, self.token, is_backtest=False)

    async def backtest(self, stop_on_error: bool = False, *args, **kwargs):
        converter = (
            tf.int2date if self._frame_type in tf.day_level_frames else tf.int2time
        )
        for i, frame in enumerate(
            tf.get_frames(self._bt_start, self._bt_end, self._frame_type)  # type: ignore
        ):
            try:
                await self.predict(
                    converter(frame), self._frame_type, i, *args, **kwargs  # type: ignore
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
        return self.broker.available_money

    @property
    def positions(self):
        return self.broker.positions

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
        self, frame: Frame, frame_type: FrameType, i: int, *args, **kwargs
    ):
        """策略评估函数。在此函数中实现交易信号检测和处理。

        Args:
            frame: 当前时间帧
            frame_type: 处理的数据主周期
            i: 当前时间离回测起始的单位数
        """
        raise NotImplementedError

    async def plot_metrics(self):
        if self.bills is None or self.metrics is None:
            raise ValueError("Please run `start_backtest` first.")

        mg = MetricsGraph(self.bills, self.metrics)
        await mg.plot()
