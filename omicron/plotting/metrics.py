"""绘制回测资产曲线和指标图。

示例:
```python
from omicron.plotting import MetricsGraph

# calling some strategy's backtest and get bills/metrics
mg = MetricsGraph(bills, metrics)

await mg.plot()
```
注意此方法需要在notebook中调用。
![](https://images.jieyu.ai/images/2023/05/20230508160012.png)

"""
import datetime
import logging
from collections import defaultdict
from copy import deepcopy
from typing import List, Optional, Union

import arrow
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from coretypes import BarsArray, Frame, FrameType
from numpy.typing import NDArray
from plotly.subplots import make_subplots

from omicron import tf
from omicron.extensions import fill_nan
from omicron.models.security import Security
from omicron.models.stock import Stock

logger = logging.getLogger(__name__)


class MetricsGraph:
    def __init__(
        self,
        bills: dict,
        metrics: dict,
        baseline_code: str = "399300.XSHE",
        indicator: Optional[pd.DataFrame] = None,
    ):
        """
        Args:
            bills: 回测生成的账单，通过Strategy.bills获得
            metrics: 回测生成的指标，通过strategy.metrics获得
            baseline_code: 基准证券代码
            indicator: 回测时使用的指标。如果存在，将叠加到策略回测图上。它应该是一个以日期为索引，指标值列名为"value"的pandas.DataFrame。如果不提供，将不会绘制指标图
        """
        self.metrics = metrics
        self.trades = bills["trades"]
        self.positions = bills["positions"]
        self.start = arrow.get(bills["assets"][0][0]).date()
        self.end = arrow.get(bills["assets"][-1][0]).date()

        self.frames = [
            tf.int2date(f) for f in tf.get_frames(self.start, self.end, FrameType.DAY)
        ]

        if indicator is not None:
            self.indicator = indicator.join(
                pd.Series(index=self.frames, name="frames", dtype=np.float64),
                how="right",
            )
        else:
            self.indicator = None

        # 记录日期到下标的反向映射
        self._frame2pos = {f: i for i, f in enumerate(self.frames)}
        self.ticks = self._format_tick(self.frames)

        # TODO: there's bug in backtesting, temporarily fix here
        df = pd.DataFrame(self.frames, columns=["frame"])
        df["assets"] = np.nan
        assets = pd.DataFrame(bills["assets"], columns=["frame", "assets"])
        df["assets"] = assets["assets"]
        self.assets = df.fillna(method="ffill")["assets"].to_numpy()
        self.nv = self.assets / self.assets[0]

        self.baseline_code = baseline_code or "399300.XSHE"

    def _fill_missing_prices(self, bars: BarsArray, frames: Union[List, NDArray]):
        """将bars中缺失值采用其前值替换

        当baseline为个股时，可能存在停牌的情况，这样导致由此计算的参考收益无法与回测的资产收益对齐，因此需要进行调整。

        出于这个目的，本函数只返回处理后的收盘价。

        Args:
            bars: 基线行情数据。
            frames: 日期索引

        Returns:
            补充缺失值后的收盘价序列
        """
        _close = pd.DataFrame(
            {
                "close": pd.Series(bars["close"], index=bars["frame"]),
                "frame": pd.Series(np.empty((len(frames),)), index=frames),
            }
        )["close"].to_numpy()

        # 这里使用omicron中的fill_nan，是因为如果数组的第一个元素即为NaN的话，那么DataFrame.fillna(method='ffill')将无法处理这样的情况(仍然保持为nan)

        return fill_nan(_close)

    def _format_tick(self, frames: Union[Frame, List[Frame]]) -> Union[str, NDArray]:
        if type(frames) == datetime.date:
            x = frames
            return f"{x.year:02}-{x.month:02}-{x.day:02}"
        elif type(frames) == datetime.datetime:
            x = frames
            return f"{x.month:02}-{x.day:02} {x.hour:02}:{x.minute:02}"
        elif type(frames[0]) == datetime.date:  # type: ignore
            return np.array([f"{x.year:02}-{x.month:02}-{x.day:02}" for x in frames])
        else:
            return np.array(
                [f"{x.month:02}-{x.day:02} {x.hour:02}:{x.minute:02}" for x in frames]  # type: ignore
            )

    async def _metrics_trace(self):
        metric_names = {
            "start": "起始日",
            "end": "结束日",
            "window": "资产暴露窗口",
            "total_tx": "交易次数",
            "total_profit": "总利润",
            "total_profit_rate": "利润率",
            "win_rate": "胜率",
            "mean_return": "日均回报",
            "sharpe": "夏普率",
            "max_drawdown": "最大回撤",
            "annual_return": "年化回报",
            "volatility": "波动率",
            "sortino": "sortino",
            "calmar": "calmar",
        }

        # bug: plotly go.Table.Cells format not work here
        metric_formatter = {
            "start": "{}",
            "end": "{}",
            "window": "{}",
            "total_tx": "{}",
            "total_profit": "{:.2f}",
            "total_profit_rate": "{:.2%}",
            "win_rate": "{:.2%}",
            "mean_return": "{:.2%}",
            "sharpe": "{:.2f}",
            "max_drawdown": "{:.2%}",
            "annual_return": "{:.2%}",
            "volatility": "{:.2%}",
            "sortino": "{:.2f}",
            "calmar": "{:.2f}",
        }

        metrics = deepcopy(self.metrics)
        baseline = metrics["baseline"] or {}
        del metrics["baseline"]

        baseline_name = (
            await Security.alias(self.baseline_code) if self.baseline_code else "基准"
        )

        metrics_formatted = []
        for k in metric_names.keys():
            if metrics.get(k):
                metrics_formatted.append(metric_formatter[k].format(metrics.get(k)))
            else:
                metrics_formatted.append("-")

        baseline_formatted = []
        for k in metric_names.keys():
            if baseline.get(k):
                baseline_formatted.append(metric_formatter[k].format(baseline.get(k)))
            else:
                baseline_formatted.append("-")

        return go.Table(
            header=dict(values=["指标名", "策略", baseline_name]),
            cells=dict(
                values=[
                    [v for _, v in metric_names.items()],
                    metrics_formatted,
                    baseline_formatted,
                ],
                font_size=10,
            ),
        )

    async def _trade_info_trace(self):
        """构建hover text 序列"""
        # convert trades into hover_info
        buys = defaultdict(list)
        sells = defaultdict(list)
        for _, trade in self.trades.items():
            trade_date = arrow.get(trade["time"]).date()

            ipos = self._frame2pos.get(trade_date)
            if ipos is None:
                logger.warning(
                    "date  %s in trade record not in backtest range", trade_date
                )
                continue

            name = await Security.alias(trade["security"])
            price = trade["price"]
            side = trade["order_side"]
            filled = trade["filled"]

            trade_text = f"{side}:{name} {filled/100:.0f}手 价格:{price:.02f} 成交额:{filled * price/10000:.1f}万"

            if side == "卖出":
                sells[trade_date].append(trade_text)
            elif side in ("买入", "分红配股"):
                buys[trade_date].append(trade_text)

        X_buy, Y_buy, data_buy = [], [], []
        X_sell, Y_sell, data_sell = [], [], []

        for dt, text in buys.items():
            ipos = self._frame2pos.get(dt)
            Y_buy.append(self.nv[ipos])
            X_buy.append(self._format_tick(dt))

            asset = self.assets[ipos]
            hover = f"资产:{asset/10000:.1f}万<br>{'<br>'.join(text)}"
            data_buy.append(hover)

        trace_buy = go.Scatter(
            x=X_buy,
            y=Y_buy,
            mode="markers",
            text=data_buy,
            name="买入成交",
            marker=dict(color="red", symbol="triangle-up"),
            hovertemplate="<br>%{text}",
        )

        for dt, text in sells.items():
            ipos = self._frame2pos.get(dt)
            Y_sell.append(self.nv[ipos])
            X_sell.append(self._format_tick(dt))

            asset = self.assets[ipos]
            hover = f"资产:{asset/10000:.1f}万<br>{'<br>'.join(text)}"
            data_sell.append(hover)

        trace_sell = go.Scatter(
            x=X_sell,
            y=Y_sell,
            mode="markers",
            text=data_sell,
            name="卖出成交",
            marker=dict(color="green", symbol="triangle-down"),
            hovertemplate="<br>%{text}",
        )

        return trace_buy, trace_sell

    async def plot(self):
        """绘制资产曲线及回测指标图"""
        n = len(self.assets)
        bars = await Stock.get_bars(self.baseline_code, n, FrameType.DAY, self.end)

        baseline_prices = self._fill_missing_prices(bars, self.frames)
        baseline_prices /= baseline_prices[0]

        fig = make_subplots(
            rows=1,
            cols=2,
            shared_xaxes=False,
            specs=[
                [{"secondary_y": True}, {"type": "table"}],
            ],
            column_width=[0.75, 0.25],
            horizontal_spacing=0.01,
            subplot_titles=("资产曲线", "策略指标"),
        )

        fig.add_trace(await self._metrics_trace(), row=1, col=2)

        if self.indicator is not None:
            indicator_on_hover = self.indicator["value"]
        else:
            indicator_on_hover = None

        baseline_name = (
            await Security.alias(self.baseline_code) if self.baseline_code else "基准"
        )

        baseline_trace = go.Scatter(
            y=baseline_prices,
            x=self.ticks,
            mode="lines",
            name=baseline_name,
            showlegend=True,
            text=indicator_on_hover,
            hovertemplate="<br>净值:%{y:.2f}" + "<br>指标:%{text:.1f}",
        )
        fig.add_trace(baseline_trace, row=1, col=1)

        nv_trace = go.Scatter(
            y=self.nv,
            x=self.ticks,
            mode="lines",
            name="策略",
            showlegend=True,
            hovertemplate="<br>净值:%{y:.2f}",
        )
        fig.add_trace(nv_trace, row=1, col=1)

        if self.indicator is not None:
            ind_trace = go.Scatter(
                y=self.indicator["value"],
                x=self.ticks,
                mode="lines",
                name="indicator",
                showlegend=True,
                visible="legendonly",
            )
            fig.add_trace(ind_trace, row=1, col=1, secondary_y=True)

        for trace in await self._trade_info_trace():
            fig.add_trace(trace, row=1, col=1)

        fig.update_xaxes(type="category", tickangle=45, nticks=len(self.ticks) // 5)
        fig.update_layout(margin=dict(l=20, r=20, t=50, b=50), width=1040, height=435)
        fig.update_layout(
            hovermode="x unified", hoverlabel=dict(bgcolor="rgba(255,255,255,0.8)")
        )
        fig.show()
