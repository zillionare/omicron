"""绘制回测资产曲线和指标图。

示例:
```python
from omicron.plotting import MetricsGraph

# calling some strategy's backtest and get bills/metrics
mg = MetricsGraph(bills, metrics)

await mg.plot()
```
注意此方法需要在notebook中调用。

"""
import datetime
import logging
from collections import defaultdict
from copy import deepcopy
from typing import List, Union

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
    def __init__(self, bills: dict, metrics: dict):
        self.metrics = metrics
        self.trades = bills["trades"]
        self.positions = bills["positions"]
        self.start = arrow.get(bills["assets"][0][0]).date()
        self.end = arrow.get(bills["assets"][-1][0]).date()

        self.frames = [
            tf.int2date(f) for f in tf.get_frames(self.start, self.end, FrameType.DAY)
        ]

        # 记录日期到下标的反向映射，这对于在不o
        self._frame2pos = {f: i for i, f in enumerate(self.frames)}
        self.ticks = self._format_tick(self.frames)

        self.assets = pd.DataFrame(bills["assets"], columns=["frame", "assets"])[
            "assets"
        ].to_numpy()
        self.nv = self.assets / self.assets[0]

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

    async def _metrics_traces(self):
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
            "code": "参照标的",
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
            "code": "{}",
        }

        metrics = deepcopy(self.metrics)
        baseline = metrics["baseline"]
        del metrics["baseline"]

        traces = []
        traces.append(
            go.Table(
                header=dict(values=["指标名", "指标值"]),
                cells=dict(
                    values=[
                        [metric_names[k] for k in metrics if metrics[k]],
                        [
                            metric_formatter[k].format(metrics[k])
                            for k in metrics
                            if metrics[k]
                        ],
                    ],
                    font_size=10,
                ),
            )
        )

        if baseline:
            baseline = {k: v for k, v in baseline.items() if v is not None}
            if "code" in baseline:
                baseline["code"] = await Security.alias(baseline["code"])

            traces.append(
                go.Table(
                    header=dict(values=["指标名", "指标值"]),
                    cells=dict(
                        values=[
                            [metric_names[k] for k in baseline if baseline[k]],
                            [
                                metric_formatter[k].format(baseline[k])
                                for k in baseline
                                if baseline[k]
                            ],
                        ],
                        font_size=10,
                    ),
                )
            )

        return traces

    async def _trade_info_trace(self):
        """构建hover text 序列"""
        X = []
        Y = []
        data = []

        # convert trades into hover_info
        merged = defaultdict(list)
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

            trade_text = f"{side}:{name} {filled/100:.0f}手 价格:{price:.02f} 成交额{filled * price/10000:.1f}万"

            merged[trade_date].append(trade_text)

        for dt, text in merged.items():
            ipos = self._frame2pos.get(dt)
            Y.append(self.nv[ipos])
            X.append(self._format_tick(dt))

            asset = self.assets[ipos]
            hover = f"资产:{asset/10000:.1f}万<br>{'<br>'.join(text)}"
            data.append(hover)

        trace = go.Scatter(x=X, y=Y, mode="markers", text=data, name="交易详情")
        return trace

    async def plot(self, baseline_code: str = "399300.XSHE"):
        """绘制资产曲线及回测指标图"""
        n = len(self.assets)
        bars = await Stock.get_bars(baseline_code, n, FrameType.DAY, self.end)

        baseline_prices = self._fill_missing_prices(bars, self.frames)
        baseline_prices /= baseline_prices[0]

        fig = make_subplots(
            rows=2,
            cols=2,
            shared_xaxes=False,
            specs=[
                [{"type": "scatter", "rowspan": 2}, {"type": "table"}],
                [None, {"type": "table"}],
            ],
            column_width=[0.8, 0.2],
            row_heights=[0.63, 0.4],
            horizontal_spacing=0.01,
            vertical_spacing=0.07,
            subplot_titles=("资产曲线", "策略指标", "基线指标"),
        )

        for i, trace in enumerate(await self._metrics_traces()):
            fig.add_trace(trace, row=i + 1, col=2)

        print("baseline", len(baseline_prices))
        baseline_trace = go.Scatter(
            y=baseline_prices,
            x=self.ticks,
            mode="lines",
            name="baseline",
            showlegend=True,
        )
        fig.add_trace(baseline_trace, row=1, col=1)

        nv_trace = go.Scatter(
            y=self.nv, x=self.ticks, mode="lines", name="策略净值", showlegend=True
        )
        fig.add_trace(nv_trace, row=1, col=1)

        trade_info_trace = await self._trade_info_trace()
        fig.add_trace(trade_info_trace, row=1, col=1)

        fig.update_xaxes(type="category", tickangle=45, nticks=len(self.ticks) // 5)
        fig.update_layout(margin=dict(l=20, r=20, t=50, b=50), width=1040, height=650)
        fig.show()
