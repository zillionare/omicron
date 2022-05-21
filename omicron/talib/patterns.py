"""
一些形态检测方法
"""
from enum import IntEnum
from typing import Callable, Tuple

import numpy as np
from zigzag import peak_valley_pivots

from omicron.talib.common import smallest_n_argpos, top_n_argpos


class BreakoutFlag(IntEnum):
    UP = 1
    DOWN = -1
    NONE = 0


def peaks_and_valleys(
    ts: np.ndarray, up_thresh: float, down_thresh: float
) -> np.ndarray:
    """
    寻找ts中的波峰和波谷，返回数组指示在该位置上是否为波峰或波谷。如果为1，则为波峰；如果为-1，则为波谷。

    本函数直接使用了zigzag中的peak_valley_pivots. 有很多方法可以实现本功能，比如scipy.signals.find_peaks_cwt, peak_valley_pivots等。本函数更适合金融时间序列，并且使用了cython加速。

    Args:
        ts (np.ndarray): 时间序列
        up_thresh (float): 波峰的阈值
        down_thresh (float): 波谷的阈值

    Returns:
        np.ndarray: 返回数组指示在该位置上是否为波峰或波谷。
    """
    if ts.dtype != np.float64:
        ts = ts.astype(np.float64)

    return peak_valley_pivots(ts, up_thresh, down_thresh)


def support_resist_lines(
    ts: np.ndarray, upthres: float = 0.01, downthres: float = -0.01
) -> Tuple[Callable, Callable, np.ndarray]:
    """计算时间序列的支撑线和阻力线

    Examples:
        ```python
            def show_support_resist_lines(ts):
                import plotly.graph_objects as go

                fig = go.Figure()

                support, resist, x_start = support_resist_lines(ts, 0.03, -0.03)
                fig.add_trace(go.Scatter(x=np.arange(len(ts)), y=ts))

                x = np.arange(len(ts))[x_start:]
                fig.add_trace(go.Line(x=x, y = support(x)))
                fig.add_trace(go.Line(x=x, y = resist(x)))

                fig.show()

            np.random.seed(1978)
            X = np.cumprod(1 + np.random.randn(100) * 0.01)
            show_support_resist_lines(X)
        ```
        the above code will show this ![](https://images.jieyu.ai/images/202204/support_resist.png)

    Args:
        ts (np.ndarray): 时间序列
        upthres (float, optional): 请参考[peaks_and_valleys][omicron.talib.patterns.peaks_and_valleys]
        downthres (float, optional): 请参考[peaks_and_valleys][omicron.talib.patterns.peaks_and_valleys]

    Returns:
        返回支撑线和阻力线的计算函数及起始点坐标，如果没有支撑线或阻力线，则返回None
    """
    if ts.dtype != np.float64:
        ts = ts.astype(np.float64)

    pivots = peak_valley_pivots(ts, upthres, downthres)

    peaks_only = np.select([pivots == 1], [ts], 0)
    arg_max = sorted(top_n_argpos(peaks_only, 2))

    valleys_only = np.select([pivots == -1], [ts], np.max(ts))
    arg_min = sorted(smallest_n_argpos(valleys_only, 2))

    resist = None
    support = None

    if len(arg_max) >= 2:
        y = ts[arg_max]
        coeff = np.polyfit(arg_max, y, deg=1)

        resist = np.poly1d(coeff)

    if len(arg_min) >= 2:
        y = ts[arg_min]
        coeff = np.polyfit(arg_min, y, deg=1)

        support = np.poly1d(coeff)

    return support, resist, np.min([*arg_min, *arg_max])


def breakout(
    ts: np.ndarray, upthres: float = 0.01, downthres: float = -0.01, confirm: int = 1
) -> BreakoutFlag:
    """检测时间序列是否突破了压力线（整理线）

    Args:
        ts (np.ndarray): 时间序列
        upthres (float, optional): 请参考[peaks_and_valleys][omicron.talib.patterns.peaks_and_valleys]
        downthres (float, optional): 请参考[peaks_and_valleys][omicron.talib.patterns.peaks_and_valleys]
        confirm (int, optional): 经过多少个bars后，才确认突破。默认为1

    Returns:
        如果上向突破压力线，返回1，如果向下突破压力线，返回-1，否则返回0
    """
    support, resist, _ = support_resist_lines(ts[:-confirm], upthres, downthres)

    x0 = len(ts) - confirm - 1
    x = list(range(len(ts) - confirm, len(ts)))

    if resist is not None:
        if np.all(ts[x] > resist(x)) and ts[x0] <= resist(x0):
            return BreakoutFlag.UP

    if support is not None:
        if np.all(ts[x] < support(x)) and ts[x0] >= support(x0):
            return BreakoutFlag.DOWN

    return BreakoutFlag.NONE
