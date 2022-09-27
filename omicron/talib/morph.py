"""形态检测相关方法"""
from enum import IntEnum
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
import talib as ta
from zigzag import peak_valley_pivots

from omicron.extensions.np import smallest_n_argpos, top_n_argpos
from omicron.talib.core import clustering


class CrossFlag(IntEnum):
    UPCROSS = 1
    DOWNCROSS = -1
    NONE = 0


def cross(f: np.ndarray, g: np.ndarray) -> CrossFlag:
    """判断序列f是否与g相交。如果两个序列有且仅有一个交点，则返回1表明f上交g；-1表明f下交g

    本方法可用以判断两条均线是否相交。

    returns:
        (flag, index), 其中flag取值为：
        0 无效
        -1 f向下交叉g
        1 f向上交叉g
    """
    indices = np.argwhere(np.diff(np.sign(f - g))).flatten()

    if len(indices) == 0:
        return CrossFlag.NONE, 0

    # 如果存在一个或者多个交点，取最后一个
    idx = indices[-1]

    if f[idx] < g[idx]:
        return CrossFlag.UPCROSS, idx
    elif f[idx] > g[idx]:
        return CrossFlag.DOWNCROSS, idx
    else:
        return CrossFlag(np.sign(g[idx - 1] - f[idx - 1])), idx


def vcross(f: np.array, g: np.array) -> Tuple:
    """判断序列f是否与g存在类型v型的相交。即存在两个交点，第一个交点为向下相交，第二个交点为向上
    相交。一般反映为洗盘拉升的特征。

    Examples:

        >>> f = np.array([ 3 * i ** 2 - 20 * i +  2 for i in range(10)])
        >>> g = np.array([ i - 5 for i in range(10)])
        >>> flag, indices = vcross(f, g)
        >>> assert flag is True
        >>> assert indices[0] == 0
        >>> assert indices[1] == 6

    Args:
        f: first sequence
        g: the second sequence

    Returns:
        (flag, indices), 其中flag取值为True时，存在vcross，indices为交点的索引。
    """
    indices = np.argwhere(np.diff(np.sign(f - g))).flatten()
    if len(indices) == 2:
        idx0, idx1 = indices
        if f[idx0] > g[idx0] and f[idx1] < g[idx1]:
            return True, (idx0, idx1)

    return False, (None, None)


def inverse_vcross(f: np.array, g: np.array) -> Tuple:
    """判断序列f是否与序列g存在^型相交。即存在两个交点，第一个交点为向上相交，第二个交点为向下
    相交。可用于判断见顶特征等场合。

    Args:
        f (np.array): [description]
        g (np.array): [description]

    Returns:
        Tuple: [description]
    """
    indices = np.argwhere(np.diff(np.sign(f - g))).flatten()
    if len(indices) == 2:
        idx0, idx1 = indices
        if f[idx0] < g[idx0] and f[idx1] > g[idx1]:
            return True, (idx0, idx1)

    return False, (None, None)


class BreakoutFlag(IntEnum):
    UP = 1
    DOWN = -1
    NONE = 0


def peaks_and_valleys(
    ts: np.ndarray, up_thresh: float = None, down_thresh: float = None
) -> np.ndarray:
    """寻找ts中的波峰和波谷，返回数组指示在该位置上是否为波峰或波谷。如果为1，则为波峰；如果为-1，则为波谷。

    本函数直接使用了zigzag中的peak_valley_pivots. 有很多方法可以实现本功能，比如scipy.signals.find_peaks_cwt, peak_valley_pivots等。本函数更适合金融时间序列，并且使用了cython加速。

    Args:
        ts (np.ndarray): 时间序列
        up_thresh (float): 波峰的阈值，如果为None,则使用ts变化率的二倍标准差
        down_thresh (float): 波谷的阈值，如果为None,则使用ts变化率的二倍标准差乘以-1

    Returns:
        np.ndarray: 返回数组指示在该位置上是否为波峰或波谷。
    """
    if ts.dtype != np.float64:
        ts = ts.astype(np.float64)

    if any([up_thresh is None, down_thresh is None]):
        change_rate = ts[1:] / ts[:-1] - 1
        std = np.std(change_rate)
        up_thresh = up_thresh or 2 * std
        down_thresh = down_thresh or -2 * std

    return peak_valley_pivots(ts, up_thresh, down_thresh)


def support_resist_lines(
    ts: np.ndarray, upthres: float = None, downthres: float = None
) -> Tuple[Callable, Callable, np.ndarray]:
    """计算时间序列的支撑线和阻力线

    使用最近的两个高点连接成阴力线，两个低点连接成支撑线。

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

    pivots = peaks_and_valleys(ts, upthres, downthres)
    pivots[0] = 0
    pivots[-1] = 0

    arg_max = np.argwhere(pivots == 1).flatten()
    arg_min = np.argwhere(pivots == -1).flatten()

    resist = None
    support = None

    if len(arg_max) >= 2:
        arg_max = arg_max[-2:]
        y = ts[arg_max]
        coeff = np.polyfit(arg_max, y, deg=1)

        resist = np.poly1d(coeff)

    if len(arg_min) >= 2:
        arg_min = arg_min[-2:]
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


def plateaus(
    numbers: np.ndarray, min_size: int, fall_in_range_ratio: float = 0.97
) -> List[Tuple]:
    """统计数组`numbers`中的可能存在的平台整理。

    如果一个数组中存在相邻的一个子数组，其中超过`fall_in_range_ratio`的元素都落在二个标准差以内，则认为该出有平台

    Args:
        numbers: 输入数组
        min_size: 平台的最小长度
        fall_in_range_ratio: 超过`fall_in_range_ratio`比例的元素落在二个标准差以内，就认为该子数组构成一个平台

    Returns:
        平台的起始位置和长度的数组
    """
    if numbers.size <= min_size:
        n = 1
    else:
        n = numbers.size // min_size

    clusters = clustering(numbers, n)

    plats = []
    for (start, length) in clusters:
        if length < min_size:
            continue

        y = numbers[start : start + length]
        mean = np.mean(y)
        std = np.std(y)

        inrange = len(y[np.abs(y - mean) < 2 * std])
        ratio = inrange / length

        if ratio >= fall_in_range_ratio:
            plats.append((start, length))

    return plats


def rsi_bottom_dev_detect(
    close: np.ndarray, thresh: Tuple[float, float] = None, rsi_limit: float = 30
) -> Tuple[int, int]:
    """寻找rsi底背离

    返回一个Tuple，其中第一个值指示在该位置发生背离的类型：0表示没有发生背离，1，表明出现了直接底背离，2表明出现了间隔底背离。
    第二个表示监测点距离底背离发生点的最近时间单位。

    Args:
        close (np.ndarray): 时间序列收盘价
        thresh (Tuple[float, float]): 请参考[peaks_and_valleys][omicron.talib.morph.peaks_and_valleys]
        rsi_limit (float, optional): RSI发生底背离时的阈值

    Returns:
        返回一个Tuple，其中第一个值指示在该位置发生背离的类型：0表示没有发生背离，1，表明出现了直接底背离，2表明出现了间隔底背离。
        数组第二个值表示最后底背离点距最终时间的时间单位，在没有底背离的情况下，返回None。
    """
    assert len(close) >= 60, "must provide an array with at least 61 length!"
    if close.dtype != np.float64:
        close = close.astype(np.float64)
    rsi = ta.RSI(close, 6)

    if thresh is None:
        std = np.std(close[-59:] / close[-60:-1] - 1)
        thresh = (2 * std, -2 * std)

    pivots = peak_valley_pivots(close, thresh[0], thresh[1])
    pivots[0], pivots[-1] = 0, 0  # 掐头去尾
    valley_pivots = -1 * (((pivots == -1) & (rsi <= rsi_limit)).astype("int"))
    bottom_dev_type = 0
    bottom_dev_distance = None
    length = len(valley_pivots)
    valley_index = np.where(valley_pivots == -1)[0]
    if len(valley_index) >= 2:  # 单个底背离
        if (close[valley_index[-1]] - close[valley_index[-2]]) * (
            rsi[valley_index[-1]] - rsi[valley_index[-2]]
        ) < 0:
            bottom_dev_type = 1
            bottom_dev_distance = length - 1 - valley_index[-1]

        elif len(valley_index) >= 3:  # 间隔背离点
            if (close[valley_index[-1]] - close[valley_index[-3]]) * (
                rsi[valley_index[-1]] - rsi[valley_index[-3]]
            ) < 0:
                bottom_dev_type = 2
                bottom_dev_distance = length - 1 - valley_index[-1]
        else:
            pass
    return bottom_dev_type, bottom_dev_distance


def rsi_watermarks(
    close: np.ndarray, thresh: Tuple[float, float] = None
) -> Tuple[float, float]:
    """给定一段行情数据和用以检测顶和底的阈值，返回该段行情中，谷和峰处RSI均值。

    其中bars的长度一般不小于60，不大于120。返回值中，前一个为low_wartermark（谷底处RSI值），
    后一个为high_wartermark（高峰处RSI值)。

    Args:
        close (np.ndarray): 具有时间序列的收盘价
        thresh (Tuple[float, float]) : 请参考[peaks_and_valleys][omicron.talib.morph.peaks_and_valleys]

    Returns:
        返回数组[low_watermark, high_watermark], 第一个为最近两个最低收盘价的RSI均值， 第二个为最近两个最高收盘价的RSI均值。
        若传入收盘价只有一个最值，则只返回一个。没有最值，则返回None。
    """
    assert len(close) >= 60, "must provide an array with at least 60 length!"

    if thresh is None:
        std = np.std(close[-59:] / close[-60:-1] - 1)
        thresh = (2 * std, -2 * std)

    if close.dtype != np.float64:
        close = close.astype(np.float64)
    rsi = ta.RSI(close, 6)

    pivots = peak_valley_pivots(close, thresh[0], thresh[1])
    pivots[0], pivots[-1] = 0, 0  # 掐头去尾

    # 峰值RSI>70; 谷处的RSI<30;
    peaks_rsi_index = np.where((rsi > 70) & (pivots == 1))[0]
    valleys_rsi_index = np.where((rsi < 30) & (pivots == -1))[0]

    if len(peaks_rsi_index) == 0:
        high_watermark = None
    elif len(peaks_rsi_index) == 1:
        high_watermark = rsi[peaks_rsi_index[0]]
    else:  # 有两个以上的峰，通过最近的两个峰均值来确定走势
        high_watermark = np.nanmean(rsi[peaks_rsi_index[-2:]])

    if len(valleys_rsi_index) == 0:
        low_watermark = None
    elif len(valleys_rsi_index) == 1:
        low_watermark = rsi[valleys_rsi_index[0]]
    else:  # 有两个以上的峰，通过最近的两个峰来确定走势
        low_watermark = np.nanmean(rsi[valleys_rsi_index[-2:]])

    return low_watermark, high_watermark


def rsi_predict_price(
    close: np.ndarray, thresh: Tuple[float, float] = None
) -> Tuple[float, float]:
    """给定一段行情，根据最近的两个RSI的极小值和极大值预测下一个周期可能达到的最低价格和最高价格。

    其原理是，以预测最近的两个最高价和最低价，求出其相对应的RSI值，求出最高价和最低价RSI的均值，
    若只有一个则取最近的一个。再由RSI公式，反推价格。此时返回值为(None, float)，即只有最高价，没有最低价。反之亦然。

    Args:
        close (np.ndarray): 具有时间序列的收盘价
        thresh (Tuple[float, float]) : 请参考[peaks_and_valleys][omicron.talib.morph.peaks_and_valleys]

    Returns:
        返回数组[predicted_low_price, predicted_high_price], 数组第一个值为利用达到之前最低收盘价的RSI预测的最低价。
        第二个值为利用达到之前最高收盘价的RSI预测的最高价。
    """
    assert len(close) >= 60, "must provide an array with at least 60 length!"

    if thresh is None:
        std = np.std(close[-59:] / close[-60:-1] - 1)
        thresh = (2 * std, -2 * std)

    if close.dtype != np.float64:
        close = close.astype(np.float64)

    valley_rsi, peak_rsi = rsi_watermarks(close, thresh=thresh)
    pivot = peak_valley_pivots(close, thresh[0], thresh[1])
    pivot[0], pivot[-1] = 0, 0  # 掐头去尾

    price_change = pd.Series(close).diff(1).values
    ave_price_change = (abs(price_change)[-6:].mean()) * 5
    ave_price_raise = (np.maximum(price_change, 0)[-6:].mean()) * 5

    if valley_rsi is not None:
        predicted_low_change = (ave_price_change) - ave_price_raise / (
            0.01 * valley_rsi
        )
        if predicted_low_change > 0:
            predicted_low_change = 0
        predicted_low_price = close[-1] + predicted_low_change
    else:
        predicted_low_price = None

    if peak_rsi is not None:
        predicted_high_change = (ave_price_raise - ave_price_change) / (
            0.01 * peak_rsi - 1
        ) - ave_price_change
        if predicted_high_change < 0:
            predicted_high_change = 0
        predicted_high_price = close[-1] + predicted_high_change
    else:
        predicted_high_price = None

    return predicted_low_price, predicted_high_price
