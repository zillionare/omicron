"""
各类动量因子。
"""
from typing import Dict, Union

import numpy as np

from .maths import NormMethod, momentum, moving_average, norm, polyfit, resample, slope


def ma_slope(ts: np.array, win: int, line_len: int = 4) -> Union[None, Dict]:
    """取``line_len``个样本数据的``win``周期均线进行直线拟合，提取``slope``和``err``因子。

    如果``err``较大，则信号无效。否则``slope``为正时，时间序列``ts``处于上升趋势；反之处于下
    降趋势。

    要求``len(ts) > win + line_len``

    归一化：
        调用者应该在获取足够多的数据之后，进行归一化。由于不同级别（月线、日线和分钟线）的
        K线的涨跌幅度不一样，本函数无法合理地进行归一化。

    返回值为一个集合，其中key为因子名称。

    Args:
        ts: 时间序列，比如收盘价
        win: 均线周期，比如5,10,20,60,120,250是常见的均线周期
        line_len: 使用多少个样本来拟合直线
    """
    ma = moving_average(ts, win)
    if len(ma) < line_len:
        return None

    a, err = slope(ma[-line_len:])
    features = {f"mom_ma_slope_{win}_err": err, f"mom_ma_slope_{win}": a}

    return features


def ma_acc(ts: np.array, win: int, line_len: int = 7):
    """取``line_len``个样本的均线数据进行二次曲线拟合，将err, a, -b/2a作为因子。

    如果err过大则信号无效；否则，当a为正时，价格加速上涨（或者下跌减缓）；当a为负时，价格加速
    下跌（或者上涨减缓）。其中``-b/2a``为顶（底）点。如果``-b/2a > line_len - 1``，说明顶（底）
    点还未确认，此时仍有变盘、破坏走势的可能。如果line_len为7，则严格确认以``-b/2a <= 4``为宜
    ，此时距顶（底）点已过了两个周期。

    本方法更适用于(5,10,20)均线。对更长周期的均线意义不大，可以考虑一次曲线拟合提取特征。

    归一化：
        对顶底点使用tanh进行归一化，避免个别情况下出现极大、极小（负值）的x值。在归一化时，对
        顶底点坐标进行了缩放，以保证落在[-line_len, +line_len]区间的顶底点有更好的响应。

    Args:
        ts: [description]
        win: [description]
        line_len: [description]. Defaults to 7.
    """
    ma = moving_average(ts, win)

    if len(ma) < line_len:
        return None

    normed = norm(ma[-line_len:])
    err, (a, b, c), vert = polyfit(normed, deg=2)

    features = {
        f"mom_ma_acc_{win}_err": err,
        f"mom_ma_acc_{win}_a": a,
        f"mom_ma_acc_{win}_x": np.tanh(vert[0] / line_len),
    }

    return features


def vol_change(vol: np.array, scale=20) -> Union[None, Dict]:
    """各以[5,10,20]周期为一组，提取最近两周期的量能变化特征

    根据观察得知，量能变化可能很剧烈（对于短周期，比如5分钟线，增量100倍或者缩量100倍都是很常见
    的），同时，量能既可能增加，也可能减少。为了便于比较，对量能变化使用带缩放的tanh进行了归一
    化。这里建议的经验值是20，可以对20倍以内的变化很好地响应。

    Args:
        vol: [description]
        scale:

    Returns:
        [description]
    """
    features = {}
    cuts = [5, 10, 20]
    for cut in cuts:
        vols = resample(vol, cut, np.sum)
        if len(vol) // cut * cut != len(vol):
            continue
        if len(vols) < 2:
            continue

        v0, v1 = vols[-2:]
        print(cut, v0, v1)
        features[f"mom_vol_change_{cut}"] = np.tanh(v1 / v0 / scale)

    return features


def vol_acc(vol) -> Union[None, Dict]:
    """按[5,10,20]个样本为一组，计算量能加速的情况

    因子：
        mom_vol_acc_{cut}
    归一化：
        先对vol执行vol[1:]/vol[:-1],再求二阶导的tanh值
    Args:
        vol: [description]

    Returns:
        [description]
    """
    cuts = [5, 10, 20]
    features = {}

    for cut in cuts:
        if len(vol) // cut * cut != len(vol):
            continue

        vols = resample(vol, cut, np.sum)
        if len(vols) < 3:
            continue

        mom = momentum(vols[-3:], deg=2, with_norm=NormMethod.max_abs_scale)

        features[f"mom_vol_acc_{cut}"] = np.tanh(mom[0])

    return features
