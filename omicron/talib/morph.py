from enum import IntEnum
from math import copysign
from typing import Sequence, Tuple

import numpy as np
from scipy.linalg import norm
from scipy.signal import savgol_filter

from omicron.talib.metrics import mean_absolute_error, pct_error


class CrossFlag(IntEnum):
    UPCROSS = 1
    DOWNCROSS = -1
    NONE = 0


def polyfit(ts: Sequence, deg: int = 2, loss_func="re") -> Tuple:
    """对给定的时间序列进行直线/二次曲线拟合。

    二次曲线可以拟合到反生反转的行情，如圆弧底、圆弧顶；也可以拟合到上述趋势中的单边走势，即其中一段曲线。对于如长期均线，在一段时间内走势可能呈现为一条直线，故也可用此函数进行直线拟合。

    为便于在不同品种、不同的时间之间对误差、系数进行比较，请事先对ts进行归一化。
    如果遇到无法拟合的情况（异常），将返回一个非常大的误差，并将其它项置为np.nan

    Examples:
        >>> ts = [i for i in range(5)]
        >>> err, (a, b) = polyfit(ts, deg=1)
        >>> print(round(err, 3), round(a, 1))
        0.0 1.0

    Args:
        ts (Sequence): 待拟合的时间序列
        deg (int): 如果要进行直线拟合，取1；二次曲线拟合取2. Defaults to 2
        loss_func (str): 误差计算方法，取值为`mae`, `rmse`,`mse` 或`re`。Defaults to `re` (relative_error)
    Returns:
        [Tuple]: 如果为直线拟合，返回误差，(a,b)(一次项系数和常数)。如果为二次曲线拟合，返回
        误差, (a,b,c)(二次项、一次项和常量）, (vert_x, vert_y)(顶点处的index，顶点值)
    """
    try:
        if any(np.isnan(ts)):
            raise ValueError("ts contains nan")

        x = np.array(list(range(len(ts))))

        z = np.polyfit(x, ts, deg=deg)

        p = np.poly1d(z)
        ts_hat = np.array([p(xi) for xi in x])

        if loss_func == "mse":
            error = np.mean(np.square(ts - ts_hat))
        elif loss_func == "rmse":
            error = np.sqrt(np.mean(np.square(ts - ts_hat)))
        elif loss_func == "mae":
            error = mean_absolute_error(ts, ts_hat)
        else:  # defaults to relative error
            error = pct_error(ts, ts_hat)

        if deg == 2:
            a, b, c = z[0], z[1], z[2]
            axis_x = -b / (2 * a)
            axis_y = (4 * a * c - b * b) / (4 * a)
            return error, z, (axis_x, axis_y)
        elif deg == 1:
            return error, z
    except Exception:
        error = 1e9
        return error, (np.nan, np.nan, np.nan), (np.nan, np.nan)


def angle(ts, threshold=0.01, loss_func="re") -> Tuple[float, float]:
    """求时间序列`ts`拟合直线相对于`x`轴的夹角的余弦值

    本函数可以用来判断时间序列的增长趋势。当`angle`处于[-1, 0]时，越靠近0，下降越快；当`angle`
    处于[0, 1]时，越接近0，上升越快。

    如果`ts`无法很好地拟合为直线，则返回[float, None]

    Examples:

        >>> ts = np.array([ i for i in range(5)])
        >>> round(angle(ts)[1], 3) # degree: 45, rad: pi/2
        0.707

        >>> ts = np.array([ np.sqrt(3) / 3 * i for i in range(10)])
        >>> round(angle(ts)[1],3) # degree: 30, rad: pi/6
        0.866

        >>> ts = np.array([ -np.sqrt(3) / 3 * i for i in range(7)])
        >>> round(angle(ts)[1], 3) # degree: 150, rad: 5*pi/6
        -0.866

    Args:
        ts:

    Returns:
        返回 (error, consine(theta))，即拟合误差和夹角余弦值。

    """
    err, (a, b) = polyfit(ts, deg=1, loss_func=loss_func)
    if err > threshold:
        return (err, None)

    v = np.array([1, a + b])
    vx = np.array([1, 0])

    return err, copysign(np.dot(v, vx) / (norm(v) * norm(vx)), a)


def cross(f: np.ndarray, g: np.ndarray) -> CrossFlag:
    """
    判断序列f是否与g相交。如果两个序列有且仅有一个交点，则返回1表明f上交g；-1表明f下交g
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
    """
    判断序列f是否与g存在类型v型的相交。即存在两个交点，第一个交点为向下相交，第二个交点为向上
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


def slope(ts: np.array, loss_func="re"):
    """求ts表示的直线（如果能拟合成直线的话）的斜率

    Args:
        ts (np.array): [description]
        loss_func (str, optional): [description]. Defaults to 're'.
    """
    err, (a, b) = polyfit(ts, deg=1, loss_func=loss_func)

    return err, a


# pragma: no cover
def smooth(ts: np.array, win: int, poly_order=1, mode="interp"):
    """平滑序列ts，使用窗口大小为win的平滑模型，默认使用线性模型

    提供本函数主要基于这样的考虑： omicron的使用者可能并不熟悉信号处理的概念，这里相当于提供了相关功能的一个入口。

    Args:
        ts (np.array): [description]
        win (int): [description]
        poly_order (int, optional): [description]. Defaults to 1.
    """
    return savgol_filter(ts, win, poly_order, mode=mode)
