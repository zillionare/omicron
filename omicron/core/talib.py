# common core functions for technical analysis


import logging
from itertools import compress
from math import copysign
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import sklearn
from numpy.lib.stride_tricks import sliding_window_view
from numpy.linalg import norm
from sklearn.preprocessing import MaxAbsScaler, Normalizer, StandardScaler, minmax_scale

from omicron.core.numpy_extensions import ffill_na
logger = logging.getLogger(__name__)


def barssince(condition: Sequence[bool], default=np.inf) -> int:
    """
    Return the number of bars since `condition` sequence was last `True`,
    or if never, return `default`.

        >>> condition = [True, True, False]
        >>> barssince(condition)
        1
    """
    return next(compress(range(len(condition)), reversed(condition)), default)


# pragma: no cover this was simply invoke bottleneck's functions
def rolling(x, win, func):
    results = []
    for subarray in sliding_window_view(x, window_shape=win):
        results.append(func(subarray))

    return np.array(results)


def moving_average(ts: np.array, win: int):
    """计算时间序列ts在win窗口内的移动平均

    Example:

        >>> ts = np.arange(7)
        >>> moving_average(ts, 5)
        array([2., 3., 4.])

    """

    return np.convolve(ts, np.ones(win) / win, "valid")


def mean_absolute_error(y: np.array, y_hat: np.array) -> float:
    """返回预测序列相对于真值序列的平均绝对值差

    Examples:

        >>> y = np.arange(5)
        >>> y_hat = np.arange(5)
        >>> y_hat[4] = 0
        >>> mean_absolute_error(y, y)
        0.0

        >>> mean_absolute_error(y, y_hat)
        0.8

    Args:
        y (np.array):
        y_hat:

    Returns:

    """
    return np.mean(np.abs(y - y_hat))


def relative_error(y: np.array, y_hat: np.array) -> float:
    """相对于序列算术均值的误差值

    Examples:
        >>> y = np.arange(5)
        >>> y_hat = np.arange(5)
        >>> y_hat[4] = 0
        >>> relative_error(y, y_hat)
        0.4

    Args:
        y (np.array): [description]
        y_hat (np.array): [description]

    Returns:
        float: [description]
    """
    mae = mean_absolute_error(y, y_hat)
    return mae / np.mean(np.abs(y))


def normalize(X, scaler="maxabs"):
    """对数据进行规范化处理。

    如果scaler为maxabs，则X的各元素被压缩到[-1,1]之间
    如果scaler为unit_vector，则将X的各元素压缩到单位范数
    如果scaler为minmax,则X的各元素被压缩到[0,1]之间
    如果scaler为standard,则X的各元素被压缩到单位方差之间，且均值为零。

    参考 [sklearn]

    [sklearn]: https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#results

    Examples:

        >>> X = [[ 1., -1.,  2.],
        ... [ 2.,  0.,  0.],
        ... [ 0.,  1., -1.]]

        >>> expected = [[ 0.4082, -0.4082,  0.8165],
        ... [ 1.,  0.,  0.],
        ... [ 0.,  0.7071, -0.7071]]

        >>> X_hat = normalize(X, scaler='unit_vector')
        >>> np.testing.assert_array_almost_equal(expected, X_hat, decimal=4)

        >>> expected = [[0.5, -1., 1.],
        ... [1., 0., 0.],
        ... [0., 1., -0.5]]

        >>> X_hat = normalize(X, scaler='maxabs')
        >>> np.testing.assert_array_almost_equal(expected, X_hat, decimal = 2)

        >>> expected = [[0.5       , 0.        , 1.        ],
        ... [1.        , 0.5       , 0.33333333],
        ... [0.        , 1.        , 0.        ]]
        >>> X_hat = normalize(X, scaler='minmax')
        >>> np.testing.assert_array_almost_equal(expected, X_hat, decimal= 3)

        >>> X = [[0, 0],
        ... [0, 0],
        ... [1, 1],
        ... [1, 1]]
        >>> expected = [[-1., -1.],
        ... [-1., -1.],
        ... [ 1., 1.],
        ... [ 1.,  1.]]
        >>> X_hat = normalize(X, scaler='standard')
        >>> np.testing.assert_array_almost_equal(expected, X_hat, decimal = 3)

    Args:
        X (2D array):
        scaler (str, optional): [description]. Defaults to 'maxabs_scale'.
    """
    if scaler == "maxabs":
        return MaxAbsScaler().fit_transform(X)
    elif scaler == "unit_vector":
        return sklearn.preprocessing.normalize(X, norm="l2")
    elif scaler == "minmax":
        return minmax_scale(X)
    elif scaler == "standard":
        return StandardScaler().fit_transform(X)


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
            error = mean_absolute_error(ts, ts_hat) / np.sqrt(np.mean(np.square(ts)))

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


def polyfit_inflextion(ts, win=10, err=0.001):
    """
    通过曲线拟合法来寻找时间序列的极值点（局部极大值、极小值）。

    ts为时间序列， win为用来寻找极值的时间序列窗口。
    erro为可接受的拟合误差。

    Returns:
        极值点在时间序列中的索引值
    """
    valleys = []
    peaks = []
    mn, mx = None, None
    for i in range(win, len(ts) - win):
        _err, coef, vert = polyfit(ts[i - win : i])
        if _err > err:
            continue
        a, b, c = coef
        x, y = vert
        if not (8 <= x <= win - 1):
            continue

        index = i - win + 1 + int(x)
        if a > 0:  # 找到最低点
            value = ts[index]
            if mn is None:
                mn = value
                valleys.append(index)
                continue

            if index - valleys[-1] <= 2:
                if value < mn:  # 相邻位置连续给出信号，合并之
                    valleys.pop(-1)
                    valleys.append(index)
                    mn = value
            else:
                valleys.append(index)
        else:  # 找到最高点
            value = ts[index]
            if mx is None:
                mx = value
                peaks.append(index)
                continue

            if index - peaks[-1] <= 2:
                if value > mx:  # 相邻位置连续给出信号，合并之
                    peaks.pop(-1)
                    peaks.append(index)
                    mx = value
                else:
                    peaks.append(index)

    return peaks, valleys


def cross(f, g):
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
        return 0, 0

    # 如果存在一个或者多个交点，取最后一个
    idx = indices[-1]

    if f[idx] < g[idx]:
        return 1, idx
    elif f[idx] > g[idx]:
        return -1, idx
    else:
        return np.sign(g[idx - 1] - f[idx - 1]), idx


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
        f:
        g:

    Returns:

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


def max_drawdown(ts: np.array):
    """求区间内的最大回撤

    如果ts中包含np.NaN，它们将被替换成前一个非np.NaN值（前向替换）。如果ts以np.NaN起头，则起头部分被替换成为序列中第一个非np.NaN值。

    如果返回0，意味着序列中不存在最大回撤。

    [See also](https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp)

    Examples:
        >>> ts = np.sin(np.arange(10) * np.pi / 10)
        >>> dd, start, end = max_drawdown(ts)
        >>> print(start, end)
        5 9

    Args:
        ts (np.array): [description]

    Returns:
        [type]: [description]
    """
    arr = ffill_na(ts)
    if np.isnan(arr[0]):
        i = np.argmin(~np.isnan(arr))
        if i + 1 == len(arr):  # arr is full of np.nan
            return 0, None, None

        arr[:i] = arr[i + 1]

    end = np.argmax(np.maximum.accumulate(arr) - arr)  # end of the period
    start = np.argmax(arr[:end])  # start of period

    return arr[end] / arr[start] - 1, start, end
