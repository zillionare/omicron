import logging
import math
from enum import IntEnum
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)


class NormMethod(IntEnum):
    start_scale = 1
    end_scale = 2
    minmax_scale = 3
    zscore = 4
    l1 = 5
    l2 = 6
    max_abs_scale = 7


def norm(ts: np.array, method=NormMethod.start_scale):
    """对时间序列进行归一化。

    为了便于在不同的证券之间进行特征比较，往往要先对其进行归一化操作。本函数在常见的l1, l2和
    min_max归一化之外，又增加了start_scale,end_scale等方法，以适用证券分析的需要。

    l1, l2归一化借用了``sklearn.preprocessing.normalize``方法,``zscore``借用了
    ``sklearn.preprocessing.scale``方法, ``minmax_scale``借用了
    ``sklearn.preprocessing.minmax_scale``方法。

    ``start_scale``和``end_scale``分别意味着将序列除以起始元素或者结束元素。对证券数据的预处理
    而言，这种方法尽可能地保留了时间序列的分析特征。

    ``max_abs_scale``类似于``minmax_scale``，但不会对序列的空间进行压缩。

    !!! note
        使用``start_scale``, ``end_scale``和``max_abs_scale``方法都可能导致除零错误

    Args:
        ts :
        method : one of [omicron.features.maths.NormMethod][]
        Defaults to [omicron.features.maths.NormMethod.start_scale][]
    """
    assert len(ts.shape) == 1

    if method == NormMethod.start_scale:
        return ts / ts[0]
    elif method == NormMethod.end_scale:
        return ts / ts[-1]
    elif method == NormMethod.minmax_scale:
        return preprocessing.minmax_scale(ts)
    elif method == NormMethod.zscore:
        return preprocessing.scale(ts)
    elif method in [NormMethod.l1, NormMethod.l2]:
        return preprocessing.normalize(ts.reshape((1, -1)), norm=method.name)[0]
    elif method == NormMethod.max_abs_scale:
        return ts / max(abs(ts))


def rmse(y_actual: np.array, y_predict: np.array):
    """
    返回预测序列相对于真值序列的标准差。

    Args:
        y_actual:
        y_predict:

    Returns:

    """
    return mean_squared_error(y_actual, y_predict, squared=False)


def slope(ts: np.array, with_norm=NormMethod.start_scale):
    """将时间序列ts拟合为直线，返回其斜率和拟合误差

    当一支股票的均线在某区间里呈现单调上升或者下降趋势时，如果均线能拟合成一条直线，
    此均线可以看成支撑线或者压力线。均线的斜率越大，支撑越强；如果斜率为负，则可看作压力线，其
    值越小，压力越强。

    在计算斜率时，使用时间序列的索引值作为x轴。为便于不同证券之间的比较，在计算之前对ts进行了归
    一化（使用各分量除以二阶范数）。如果不进行这样的归一化，则会出现这样的情况：

    比如某个区间内大盘指数为半年线为：

    ``` python
    array([3271.0570, 3275.3785, 3279.5816, 3283.6522, 3287.5926, 3291.6699,
       3295.4384, 3299.4655, 3302.9275, 3306.5379, 3310.1039])

    ```

    而某只股票的5日线为:

    ```Python
    [6.1051 6.7156 7.3872 8.1259 8.9385 9.8323]
    ```

    分别对两个时间序列求``slope``,在未归一化的情况下，前者得到的结果为3.9，后者得到的结果为
    0.744，大盘半年线的斜率更大。这与实际情况刚好相反。实际上后者是模拟的一支初始价为5元，每天以
    10%幅度上涨的股票的5日线，显然直观上看，后者应该有更大的倾角；而大盘指数的半年线实际上是比较
    平的。

    对时间序列进行归一化之后，两条拟合直线的斜率分别为0.00119和0.1228，分别对应的倾角为
    0.068度和6.94度，两者的次序现在正确了。

    !!! note
        您可能感觉到每天以10%速度上涨的股票，其5日均线的倾角应该是60度，而不是这里的6.94度。这
        取决于您的行情软件如何绘制图形。行情软件绘制图形的方法会带来各种视觉误差。这也是为什么
        需要量化的原因。[omicron.features.maths.slope2degree][] 给出了如何通过这里求得的slope值来
        获取视觉友好的倾角值的建议。

    Args:
        ts:

    Returns:
        返回拟合直线的斜率和拟合误差。当拟合误差过大时，拟合直线的斜率没有意义。

    """
    if with_norm is not None:
        ts = norm(ts, with_norm)

    err, (a, b) = polyfit(ts, deg=1)
    return a, err


def slope2degree(s: float):
    """给定直线（斜率为s），求其与x轴的夹角theta

    在交易因子设计中，一般使用slope就够了。将其转换为角度表示只是为了便于理解，是一种视觉上的
    主观信号。注意在 [omicron.features.maths.slope][] 中求斜率时，由于缺省对应的x坐标，我们引入了时间
    序列本身的索引作为坐标。这样求出来的斜率，常常与行情软件绘制的图形不相一致，但用于量化因子计
    算则是合适的、客观的。

    如果我们将[omicron.features.maths.slope][]计算出来的斜率传入本函数进行角度转换，其倾角一般都较
    小，因而显得不够直观。

    这里建议如果s是经由[omicron.features.maths.slope][]求得，在进行角度转换时，建议可将s*10再进行转
    换。这样，对一支以10%涨速上涨的股票，其5日均线的倾角将在50度左右，见下例。这与多数人的视觉
    经验相一致。


    Examples:
        >>> c = np.array([1.1**x for x in range(10)]) * 5
        >>> ma = moving_average(c, 5)
        >>> slope2degree(slope(ma)[0] * 10)
        >>> 50.63

    """
    return math.degrees(math.atan(s))


def moving_average(ts: np.array, win: int):
    """计算时间序列ts在win窗口内的移动平均

    Example:

        >>> ts = np.arange(7)
        >>> moving_average(ts, 5)
        >>> array([2.0000, 3.0000, 4.0000])

    """

    return np.convolve(ts, np.ones(win) / win, "valid")


def momentum(ts: Sequence, deg=1, with_norm=NormMethod.start_scale):
    """时间序列的一阶动量或者二阶动量

    Example:
        > momentum(np.arange(3))
        > array([1.0000, 1.0000])

        >
    """
    if with_norm is not None:
        ts = norm(ts, with_norm)

    dts = np.diff(ts)
    if deg == 2:
        return np.diff(dts)
    else:
        return dts


def find_runs(x: Sequence) -> Tuple[np.array]:
    """查找序列中连续个相同项

    Examples:
        >>> arr = [1, 2, 2, 2, 3, 1, 1, 1]
        >>> find_runs(arr)
        >>> (array([1, 2, 3, 1]), array([0, 1, 4, 5]), array([1, 3, 1, 3]))
    returns:
    """

    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError("only 1D array supported")
    n = x.shape[0]

    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # find run values
        run_values = x[loc_run_start]

        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        return run_values, run_starts, run_lengths


def polyfit(ts: Sequence, deg: int = 2, decimals: Optional[tuple] = None):
    """对给定的时间序列进行一阶或二阶拟合。

    二次曲线可以拟合到发生反转的行情，如圆弧底、圆弧顶；也可以拟合到上述趋势中的单边走势，即其中
    一段曲线。一次曲线的斜率可用来判断股价延伸走势。

    为了使得给出的error更加直观，这里的error计算方法是：取真值与预测值各项差的绝对值平均，再除于
    ts的均值。

    args:
        ts: 时间序列
        deg: 取值1或者2，分别对应一次曲线或者二次曲线拟合
        decimals: tuple of three, decimals for err, coef and vertex

    returns:
        error, coef, vertex(axis_x, axis_y)（仅当拟合二次曲线时存在）
    """
    x = np.arange(len(ts))

    try:
        z = np.polyfit(x, ts, deg=deg)

        # polyfit给出的残差是各项残差的平方和，这里返回相对于单项的误差比。对股票行情而言，最大
        # 可接受的std_err也许是小于1%
        p = np.poly1d(z)
        ts_hat = np.array([p(xi) for xi in x])
        error = np.mean(np.abs(ts - ts_hat)) / np.mean(np.abs(ts))

        if deg == 2:
            a, b, c = z[0], z[1], z[2]
            # 防止除零错误
            a = 1e-10 if a == 0 else a
            axis_x = -b / (2 * a)
            axis_y = (4 * a * c - b * b) / (4 * a)

            if decimals:
                return (
                    round(error, decimals[0]),
                    np.round(z, decimals[1]),
                    np.round((axis_x, axis_y), decimals[2]),
                )
            else:
                return error, z, (axis_x, axis_y)
        elif deg == 1:
            if decimals:
                return round(error, decimals[0]), np.round(z, decimals[1])
            else:
                return error, z
    except Exception as e:
        error = 1e9
        logger.warning("ts %s caused calculation error.")
        logger.exception(e)
        return error, (np.nan, np.nan, np.nan), (np.nan, np.nan)


def exp_fit(ts):
    """对时间序列进行指数拟合

    比如，某只股票近期每天以涨停收盘：
    ```Python
    c = np.array([1.01 ** x for x in range(5)]) * 5
    print(c)
    ```
    > [5.0000 5.0500 5.1005 5.1515 5.2030 5.2551 5.3076 5.3607 5.4143 5.4684]

    对该序列进行拟合，得到：
    > (1.4199675707078513e-16, (0.009950330853168127, 1.6094379124341003))

    这里的误差表明时间序列能够完全拟合为指数曲线。我们得到的是其对数项序列的一次多项项拟合参数。
    要还原原时间序列：
    ```
    print([exp(b) * exp(a * x) for x in range(5)])
    ```
    > [4.999999999999999, 5.049999999999999, 5.100499999999999, 5.151504999999999, 5.20302005]

    """
    try:
        x = list(range(len(ts)))
        y = np.log(ts)
        # https://stackoverflow.com/a/3433503/13395693 设置权重可以对small values更友好。
        z = np.polyfit(x, y, deg=1, w=np.sqrt(np.abs(y)))
        a, b = z[0], z[1]

        ts_hat = np.array([np.exp(a * x) * np.exp(b) for x in range(len(ts))])
        error = np.mean(np.abs(ts - ts_hat)) / np.mean(np.abs(ts))

        return error, (a, b)
    except Exception as e:
        error = 1e9
        logger.warning("ts %s caused calculation error", ts)
        logger.exception(e)
        return error, (None, None)


def resample(arr: np.array, cut: int, func: Callable) -> np.array:
    """将一维数组``arr``按长度``cut``进行分割，再应按分割后的组应用func进行计算，返回结果。

    Examples:
        >>> arr = np.arange(9)
        >>> resample(arrarr, 3, np.sum)
        >>> [3 12 21]
    Args:
        arr: [description]
        cut: [description]
        func: [description]

    Returns:
        [description]
    """
    rows = len(arr) // cut
    arr_ = arr.reshape((rows, -1))
    return np.apply_along_axis(func, 1, arr_)


def sigmoid(x: float) -> float:
    """计算x的sigmoid值

    Examples:

        >>> sigmoid(0.5)
        >>> 0.6224593312018546
        >>> sigmoid(-0.5)
        >>> 0.3775406687981454

    Args:
        x: [description]

    Returns:
        [description]
    """
    return 1 / (1 + np.exp(-x))
