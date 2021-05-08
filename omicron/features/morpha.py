"""
K线形态特征提取
"""

import logging
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from omicron.features.maths import polyfit

logger = logging.getLogger(__name__)


def cross(f: np.array, g: np.array) -> Tuple[int, int]:
    """判断序列f是否与g相交。

    如果两个序列有且仅有一个交点，则返回1表明f上交g；-1表明f下交g。如果f与g有多个交点，返回
    最后一个。

    examples:
        >>> import matplotlib.pyplot as plt

        >>> x = np.arange(100)/10
        >>> y1 = np.array([math.sin(xi) for xi in x])
        >>> y2 = np.array([(0.5 * xi -0.5) for xi in x])
        >>> plt.plot(x, y1)
        >>> plt.plot(x, y2)

        >>> cross(y2, y1)
        >>> (1,23)

        >>> y4 = x/100 + 0.5
        >>> cross(y1, y4)
        >>> (-1, 87)

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


def vcross(f: np.array, g: np.array) -> Tuple[int, Tuple[int]]:
    """判断序列f是否与g存在类型v型的相交。

    即存在两个交点，第一个交点为向下相交，第二个交点为向上相交。一般反映为洗盘拉升的特征。

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> x = np.arange(100)/10
        >>> y1 = np.array([math.sin(xi) for xi in x])
        >>> y5 = -0.75 + x /100
        >>> plt.plot(x[30:60], y5[30:60])
        >>> plt.plot(x[30:60], y1[30:60])
        >>> vcross(y1,y5)
        >>> (True, (39, 55))

    Args:
        f:
        g:

    Returns:
        flag:
        indice:
    """
    indices = np.argwhere(np.diff(np.sign(f - g))).flatten()
    if len(indices) == 2:
        idx0, idx1 = indices
        if f[idx0] > g[idx0] and f[idx1] < g[idx1]:
            return True, (idx0, idx1)

    return False, (None, None)


def ncross(f: np.array, g: np.array) -> Tuple[int, Tuple[int]]:
    """判断序列f是否与g存在n型相交

    如果f与g存在两个交点，第一个为向上相交，第二个为向下相交，即为n型相交。一般反映为见顶形态。

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> import math
        >>> x = np.arange(100)/10
        >>> y1 = np.array([math.sin(xi) for xi in x])
        >>> y5 = 0.5 + x /100
        >>> plt.plot(x[60:95], y5[60:95])
        >>> plt.plot(x[60:95], y1[60:95])
        >>> ncross(y1,y5)
        >>> True (8,27)
    Args:
        f: [description]
        g: [description]

    Returns:
        [description]
    """
    indices = np.argwhere(np.diff(np.sign(g - f))).flatten()
    if len(indices) == 2:
        idx0, idx1 = indices
        if f[idx0] < g[idx0] and f[idx1] > g[idx1]:
            return True, (idx0, idx1)

    return False, (None, None)


def is_curve_up(a: float, vx: Union[float, int], win: int):
    """判断二次曲线是否向上

    在一个起始点为1.0（即已标准化）的时间序列中，如果经过polyfit以后，
        1）a > 0, b > 0， 向上开口抛物线，最低点在序列左侧（vx < 0)
        2) a > 0, b < 0, 向上开口抛物线，序列从vx之后开始向上。即如果vx>=win，则还要等待
            win - vx + 1个周期才能到最低点，然后序列开始向上
        3）a < 0, b > 0, 向下开口抛物线，序列从vx之后开始向下。即如果vx>win，则序列还将向上
            运行一段时间（vx-win+1个frame)后再向下
        4） a < 0, b < 0，向下开口抛物线，最高点在序列左侧(vx < 0)

    Example code:
    ```python
        # 观察a,b与曲线顶关系：
        def test_abc(a,b):
        p = np.poly1d((a,b,1))
        x = [i for i in range(10)]
        y = [p(i) for i in range(10)]

        plt.plot(x,y)

        err, (a,b,c),(vx,_) = polyfit(y)
        print(np.round([a,b,c,vx],4))
    ```

    由于a,b,c和vx相互决定，c==1.0，因此只需要a和vx两个变量就可以决定曲线未来走向。
    Args:
        a: 即二次曲线的系数a
        vx:
        win:

    Returns:

    """
    return (a > 0 and vx < win - 1) or (a < 0 and vx > win)


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
        if not (7 <= x <= win - 1):
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
                if value > mx:  # 相邻位置连续给出信号，合并之，否则忽略
                    peaks.pop(-1)
                    peaks.append(index)
                    mx = value
            else:
                peaks.append(index)

    return peaks, valleys
