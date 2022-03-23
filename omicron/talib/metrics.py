import numpy as np
from bottleneck import nanmean


def mean_absolute_error(y: np.array, y_hat: np.array) -> float:
    """返回预测序列相对于真值序列的平均绝对值差

    两个序列应该具有相同的长度。如果存在nan，则nan的值不计入平均值。

    Examples:

        >>> y = np.arange(5)
        >>> y_hat = np.arange(5)
        >>> y_hat[4] = 0
        >>> mean_absolute_error(y, y)
        0.0

        >>> mean_absolute_error(y, y_hat)
        0.8

    Args:
        y (np.array): 真值序列
        y_hat: 比较序列

    Returns:
        float: 平均绝对值差
    """
    return nanmean(np.abs(y - y_hat))


def pct_error(y: np.array, y_hat: np.array) -> float:
    """相对于序列算术均值的误差值

    Examples:
        >>> y = np.arange(5)
        >>> y_hat = np.arange(5)
        >>> y_hat[4] = 0
        >>> pct_error(y, y_hat)
        0.4

    Args:
        y (np.array): [description]
        y_hat (np.array): [description]

    Returns:
        float: [description]
    """
    mae = mean_absolute_error(y, y_hat)
    return mae / nanmean(np.abs(y))
