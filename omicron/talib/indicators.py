from typing import Sequence

import numpy as np
import pandas as pd
from bottleneck import move_mean


def moving_average(ts: Sequence, win: int, padding=True) -> np.ndarray:
    """生成ts序列的移动平均值

    Examples:

        >>> ts = np.arange(7)
        >>> moving_average(ts, 5)
        array([nan, nan, nan, nan,  2.,  3.,  4.])

    Args:
        ts (Sequence): the input array
        win (int): the window size
        padding: if True, then the return will be equal length as input, padding with np.NaN at the beginning

    Returns:
        The moving mean of the input array along the specified axis. The output has the same shape as the input.
    """
    ma = move_mean(ts, win)
    if padding:
        return ma
    else:
        return ma[len(ts) - win :]


def weighted_moving_average(ts: np.array, win: int) -> np.array:
    """计算加权移动平均

    Args:
        ts (np.array): [description]
        win (int): [description]

    Returns:
        np.array: [description]
    """
    w = [2 * (i + 1) / (win * (win + 1)) for i in range(win)]

    return np.convolve(ts, w, "valid")


def exp_moving_average(values, window):
    """Numpy implementation of EMA"""
    weights = np.exp(np.linspace(-1.0, 0.0, window))
    weights /= weights.sum()
    a = np.convolve(values, weights, mode="full")[: len(values)]
    a[:window] = a[window]
    return a
