#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors:

"""
import logging
import warnings

import numpy as np
from numba import njit, NumbaPendingDeprecationWarning

warnings.filterwarnings("ignore", category=NumbaPendingDeprecationWarning)

logger = logging.getLogger(__name__)


@njit
def index(arr, item):
    for idx, val in np.ndenumerate(arr):
        if val == item:
            return idx
    # If no item was found return None, other return types might be a problem due to
    # numba's type inference.
    return -1


@njit
def index_sorted(arr, item):
    pos = np.searchsorted(arr, item)
    if arr[pos] == item:
        return pos
    else:
        return -1

@njit
def count_between(arr, start, end):
    """
    arr is sorted.
    """
    pos_start = np.searchsorted(arr, start, side='right')
    pos_end = np.searchsorted(arr, end, side='right')

    return pos_end - pos_start + 1

@njit
def shift(arr, start, offset):
    """
    在numpy数组arr中，找到start(或者最接近的一个），取offset对应的元素
    """
    pos = np.searchsorted(arr, start, side='right')

    if pos + offset - 1 >= len(arr):
        return start
    else:
        return arr[pos + offset - 1]

@njit
def minute_frames_floor(ticks, moment):
    """
    对于分钟级的frame,返回它们与frame刻度向下对齐后的frame及日期进位。如果需要对齐到上一个交易
    日，则进位为-1，否则为0.
    Examples:
        > ticks = [600, 630, 660, 690, 810, 840, 870, 900]
        > floor(ticks, 545) -> 900, -1
        > floor(ticks, 600) -> 600, 0
        > floor(ticks, 605) -> 600, 0
        > floor(ticks, 899) -> 870, 0
        > floor(ticks, 900) -> 900, 0
        > floor(ticks, 905) -> 900, 0
    Args:
        ticks (np.array or list): frames刻度
        moment (int): 整数表示的分钟数，比如900表示15：00

    Returns:
        tuple, the first is the new moment, the second is carry-on
    """
    if moment < ticks[0]:
        return ticks[-1], -1
    # ’right' 相当于 ticks <= m
    index = np.searchsorted(ticks, moment, side='right')
    return ticks[index - 1], 0

@njit
def floor(arr, item):
    """
    在数据arr中，找到小于等于item的那一个值。如果item小于所有arr元素的值，返回arr[0];如果item
    大于所有arr元素的值，返回arr[-1]

    与`minute_frames_floor`不同的是，本函数不做回绕与进位.

    Examples:
        > a = [3, 6, 9]
        > floor(a, -1)
        0
        > floor(a, 9)
        9
        > floor(a, 1)
        0
        > floor(a, 4)
        3
        > floor(a,6)
        6
    Args:
        arr:
        item:

    Returns:

    """
    if item < arr[0]:
        return arr[0]
    index = np.searchsorted(arr, item, side='right')
    return arr[index - 1]

