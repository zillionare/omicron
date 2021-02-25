#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import warnings

import numpy as np

logger = logging.getLogger(__name__)


def index(arr, item):
    for idx, val in np.ndenumerate(arr):
        if val == item:
            return idx
    # If no item was found return None, other return types might be a problem due to
    # numba's type inference.
    return -1


def index_sorted(arr, item):
    pos = np.searchsorted(arr, item)
    if arr[pos] == item:
        return pos
    else:
        return -1


def count_between(arr, start, end):
    """
    arr is sorted.
    """
    pos_start = np.searchsorted(arr, start, side="right")
    pos_end = np.searchsorted(arr, end, side="right")

    return pos_end - pos_start + 1


def shift(arr, start, offset):
    """
    在numpy数组arr中，找到start(或者最接近的一个），取offset对应的元素
    """
    pos = np.searchsorted(arr, start, side="right")

    if pos + offset - 1 >= len(arr):
        return start
    else:
        return arr[pos + offset - 1]


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
    index = np.searchsorted(ticks, moment, side="right")
    return ticks[index - 1], 0


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
    index = np.searchsorted(arr, item, side="right")
    return arr[index - 1]


def join_by_left(key, r1, r2, mask=False):
    # figure out the dtype of the result array
    descr1 = r1.dtype.descr
    descr2 = [d for d in r2.dtype.descr if d[0] not in r1.dtype.names]
    descrm = descr1 + descr2

    # figure out the fields we'll need from each array
    f1 = [d[0] for d in descr1]
    f2 = [d[0] for d in descr2]

    # cache the number of columns in f1
    ncol1 = len(f1)

    # get a dict of the rows of r2 grouped by key
    rows2 = {}
    for row2 in r2:
        rows2.setdefault(row2[key], []).append(row2)

    # figure out how many rows will be in the result
    nrowm = 0
    for k1 in r1[key]:
        if k1 in rows2:
            nrowm += len(rows2[k1])
        else:
            nrowm += 1

    # allocate the return array
    _ret = np.recarray(nrowm, dtype=descrm)
    if mask:
        ret = np.ma.array(_ret, mask=True)
    else:
        ret = _ret

    # merge the data into the return array
    i = 0
    for row1 in r1:
        if row1[key] in rows2:
            for row2 in rows2[row1[key]]:
                ret[i] = tuple(row1[f1]) + tuple(row2[f2])
                i += 1
        else:
            for j in range(ncol1):
                ret[i][j] = row1[j]
            i += 1

    return ret


# todo: check if numpy 1.19 has fixed the bug
def numpy_append_fields(base, names, data, dtypes):
    """numpy.lib.recfunctions.rec_append_fields 不能处理new_arr的类型为Object的情况

    Args:
        base ([type]): [description]
        name ([type]): [description]
        new_arr ([type]): [description]
        dtypes ([type]): [description]
    """
    # check the names
    if isinstance(names, (tuple, list)):
        if len(names) != len(data):
            msg = "The number of arrays does not match the number of names"
            raise ValueError(msg)
    elif isinstance(names, str):
        names = [
            names,
        ]
        data = [
            data,
        ]

    result = np.empty(base.shape, dtype=base.dtype.descr + dtypes)
    for col in base.dtype.names:
        result[col] = base[col]

    for i in range(len(names)):
        result[names[i]] = data[i]

    return result
