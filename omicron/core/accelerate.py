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
    # numbas type inference.
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
