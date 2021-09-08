"""Extension function related to numpy
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas
from numpy.typing import ArrayLike


def dict_to_numpy_array(d: dict, dtype: List[Tuple]) -> np.array:
    """convert dictionary to numpy array

    Examples:

    >>> d = {"aaron": 5, "jack": 6}
    >>> dtype = [("name", "S8"), ("score", "<i4")]
    >>> dict_to_numpy_array(d, dtype)
    array([(b'aaron', 5), (b'jack', 6)],
          dtype=[('name', 'S8'), ('score', '<i4')])

    Args:
        d (dict): [description]
        dtype (List[Tuple]): [description]

    Returns:
        np.array: [description]
    """
    return np.fromiter(d.items(), dtype=dtype, count=len(d))


def dataframe_to_structured_array(
    df: pandas.DataFrame, dtypes: List[Tuple] = None
) -> ArrayLike:
    """convert dataframe (with all columns, and index possibly) to numpy structured arrays

    `len(dtypes)` should be either equal to `len(df.columns)` or `len(df.columns) + 1`. In the later case, it implies to include `df.index` into converted array.

    Args:
        df: the one needs to be converted
        dtypes: Defaults to None. If it's `None`, then dtypes of `df` is used, in such case, the `index` of `df` will not be converted.

    Returns:
        ArrayLike: [description]
    """
    v = df
    if dtypes is not None:
        dtypes_in_dict = {key: value for key, value in dtypes}

        col_len = len(df.columns)
        if len(dtypes) == col_len + 1:
            v = df.reset_index()

            rename_index_to = set(dtypes_in_dict.keys()).difference(set(df.columns))
            v.rename(columns={"index": list(rename_index_to)[0]}, inplace=True)
        elif col_len != len(dtypes):
            raise ValueError(
                f"length of dtypes should be either {col_len} or {col_len + 1}, is {len(dtypes)}"
            )

        # re-arrange order of dtypes, in order to align with df.columns
        dtypes = []
        for name in v.columns:
            dtypes.append((name, dtypes_in_dict[name]))
    else:
        dtypes = df.dtypes

    return np.array(np.rec.fromrecords(v.values), dtype=dtypes)


def numpy_array_to_dict(arr: np.array, key: str, value: str) -> dict:
    return {item[key]: item[value] for item in arr}


def find_runs(x):
    """Find runs of consecutive items in an array."""

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


def count_between(arr, start, end):
    """计算数组中，`start`元素与`end`元素之间共有多少个元素

    要求arr必须是已排序。计算结果会包含区间边界点。

    Examples:
        >>> arr = [20050104, 20050105, 20050106, 20050107, 20050110, 20050111]
        >>> count_between(arr, 20050104, 20050111)
        6

        >>> count_between(arr, 20050104, 20050109)
        4
    """
    pos_start = np.searchsorted(arr, start, side="right")
    pos_end = np.searchsorted(arr, end, side="right")

    counter = pos_end - pos_start + 1
    if start < arr[0]:
        counter -= 1
    if end > arr[-1]:
        counter -= 1

    return counter


def shift(arr, start, offset):
    """在numpy数组arr中，找到start(或者最接近的一个），取offset对应的元素。

    要求`arr`已排序。`offset`为正，表明向后移位；`offset`为负，表明向前移位

    Examples:
        >>> arr = [20050104, 20050105, 20050106, 20050107, 20050110, 20050111]
        >>> shift(arr, 20050104, 1)
        20050105

        >>> shift(arr, 20050105, -1)
        20050104

        >>> # 起始点已右越界，且向右shift，返回起始点
        >>> shift(arr, 20050120, 1)
        20050120


    Args:
        arr : 已排序的数组
        start : numpy可接受的数据类型
        offset (int): [description]

    Returns:
        移位后得到的元素值
    """
    pos = np.searchsorted(arr, start, side="right")

    if pos + offset - 1 >= len(arr):
        return start
    else:
        return arr[pos + offset - 1]


def floor(arr, item):
    """
    在数据arr中，找到小于等于item的那一个值。如果item小于所有arr元素的值，返回arr[0];如果item
    大于所有arr元素的值，返回arr[-1]

    与`minute_frames_floor`不同的是，本函数不做回绕与进位.

    Examples:
        >>> a = [3, 6, 9]
        >>> floor(a, -1)
        3
        >>> floor(a, 9)
        9
        >>> floor(a, 10)
        9
        >>> floor(a, 4)
        3
        >>> floor(a,10)
        9

    Args:
        arr:
        item:

    Returns:

    """
    if item < arr[0]:
        return arr[0]
    index = np.searchsorted(arr, item, side="right")
    return arr[index - 1]


def join_by_left(key, r1, r2, mask=True):
    """左连接 `r1`, `r2` by `key`

    如果`r1`中存在`r2`中没有的行，则该行对应的`r2`中的那些字段的取值将使用`fill`来填充。如果
    same as numpy.lib.recfunctions.join_by(key, r1, r2, jointype='leftouter'), but allows
    r1 have duplicat keys

    [Reference: stackoverflow](https://stackoverflow.com/a/53261882/13395693)

    Examples:
        >>> # to join the following
        >>> # [[ 1, 2],
        >>> #  [ 1, 3],   x   [[1, 5],
        >>> #  [ 2, 3]]        [4, 7]]
        >>> # only first two rows in left will be joined

        >>> r1 = np.array([(1, 2), (1,3), (2,3)], dtype=[('seq', 'i4'), ('score', 'i4')])
        >>> r2 = np.array([(1, 5), (4,7)], dtype=[('seq', 'i4'), ('age', 'i4')])
        >>> joined = join_by_left('seq', r1, r2)
        >>> print(joined)
        [(1, 2, 5) (1, 3, 5) (2, 3, --)]

        >>> print(joined.dtype)
        (numpy.record, [('seq', '<i4'), ('score', '<i4'), ('age', '<i4')])

        >>> joined[2][2]
        masked

        >>> joined.tolist()[2][2] == None
        True

    Args:
        key : join关键字
        r1 : 数据集1
        r2 : 数据集2
        fill : 对匹配不上的cell进行填充时使用的值

    Returns:
        a numpy array
    """
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
    # ret = np.full((nrowm, ), fill, dtype=descrm)
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


def numpy_append_fields(base, names, data, dtypes):
    """给现有的数组`base`增加新的字段

    实现了`numpy.lib.recfunctions.rec_append_fields`的功能。因为`rec_append_fields`不能处
    理`data`元素的类型为Object的情况

    Example:
        >>> # 新增单个字段
        >>> import numpy
        >>> old = np.array([i for i in range(3)], dtype=[('col1', '<f4')])
        >>> new_list = [2 * i for i in range(3)]
        >>> res = numpy_append_fields(old, 'new_col', new_list, [('new_col', '<f4')])
        >>> print(res)
        ... # doctest: +NORMALIZE_WHITESPACE
        [(0., 0.) (1., 2.) (2., 4.)]

        >>> # 新增多个字段
        >>> data = [res['col1'].tolist(), res['new_col'].tolist()]
        >>> print(numpy_append_fields(old, ('col3', 'col4'), data, [('col3', '<f4'), ('col4', '<f4')]))
        ... # doctest: +NORMALIZE_WHITESPACE
        [(0., 0., 0.) (1., 1., 2.) (2., 2., 4.)]

    Args:
        base ([numpy.array]): 基础数组
        name ([type]): 新增字段的名字，可以是字符串（单字段的情况），也可以是字符串列表
        data (list): 增加的字段的数据，list类型
        dtypes ([type]): 新增字段的dtype
    """
    if isinstance(names, str):
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


def ffill_na(s: np.array) -> np.array:
    """前向替换一维数组中的np.NaN

    如果s以np.NaN起头，则起头处的np.NaN将无法被替换。

    Examples:

        >>> arr = np.arange(6, dtype=np.float32)
        >>> arr[3:5] = np.NaN
        >>> ffill_na(arr)
        ... # doctest: +NORMALIZE_WHITESPACE
        array([0., 1., 2., 2., 2., 5.], dtype=float32)

        >>> arr[0:2] = np.nan
        >>> ffill_na(arr)
        ... # doctest: +NORMALIZE_WHITESPACE
        array([nan, nan, 2., 2., 2., 5.], dtype=float32)

    Args:
        s (np.array): [description]

    Returns:
        np.array: [description]
    """
    mask = np.isnan(s)
    idx = np.where(~mask, np.arange(len(mask)), 0)
    np.maximum.accumulate(idx, out=idx)
    return s[idx]
