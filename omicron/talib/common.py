from itertools import compress
from typing import Sequence

import numpy as np
import sklearn
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, minmax_scale


def barssince(condition: Sequence[bool], default=None) -> int:
    """
    Return the number of bars since `condition` sequence was last `True`,
    or if never, return `default`.

        >>> condition = [True, True, False]
        >>> barssince(condition)
        1
    """
    return next(compress(range(len(condition)), reversed(condition)), default)


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


def top_n_argpos(ts: np.array, n: int) -> np.array:
    """get top n (max->min) elements and return argpos which its value ordered in descent

    Example:
        >>> top_n_argpos([4, 3, 9, 8, 5, 2, 1, 0, 6, 7], 2)
        array([2, 3])

    Args:
        ts (np.array): [description]
        n (int): [description]

    Returns:
        np.array: [description]
    """
    return np.argsort(ts)[-n:][::-1]
