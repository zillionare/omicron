from typing import List, Tuple

import numpy as np


def dict_to_numpy_array(d: dict, dtype: List[Tuple]) -> np.array:
    return np.fromiter(d.items(), dtype=dtype, count=len(d))


def numpy_array_to_dict(arr: np.array, key: str, value: str) -> dict:
    return {item[key]: item[value] for item in arr}


def stride(a, L, S=1):  # Window len = L, Stride len/stepsize = S
    try:
        nrows = ((a.size - L) // S) + 1
        n = a.strides[0]
        return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n))
    except ValueError:
        return np.empty(L)


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
