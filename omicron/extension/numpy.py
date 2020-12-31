from typing import List, Tuple

import numpy as np


def dict_to_numpy_array(d: dict, dtype: List[Tuple]) -> np.array:
    return np.fromiter(d.items(), dtype=dtype, count=len(d))


def numpy_array_to_dict(arr: np.array, key: str, value: str) -> dict:
    return {item[key]: item[value] for item in arr}
