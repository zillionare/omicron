import io
import time
from typing import List

import fire
import numpy as np
from numpy.typing import DTypeLike

from omicron.dal.influx.serialize import DataFrameDeserializer, NumpyDeserializer
from tests.dal.influx import mock_data_for_influx


def _serialize(
    arr: DTypeLike, format: List[str], sep=",", header: str = None, encoding="utf-8"
) -> str:
    stream = io.StringIO()

    header = header or ",".join(arr.dtype.names)

    np.savetxt(stream, arr, fmt=format, delimiter=sep, header=header, encoding=encoding)

    stream.seek(0)
    return stream.read()


def test_numpy_deserializer(lines, runs):
    # frame, open, close, code, name
    data = mock_data_for_influx(lines)
    assert data.shape[0] == lines

    txt = _serialize(data, format=["%d", "%.2f", "%.2f", "%s", "%s"], sep=",")

    des = NumpyDeserializer(dtype=data.dtype, sep=",", sort_values="frame")

    t0 = time.time()
    for i in range(runs):
        des(txt)

    print(f"numpy deserializer: {lines} lines: {(time.time() - t0)/runs} seconds")


def test_dataframe_deserializer(lines, runs):
    # frame, open, close, code, name
    data = mock_data_for_influx(lines)
    assert data.shape[0] == lines

    txt = _serialize(data, format=["%d", "%.2f", "%.2f", "%s", "%s"], sep=",")

    names = data.dtype.names
    des = DataFrameDeserializer(
        names=names, engine="c", parse_dates="frame", sort_values="frame"
    )

    t0 = time.time()
    for i in range(runs):
        des(txt)

    print(f"dataframe deserializer: {lines} lines: {(time.time() - t0)/runs} seconds")


if __name__ == "__main__":
    lines = 10_000
    runs = 10

    test_numpy_deserializer(lines, runs)
    test_dataframe_deserializer(lines, runs)
