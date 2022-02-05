"""performance benchmark for influx reltaed functions

Returns:
    [description]
"""
import io
import time
from typing import List

import numpy as np
import pandas as pd
from numpy import ndarray

from omicron.dal.influx.serialize import (
    DataframeDeserializer,
    DataframeSerializer,
    NumpyDeserializer,
    NumpySerializer,
)
from tests.dal.influx import mock_data_for_influx


def _serialize(
    arr: ndarray, format: List[str], sep=",", header: str = None, encoding="utf-8"
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
    des = DataframeDeserializer(
        names=names, engine="c", parse_dates="frame", sort_values="frame"
    )

    t0 = time.time()
    for i in range(runs):
        des(txt)

    print(f"dataframe deserializer: {lines} lines: {(time.time() - t0)/runs} seconds")


def test_line_protocol_escape(runs):
    from omicron.dal.influx.escape import (
        KEY_ESCAPE,
        MEASUREMENT_ESCAPE,
        STR_ESCAPE,
        TAG_ESCAPE,
        escape,
    )

    line = "a,b,c=d,little fox jump over the fence"
    print(escape(line, TAG_ESCAPE))
    t0 = time.time()
    for i in range(runs):
        pattern = [TAG_ESCAPE, TAG_ESCAPE, MEASUREMENT_ESCAPE, KEY_ESCAPE, STR_ESCAPE][
            i % 5
        ]
        escape(line, pattern)
    print(f"escape {runs} lines: {(time.time() - t0)/runs} seconds")


def test_dataframe_serializer(lines, runs):
    data = mock_data_for_influx(lines)

    df = pd.DataFrame(
        data, columns=["open", "close", "code", "name"], index=data["frame"]
    )

    t0 = time.time()
    for i in range(runs):  # noqa
        df_serializer = DataframeSerializer(df, "test", tag_keys=["name", "code"])
        for lp in df_serializer.serialize(lines):
            pass

    print(f"dataframe serializer: {lines} lines: {(time.time() - t0)/runs} seconds")


def test_numpy_serializer(lines, runs):
    data = mock_data_for_influx(lines)

    arr = np.array(
        data,
        dtype=[
            ("frame", "M8[ns]"),
            ("open", "f8"),
            ("close", "f8"),
            ("code", "U10"),
            ("name", "U10"),
        ],
    )

    t0 = time.time()
    for i in range(runs):
        serialize = NumpySerializer(
            arr,
            "test",
            "frame",
            ["name", "code"],
            precisions={"open": 1, "close": 2},
        )
        for lp in serialize.serialize(lines):
            pass

    print(f"numpy serializer: {lines} lines: {(time.time() - t0)/runs} seconds")


if __name__ == "__main__":
    lines = 10_000
    runs = 5

    # 2.75e-6 seconds per line
    test_line_protocol_escape(lines)
    test_numpy_deserializer(lines, runs)
    test_numpy_serializer(lines, runs)

    test_dataframe_serializer(lines, runs)
    test_dataframe_deserializer(lines, runs)
