"""performance benchmark for influx reltaed functions

Returns:
    [description]
"""
import io
import time
from typing import List

import numpy as np
import pandas as pd
from influxdb_client.client.write import (
    dataframe_serializer as influx_dataframe_serializer,
)
from numpy.typing import DTypeLike

from omicron.dal.influx import serialize as s
from omicron.dal.influx.serialize import (
    DataframeDeserializer,
    NumpyDeserializer,
    NumpySerializer,
)
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
    des = DataframeDeserializer(
        names=names, engine="c", parse_dates="frame", sort_values="frame"
    )

    t0 = time.time()
    for i in range(runs):
        des(txt)

    print(f"dataframe deserializer: {lines} lines: {(time.time() - t0)/runs} seconds")


def test_line_protocol_escape(lines, runs):
    from omicron.dal.influx.escape import _escape_comma_equal_space

    data = []
    for i in range(lines):
        data.append("a,b,c=d,little fox jump over the fence")

    print(_escape_comma_equal_space([data[0]]))
    t0 = time.time()
    for i in range(runs):
        _escape_comma_equal_space(np.array(data))
    print(f"escape: {lines} lines: {(time.time() - t0)/runs} seconds")


def test_mydataframe_serializer(lines, runs):
    data = mock_data_for_influx(lines)

    df = pd.DataFrame(
        data, columns=["open", "close", "code", "name"], index=data["frame"]
    )

    t0 = time.time()
    for i in range(runs):
        serialize = s.DataframeSerializer("test", ["open", "close"], ["code", "name"])

        for lp in serialize(df, lines):
            pass

    print(f"my dataframe serializer: {lines} lines: {(time.time() - t0)/runs} seconds")


def test_dataframe_serializer(lines, runs):
    data = mock_data_for_influx(lines)

    df = pd.DataFrame(
        data, columns=["open", "close", "code", "name"], index=data["frame"]
    )

    t0 = time.time()
    for i in range(runs):  # noqa
        ser = influx_dataframe_serializer.DataframeSerializer(
            df,
            PointSettings(),
            chunk_size=lines,
            data_frame_measurement_name="test",
            data_frame_tag_columns=["code", "name"],
        )
        lp = "\n".join(ser.serialize(0))  # noqa

    print(
        f"influxdb dataframe serializer: {lines} lines: {(time.time() - t0)/runs} seconds"
    )


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
            ["open", "close"],
            precisions={"open": 1, "close": 2},
        )
        for lp in serialize(lines):
            pass

    print(
        f"influxdb numpy serializer: {lines} lines: {(time.time() - t0)/runs} seconds"
    )


if __name__ == "__main__":
    lines = 1_000_000
    runs = 5

    from influxdb_client.client.write.dataframe_serializer import DataframeSerializer
    from influxdb_client.client.write_api import PointSettings

    # test_line_protocol_escape(lines, runs)
    # test_numpy_deserializer(lines, runs)
    # test_dataframe_deserializer(lines, runs)
    test_mydataframe_serializer(lines, runs)
    test_dataframe_serializer(lines, runs)
    test_numpy_serializer(lines, runs)
