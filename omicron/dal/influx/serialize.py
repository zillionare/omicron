import io
import itertools
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from unittest.mock import DEFAULT

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, DTypeLike
from pandas import DataFrame


class Serializer(object):
    """base class of all serializer/deserializer"""

    pass


class DataframeDeserializer(Serializer):
    def __init__(
        self,
        sort_values: Union[str, List[str]] = None,
        encoding: str = None,
        sep: str = ",",
        header: Union[int, List[int], str] = "infer",
        names: List[str] = None,
        usecols: Union[List[int], List[str]] = None,
        dtype: dict = None,
        engine: str = None,
        parse_dates=None,
        infer_datetime_format=True,
        lineterminator: str = None,
        converters: dict = None,
        date_parser=None,
        skipfooter=0,
        index_col: Union[int, str, List[int], List[str], bool] = None,
        skiprows: Union[int, List[int], Callable] = None,
        **kwargs,
    ):
        """constructor a deserializer which convert a csv-like bytes array to pandas.DataFrame

        the args are the same as pandas.read_csv. for details, please refer to the official doc: [pandas.read_csv](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)

        for performance consideration, please specify the following args:
            - engine = 'c' or 'pyarrow' when possible. Be noticed that 'pyarrow' is the fastest (multi-threaded supported) but may be error-prone. Only use it when you have thoroughly tested.

            - specify parse_date and infer_datetime_format to True, if the data contains date-time format. The official doc says that will be 3~5 times faster.

            - specify lineterminator to '\n' or '\r\n', when the engine is 'c'

            - specify dtype when possible

        use `usecols` to specify the columns to read, and `names` to specify the column names (i.e., rename the columns), otherwise, the column names will be inferred from the first line.

        Args:
            sort_values: sort the dataframe by the specified columns
            encoding: if the data is bytes, then encoding is required, due to pandas.read_csv only handle string array
            sep : the seperator/delimiter of each fields
            header : the row number of the header, default is 'infer'
            names : the column names of the dataframe
            index_col : the column number or name of the index column
            usecols : the column number or name of the columns to use
            dtype : the dtype of the columns
            engine : the engine of the csv file, default is None
            converters : the converters of the columns
            skiprows : the row number to skip
            skipfooter : the row number to skip at the end of the file
            parse_dates : the columns to parse as dates
            infer_datetime_format : whether to infer the datetime format
            date_parser : the function to parse the date
            lineterminator: the line terminator of the csv file, only valid when engine is 'c'
            kwargs : other arguments
        """
        self.sort_values = sort_values
        self.encoding = encoding
        self.sep = sep
        self.header = header
        self.names = names
        self.index_col = index_col
        self.usecols = usecols
        self.dtype = dtype
        self.engine = engine
        self.converters = converters
        self.skiprows = skiprows
        self.skipfooter = skipfooter
        self.infer_datetime_format = infer_datetime_format
        self.date_parser = date_parser
        self.lineterminator = lineterminator
        self.kwargs = kwargs

        if isinstance(parse_dates, str):
            parse_dates = [parse_dates]

        self.parse_dates = parse_dates

        if names is not None:
            self.header = 0

    def __call__(self, data: bytes) -> pd.DataFrame:
        if self.encoding is None:
            # treat data as string
            stream = io.StringIO(data)
        else:
            stream = io.StringIO(data.decode(self.encoding))

        df = pd.read_csv(
            stream,
            sep=self.sep,
            header=self.header,
            names=self.names,
            index_col=self.index_col,
            usecols=self.usecols,
            dtype=self.dtype,
            engine=self.engine,
            converters=self.converters,
            skiprows=self.skiprows,
            skipfooter=self.skipfooter,
            parse_dates=self.parse_dates,
            infer_datetime_format=self.infer_datetime_format,
            date_parser=self.date_parser,
            lineterminator=self.lineterminator,
            **self.kwargs,
        )

        if self.sort_values is not None:
            return df.sort_values(self.sort_values)
        else:
            return df


class NumpyDeserializer(Serializer):
    def __init__(
        self,
        dtype: Union[dict, List[tuple]] = "float",
        sort_values: Union[str, List[str]] = None,
        use_cols: Union[List[int], List[str]] = None,
        sep: str = ",",
        encoding: str = None,
        skip_rows: Union[int, List[int]] = 0,
        comments: str = "#",
        converters: Dict[str, Callable] = None,
    ):
        """construct a deserializer, which will convert a csv like multiline string/bytes array to a numpy array

        dtype默认为float。如果反序列化后的对象是一个numpy structured array,则dtype可以设置为以下两种：

        by default dtype is float, which means the data will be converted to float. If you need to convert to a numpy structured array, then dtype could be one of the following format:

        ```
        dtype = [('col_1', 'datetime64[s]'), ('col_2', '<U12'), ('col_3', '<U4')]

        ```
        or

        ```
        dtype = {
            "names": ("col1", "col2", "col3"),
            "formats": ("datetime64[s]", "O", "float"),
        }
        ```

        if the `data` to be converted is a bytes array, then you should pass in the encoding to decode the bytes array.

        by default, the deserializer will try to convert every line from the very first line, if the very first lines contains comments and headers, these lines should be skipped by deserializer, you should set skip_rows to number of lines to skip.

        if you don't like to convert every column, you can set use_cols to a list of column names, to keep only those columns. If you do so and specified complex dtype, then the dtype and columns should be aligned.

        for more information, please refer to [numpy.loadtxt](https://numpy.org/doc/stable/reference/generated/numpy.loadtxt.html)

        """
        self.dtype = dtype
        self.use_cols = use_cols
        self.sep = sep
        self.encoding = encoding
        self.skip_rows = skip_rows
        self.comments = comments
        self.converters = converters
        self.sort_values = sort_values

    def __call__(self, data: bytes) -> DTypeLike:
        if self.encoding:
            stream = io.StringIO(data.decode(self.encoding))
        else:
            stream = io.StringIO(data)

        arr = np.loadtxt(
            stream.readlines(),
            delimiter=self.sep,
            skiprows=self.skip_rows,
            dtype=self.dtype,
            usecols=self.use_cols,
            converters=self.converters,
        )

        if self.sort_values is not None:
            return np.sort(arr, order=self.sort_values)
        else:
            return arr


class PyarrowDeserializer(Serializer):
    """PyArrow can provide best performance for large data."""

    def __init__(self) -> None:
        raise NotImplementedError


class DataframeSerializer(Serializer):
    DEFAULT_DECIMALS = 6

    def __init__(
        self,
        measurement: str,
        field_columns: List[str] = [],
        tag_columns: List[str] = [],
        global_tags: List[str] = {},
        time_precision: str = "s",
        precisions: dict = None,
        escape=False,
    ):
        """
        borrow ideas from [influxdb 1.x dataframe serializer](https://github.com/influxdata/influxdb-python/issues/363)

        compares to DataframeSerialize(influxdbClient), this one can NOT perform escape, but can set precisions per each column.

        Performance benchmark

        - to serialize 10000 points
            DataframeSerializer(Omicron): 0.1142 seconds
            DataframeSerializer(InfluxdbClient): 0.0893 seconds
            NumpySerializer(Omicron): 0.0698 seconds
        - to serialize 1M points
            DataframeSerializer(Omicron): 11.5 seconds
            DataframeSerializer(InfluxdbClient): 7.7 seconds
            NumpySerializer(Omicron): 7.2 seconds

        precisions is type of dict<int, List<str>>. the int is the precision of the field, the List<str> is the field names.

        Args:
            measurement : name of the measurement
            field_columns : the field columns. If not specified, then columns minus tag columns will be used.
            tag_columns : columns in dataframe which should be considered as tag columns
            global_tags : static tags, which will be added to every row.
            time_precision : precision for time field.
            precisions : precisions for floating fields. If not specified, then we'll stringify the column according to the type of the column, and default precision is assumed if it's floating type.
            escape: whether to escape special chars. If the data don't need to be escaped, then it's better to set it to False to improve performance.
        """
        self.measurement = measurement

        if isinstance(field_columns, str):
            self.field_columns = [field_columns]
        elif field_columns is None:
            self.field_columns = []
        else:
            self.field_columns = field_columns

        if isinstance(tag_columns, str):
            self.tag_columns = [tag_columns]
        elif tag_columns is None:
            self.tag_columns = []
        else:
            self.tag_columns = tag_columns

        self.global_tags = global_tags or {}
        self.time_precision = time_precision

        self.precisions = precisions or {}
        self.cols_with_precision = [
            v for v in itertools.chain(*self.precisions.values())
        ]

        self.escape = escape
        self.precision_factor = {"ns": 1, "us": 1e3, "ms": 1e6, "s": 1e9}.get(
            time_precision, 1
        )

    def _get_lines(self, dataframe):

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                "Must be DataFrame, but type was: {0}.".format(type(dataframe))
            )
        if not (
            isinstance(dataframe.index, pd.PeriodIndex)
            or isinstance(dataframe.index, pd.DatetimeIndex)
        ):
            raise TypeError(
                "Must be DataFrame with DatetimeIndex or \
                            PeriodIndex."
            )

        # Make array of timestamp ints
        time = (
            (dataframe.index.values.astype(int) / self.precision_factor)
            .astype(int)
            .astype(str)
        )

        # If tag columns exist, make an array of formatted tag keys and values
        if self.tag_columns:
            tag_df = dataframe[self.tag_columns]
            tags = ("," + ((tag_df.columns.values + "=").tolist() + tag_df)).sum(axis=1)
            del tag_df

        else:
            tags = ""

        # Make an array of formatted field keys and values
        fields = pd.DataFrame([], columns=self.field_columns)

        # handle fields which precision is specified
        for precision, cols in self.precisions.items():
            fields[cols] = dataframe[cols].round(precision)

        other_cols = set(fields.columns.tolist()) - set(self.cols_with_precision)

        # stringify the columns which is not floating type
        for col in other_cols:
            fields[col] = self._stringify(dataframe[col])

        fields = fields.astype(str)
        fields = fields.columns.values + "=" + fields
        fields[fields.columns[1:]] = "," + fields[fields.columns[1:]]
        fields = fields.sum(axis=1)

        # Add any global tags to formatted tag strings
        if self.global_tags:
            global_tags = ",".join(
                ["=".join([tag, self.global_tags[tag]]) for tag in self.global_tags]
            )
            if self.tag_columns:
                tags = tags + "," + global_tags
            else:
                tags = "," + global_tags

        # Generate line protocol string
        points = (self.measurement + tags + " " + fields + " " + time).tolist()
        return points

    def _stringify(self, data: ArrayLike) -> ArrayLike:
        if np.issubdtype(data.dtype, np.number):
            if np.issubdtype(data.dtype, np.signedinteger):
                return data.astype(str) + "i"
            elif np.issubdtype(data.dtype, np.unsignedinteger):
                return data.astype(str) + "u"
            else:
                return data.round(self.DEFAULT_DECIMALS).astype(str)
        elif np.issubdtype(data.dtype, np.bool_):
            return data.astype(str)
        else:  # treat as string
            return '"' + data.astype(str) + '"'

    def __call__(self, data: DataFrame, batch: int) -> str:
        for i in range(math.ceil(len(data) / batch)):
            yield "\n".join(self._get_lines(data[i * batch : (i + 1) * batch]))


class NumpySerializer(Serializer):
    DEFAULT_DECIMALS = 6

    def __init__(
        self,
        data: DTypeLike,
        measurement: str,
        time_column: str,
        tag_columns: List[str] = [],
        field_columns: List[str] = [],
        global_tags: Dict[str, Any] = {},
        time_precision: str = "s",
        precisions: Dict[str, int] = {},
    ):
        """
        serialize numpy structured array to influxdb line protocol.

        compares to DataframeSerialize(influxdbClient), this one can NOT perform escape, but can set precisions per each column.

        Performance benchmark

        - to serialize 10000 points
            DataframeSerializer(Omicron): 0.1142 seconds
            DataframeSerializer(InfluxdbClient): 0.0893 seconds
            NumpySerializer(Omicron): 0.0698 seconds
        - to serialize 1M points
            DataframeSerializer(Omicron): 11.5 seconds
            DataframeSerializer(InfluxdbClient): 7.7 seconds
            NumpySerializer(Omicron): 7.2 seconds

        Args:
            data: the numpy structured array to be serialized.
            measurement : name of the measurement
            time_column: from which column to get the timestamp.
            tag_columns : columns in dataframe which should be considered as tag columns
            field_columns : the field columns. If not specified, then columns minus tag columns will be used.
            global_tags : static tags, which will be added to every row.
            time_precision : precision for time field.
            escape: whether to escape special chars. If the data don't need to be escaped, then it's better to set it to False to improve performance.
            precisions: precisions for floating fields. If not specified, then we'll stringify the column according to the type of the column, and default precision is assumed if it's floating type.
        """
        if isinstance(tag_columns, str):
            tag_columns = [tag_columns]

        if isinstance(field_columns, str):
            field_columns = [field_columns]

        if len(field_columns) == 0:
            field_columns = sorted(set(data.dtype.names) - set(tag_columns)) - set(
                [time_column]
            )

        assert len(field_columns) > 0, "field_columns must not be empty"

        precision_factor = {"ns": 1, "us": 1e3, "ms": 1e6, "s": 1e9}.get(
            time_precision, 1
        )

        # construct format string
        # test,code=000001.XSHE a=1.1,b=2.024 631152000
        tags = [f"{tag}={{}}" for tag in tag_columns]

        fields = []
        for field in field_columns:
            if field in precisions:
                fields.append(f"{field}={{:.{precisions[field]}}}")
            else:
                if np.issubdtype(data.dtype[field], np.unsignedinteger):
                    fields.append(f"{field}={{}}u")
                elif np.issubdtype(data.dtype[field], np.signedinteger):
                    fields.append(f"{field}={{}}i")
                elif np.issubdtype(data.dtype[field], np.bool_):
                    fields.append(f"{field}={{}}")
                else:
                    fields.append(f'{field}="{{}}"')

        global_labels = "".join(
            f'{tag}="{value},"' for tag, value in global_tags.items()
        )
        self.format_string = (
            f"{measurement},"
            + global_labels
            + ",".join(tags)
            + " "
            + ",".join(fields)
            + " {}"
        )

        # transform data array so it can be serialized
        output_dtype = [
            (name, "O") for name in itertools.chain(tag_columns, field_columns)
        ]

        output_dtype.append(("frame", "i8"))

        # self.data = np.empty(len(data), dtype=output_dtype)

        # for col in self.tag_columns + self.field_columns:
        #     self.data[col] = data[col]

        cols = tag_columns + field_columns + [time_column]
        self.data = data[cols].astype(output_dtype)

        self.data["frame"] = (
            data[time_column].astype("M8[ns]").astype(int) / precision_factor
        )

    def _get_lines(self, data):
        return "\n".join([self.format_string.format(*row) for row in data])

    def __call__(self, batch: int) -> str:
        for i in range(math.ceil(len(self.data) / batch)):
            yield self._get_lines(self.data[i * batch : (i + 1) * batch])


if __name__ == "__main__":
    import datetime

    data = np.array(
        [
            (datetime.date(2022, 2, 1), "000001.XSHE", 1.1, 2, True, "上海银行"),
            (datetime.date(2022, 2, 2), "000002.XSHE", 1.1, 3, False, "平安银行"),
        ],
        dtype=[
            ("frame", "<M8[s]"),
            ("code", "<U11"),
            ("open", "<f4"),
            ("seq", "i4"),
            ("limit", "bool"),
            ("name", "<U11"),
        ],
    )

    serialize = NumpySerializer(
        data,
        "test",
        "frame",
        ["code", "name"],
        ["open", "seq", "limit"],
        global_tags={"a": "1", "b": "2"},
        precisions={"open": 3},
        time_precision="ns",
    )

    for lp in serialize(1):
        print(lp)

    df = pd.DataFrame(
        [("000001.XSHE", 1.1, 2.0235353), ("000002.XSHE", 2.0001, 3.1)],
        columns=["code", "a", "b"],
        index=[datetime.datetime(1990, 1, 1), datetime.datetime(1990, 1, 2)],
    )

    des = DataframeSerializer(
        measurement="test",
        field_columns=["a", "b"],
        tag_columns="code",
        precisions={2: ["a"], 3: ["b"]},
    )
    for lp in des(df, 1):
        print(lp)
