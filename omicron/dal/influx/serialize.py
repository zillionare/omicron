import datetime
import io
import itertools
import math
import re
import time
from email.generator import Generator
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import ANDArray
from pandas import DataFrame

from omicron.dal.influx.escape import KEY_ESCAPE, MEASUREMENT_ESCAPE, STR_ESCAPE


def _itertuples(data_frame):
    cols = [data_frame.iloc[:, k] for k in range(len(data_frame.columns))]
    return zip(data_frame.index, *cols)


def _not_nan(x):
    return x == x


def _any_not_nan(p, indexes):
    return any(map(lambda x: _not_nan(p[x]), indexes))


class Serializer(object):
    """base class of all serializer/deserializer"""

    pass


class DataframeSerializer:
    """Serialize DataFrame into LineProtocols.

    Most code is copied from [influxdb-python-client](https://github.com/influxdata/influxdb-client-python/blob/master/influxdb_client/client/write/dataframe_serializer.py), but modified interfaces.
    """

    def __init__(
        self,
        data_frame: DataFrame,
        measurement: str,
        tag_keys: Union[str, List[str]] = [],
        global_tags: Dict = {},
        precision="s",
    ) -> None:
        """Initialize DataframeSerializer.

        field keys are column names minus tag keys.

        Performance benchmark

        - to serialize 10000 points
            DataframeSerializer: 0.0893 seconds
            NumpySerializer: 0.0698 seconds
        - to serialize 1M points
            DataframeSerializer: 8.06 seconds
            NumpySerializer: 7.16 seconds

        Args:
            data_frame: DataFrame to be serialized.
            measurement: measurement name.
            tag_keys: List of tag keys.
            global_tags: global tags to be added to every row.
            precision: precision for write.
        """
        # This function is hard to understand but for good reason:
        # the approach used here is considerably more efficient
        # than the alternatives.
        #
        # We build up a Python expression that efficiently converts a data point
        # tuple into line-protocol entry, and then evaluate the expression
        # as a lambda so that we can call it. This avoids the overhead of
        # invoking a function on every data value - we only have one function
        # call per row instead. The expression consists of exactly
        # one f-string, so we build up the parts of it as segments
        # that are concatenated together to make the full f-string inside
        # the lambda.
        #
        # Things are made a little more complex because fields and tags with NaN
        # values and empty tags are omitted from the generated line-protocol
        # output.
        #
        # As an example, say we have a data frame with two value columns:
        #        a float
        #        b int
        #
        # This will generate a lambda expression to be evaluated that looks like
        # this:
        #
        #     lambda p: f"""{measurement_name} {keys[0]}={p[1]},{keys[1]}={p[2]}i {p[0].value}"""
        #
        # This lambda is then executed for each row p.
        #
        # When NaNs are present, the expression looks like this (split
        # across two lines to satisfy the code-style checker)
        #
        #    lambda p: f"""{measurement_name} {"" if math.isnan(p[1])
        #    else f"{keys[0]}={p[1]}"},{keys[1]}={p[2]}i {p[0].value}"""
        #
        # When there's a NaN value in column a, we'll end up with a comma at the start of the
        # fields, so we run a regexp substitution after generating the line-protocol entries
        # to remove this.
        #
        # We're careful to run these potentially costly extra steps only when NaN values actually
        # exist in the data.

        if not isinstance(data_frame, pd.DataFrame):
            raise TypeError(
                "Must be DataFrame, but type was: {0}.".format(type(data_frame))
            )

        data_frame = data_frame.copy(deep=False)
        if isinstance(data_frame.index, pd.PeriodIndex):
            data_frame.index = data_frame.index.to_timestamp()

        if not isinstance(data_frame.index, pd.DatetimeIndex):
            raise TypeError(
                "Must be DatetimeIndex, but type was: {0}.".format(
                    type(data_frame.index)
                )
            )

        if data_frame.index.tzinfo is None:
            data_frame.index = data_frame.index.tz_localize("UTC")

        if isinstance(tag_keys, str):
            tag_keys = [tag_keys]

        tag_keys = set(tag_keys or [])

        # keys holds a list of string keys.
        keys = []
        # tags holds a list of tag f-string segments ordered alphabetically by tag key.
        tags = []
        # fields holds a list of field f-string segments  ordered alphebetically by field key
        fields = []
        # field_indexes holds the index into each row of all the fields.
        field_indexes = []

        for key, value in global_tags.items():
            data_frame[key] = value
            tag_keys.add(key)

        # Get a list of all the columns sorted by field/tag key.
        # We want to iterate through the columns in sorted order
        # so that we know when we're on the first field so we
        # can know whether a comma is needed for that
        # field.
        columns = sorted(
            enumerate(data_frame.dtypes.items()), key=lambda col: col[1][0]
        )

        # null_columns has a bool value for each column holding
        # whether that column contains any null (NaN or None) values.
        null_columns = data_frame.isnull().any()

        # Iterate through the columns building up the expression for each column.
        for index, (key, value) in columns:
            key = str(key)
            key_format = f"{{keys[{len(keys)}]}}"
            keys.append(key.translate(KEY_ESCAPE))
            # The field index is one more than the column index because the
            # time index is at column zero in the finally zipped-together
            # result columns.
            field_index = index + 1
            val_format = f"p[{field_index}]"

            if key in tag_keys:
                # This column is a tag column.
                if null_columns[index]:
                    key_value = f"""{{
                            '' if {val_format} == '' or type({val_format}) == float and math.isnan({val_format}) else
                            f',{key_format}={{str({val_format}).translate(_ESCAPE_STRING)}}'
                        }}"""
                else:
                    key_value = (
                        f",{key_format}={{str({val_format}).translate(_ESCAPE_KEY)}}"
                    )
                tags.append(key_value)
                continue

            # This column is a field column.
            # Note: no comma separator is needed for the first field.
            # It's important to omit it because when the first
            # field column has no nulls, we don't run the comma-removal
            # regexp substitution step.
            sep = "" if len(field_indexes) == 0 else ","
            if issubclass(value.type, np.integer):
                field_value = f"{sep}{key_format}={{{val_format}}}i"
            elif issubclass(value.type, np.bool_):
                field_value = f"{sep}{key_format}={{{val_format}}}"
            elif issubclass(value.type, np.floating):
                if null_columns[index]:
                    field_value = f"""{{"" if math.isnan({val_format}) else f"{sep}{key_format}={{{val_format}}}"}}"""
                else:
                    field_value = f"{sep}{key_format}={{{val_format}}}"
            else:
                if null_columns[index]:
                    field_value = f"""{{
                            '' if type({val_format}) == float and math.isnan({val_format}) else
                            f'{sep}{key_format}="{{str({val_format}).translate(_ESCAPE_STRING)}}"'
                        }}"""
                else:
                    field_value = f'''{sep}{key_format}="{{str({val_format}).translate(_ESCAPE_STRING)}}"'''
            field_indexes.append(field_index)
            fields.append(field_value)

        measurement_name = str(measurement).translate(MEASUREMENT_ESCAPE)

        tags = "".join(tags)
        fields = "".join(fields)
        timestamp = "{p[0].value}"
        if precision.lower() == "us":
            timestamp = "{int(p[0].value / 1e3)}"
        elif precision.lower() == "ms":
            timestamp = "{int(p[0].value / 1e6)}"
        elif precision.lower() == "s":
            timestamp = "{int(p[0].value / 1e9)}"

        f = eval(
            f'lambda p: f"""{{measurement_name}}{tags} {fields} {timestamp}"""',
            {
                "measurement_name": measurement_name,
                "_ESCAPE_KEY": KEY_ESCAPE,
                "_ESCAPE_STRING": STR_ESCAPE,
                "keys": keys,
                "math": math,
            },
        )

        for k, v in dict(data_frame.dtypes).items():
            if k in tag_keys:
                data_frame[k].replace("", np.nan, inplace=True)

        self.data_frame = data_frame
        self.f = f
        self.field_indexes = field_indexes
        self.first_field_maybe_null = null_columns[field_indexes[0] - 1]

    def serialize(self, chunk_size: int) -> Generator:
        """Serialize chunk into LineProtocols."""
        for i in range(math.ceil(len(self.data_frame) / chunk_size)):
            chunk = self.data_frame[i * chunk_size : (i + 1) * chunk_size]
            if self.first_field_maybe_null:
                # When the first field is null (None/NaN), we'll have
                # a spurious leading comma which needs to be removed.
                lp = (
                    re.sub("^(( |[^ ])* ),([a-zA-Z])(.*)", "\\1\\3\\4", self.f(p))
                    for p in filter(
                        lambda x: _any_not_nan(x, self.field_indexes),
                        _itertuples(chunk),
                    )
                )
                yield "\n".join(lp)
            else:
                yield "\n".join(map(self.f, _itertuples(chunk)))


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


class NumpySerializer(Serializer):
    DEFAULT_DECIMALS = 6

    def __init__(
        self,
        data: ANDArray,
        measurement: str,
        time_key: str,
        tag_keys: List[str] = [],
        global_tags: Dict[str, Any] = {},
        time_precision: str = "s",
        precisions: Dict[str, int] = {},
    ):
        """
        serialize numpy structured array to influxdb line protocol.

        field keys are column names minus tag keys.

        compares to DataframeSerialize(influxdbClient), this one can NOT perform escape, but can set precisions per each column.

        Performance benchmark

        - to serialize 10000 points
            DataframeSerializer: 0.0893 seconds
            NumpySerializer: 0.0698 seconds
        - to serialize 1M points
            DataframeSerializer: 8.06 seconds
            NumpySerializer: 7.16 seconds

        Args:
            data: the numpy structured array to be serialized.
            measurement : name of the measurement
            time_key: from which column to get the timestamp.
            tag_keys : columns in dataframe which should be considered as tag columns
            global_tags : static tags, which will be added to every row.
            time_precision : precision for time field.
            escape: whether to escape special chars. If the data don't need to be escaped, then it's better to set it to False to improve performance.
            precisions: precisions for floating fields. If not specified, then we'll stringify the column according to the type of the column, and default precision is assumed if it's floating type.
        """
        if isinstance(tag_keys, str):
            tag_keys = [tag_keys]

        field_keys = sorted(set(data.dtype.names) - set(tag_keys) - set([time_key]))

        assert len(field_keys) > 0, "field_columns must not be empty"

        precision_factor = {"ns": 1, "us": 1e3, "ms": 1e6, "s": 1e9}.get(
            time_precision, 1
        )

        data = data.copy()
        if np.issubdtype(data[time_key].dtype, np.datetime64):
            data[time_key] = (
                data[time_key].astype("M8[ns]").astype(int) / precision_factor
            )
        elif isinstance(data[time_key][0], datetime.date):
            factor = 1e9 / precision_factor
            data[time_key] = [
                time.mktime(x.timetuple()) * factor for x in data[time_key]
            ]
        else:
            raise TypeError(
                f"unsupported data type: expected datetime64 or date, got {type(data[time_key][0])}"
            )

        # construct format string
        # test,code=000001.XSHE a=1.1,b=2.024 631152000
        tags = [f"{tag}={{}}" for tag in tag_keys]

        fields = []
        for field in field_keys:
            if field in precisions:
                fields.append(f"{field}={{:.{precisions[field]}}}")
            else:
                if np.issubdtype(data[field].dtype, np.floating):
                    fields.append(f"{field}={{:.{self.DEFAULT_DECIMALS}}}")
                elif np.issubdtype(data.dtype[field], np.unsignedinteger):
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
        output_dtype = [(name, "O") for name in itertools.chain(tag_keys, field_keys)]

        output_dtype.append(("frame", "i8"))

        # self.data = np.empty(len(data), dtype=output_dtype)

        # for col in self.tag_columns + self.field_columns:
        #     self.data[col] = data[col]

        cols = tag_keys + field_keys + [time_key]
        self.data = data[cols].astype(output_dtype)

    def _get_lines(self, data):
        return "\n".join([self.format_string.format(*row) for row in data])

    def serialize(self, batch: int) -> Generator:
        for i in range(math.ceil(len(self.data) / batch)):
            yield self._get_lines(self.data[i * batch : (i + 1) * batch])


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

    def __call__(self, data: bytes) -> ANDArray:
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


if __name__ == "__main__":
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

    df_serializer = DataframeSerializer(df, "test", tag_keys="code", chunk_size=1)
    for lp in df_serializer():
        print(lp)
