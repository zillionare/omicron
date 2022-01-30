import io
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import DTypeLike


class Serializer(object):
    """base class of all serializer/deserializer"""

    pass


class DataFrameDeserializer(Serializer):
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
        **kwargs
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
            **self.kwargs
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
