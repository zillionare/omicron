import datetime
from collections import defaultdict
from typing import DefaultDict, List

import arrow
import numpy as np
from coretypes import Frame

from omicron.core.errors import DuplicateOperationError


class Flux(object):
    """Helper functions for building flux query expression"""

    EPOCH_START = datetime.datetime(1970, 1, 1, 0, 0, 0)

    def __init__(self):
        self._cols = None
        self.expressions = defaultdict(list)

    def __str__(self):
        return self._compose()

    def _compose(self):
        """将所有表达式合并为一个表达式"""
        if not all(
            [
                "bucket" in self.expressions,
                "measurement" in self.expressions,
                "range" in self.expressions,
            ]
        ):
            raise AssertionError("bucket, measurement and range must be set")

        expr = [self.expressions[k] for k in ("bucket", "range", "measurement")]

        if self.expressions.get("tags"):
            expr.append(self.expressions["tags"])

        if self.expressions.get("fields"):
            expr.append(self.expressions["fields"])

        if self.expressions.get("pivot"):
            expr.append(self.expressions["pivot"])

        if self.expressions.get("keep"):
            expr.append(self.expressions["keep"])

        if self.expressions.get("limit"):
            expr.append(self.expressions["limit"])

        return "\n".join(expr)

    def bucket(self, bucket: str) -> "Flux":
        """add bucket to query expression

        Raises:
            DuplicateOperationError: 一个查询中只允许指定一个source，如果表达式中已经指定了bucket，则抛出异常

        Returns:
            Flux对象
        """
        if "bucket" in self.expressions:
            raise DuplicateOperationError("bucket has been set")

        self.expressions["bucket"] = f'from(bucket: "{bucket}")'

        return self

    def measurement(self, measurement: str) -> "Flux":
        """add measurement filter to query

        Raises:
            DuplicateOperationError: 一次查询中只允许指定一个measurement, 如果表达式中已经存在measurement, 则抛出异常

        Returns:
            Flux对象自身，以便进行管道操作
        """
        if "measurement" in self.expressions:
            raise DuplicateOperationError("measurement has been set")

        self.expressions[
            "measurement"
        ] = f'  |> filter(fn: (r) => r["_measurement"] == "{measurement}")'

        return self

    def range(
        self, start: Frame, end: Frame, right_close=True, precision="s"
    ) -> "Flux":
        """添加时间范围过滤

        必须指定的查询条件，否则influxdb会报unbound查询错，因为这种情况下，返回的数据量将非常大。

        在格式化时间时，需要根据`precision`生成时间字符串。在向Influxdb发送请求时，应该注意查询参数中指定的时间精度与这里使用的保持一致。

        Influxdb的查询结果默认不包含结束时间，当`right_close`指定为True时，我们将根据指定的精度修改`end`时间，使之仅比`end`多一个时间单位，从而保证查询结果会包含`end`。

        Raises:
            DuplicateOperationError: 一个查询中只允许指定一次时间范围，如果range表达式已经存在，则抛出异常
        Args:
            start: 开始时间
            end: 结束时间
            right_close: 查询结果是否包含结束时间。
            precision: 时间精度，默认为秒。

        Returns:
            Flux对象，以支持管道操作
        """
        if "range" in self.expressions:
            raise DuplicateOperationError("range has been set")

        if precision not in ["s", "ms", "us"]:
            raise AssertionError("precision must be 's', 'ms' or 'us'")

        end = self.format_time(end, precision, right_close)
        start = self.format_time(start, precision)

        self.expressions["range"] = f"  |> range(start: {start}, stop: {end})"
        return self

    def limit(self, limit: int) -> "Flux":
        """添加返回记录数限制

        Raises:
            DuplicateOperationError: 一个查询中只允许指定一次limit，如果limit表达式已经存在，则抛出异常

        Args:
            limit: 返回记录数限制

        Returns:
            Flux对象，以便进行管道操作
        """
        if "limit" in self.expressions:
            raise DuplicateOperationError("limit has been set")

        self.expressions["limit"] = "  |> limit(n: %d)" % limit
        return self

    @classmethod
    def to_timestamp(cls, tm: Frame, precision: str = "s") -> int:
        """将时间根据精度转换为unix时间戳

        在往influxdb写入数据时，line-protocol要求的时间戳为unix timestamp，并且与其精度对应。

        Args:
            tm: 时间
            precision: 时间精度，默认为秒。

        Returns:
            时间戳
        """
        if precision not in ["s", "ms", "us"]:
            raise AssertionError("precision must be 's', 'ms' or 'us'")

        # get int repr of tm, in seconds unit
        if isinstance(tm, np.datetime64):
            tm = tm.astype("datetime64[s]").astype("int")
        elif getattr(tm, "timestamp", None):
            tm = tm.timestamp
        else:
            tm = arrow.get(tm).timestamp

        return int(tm * 10 ** ({"s": 0, "ms": 3, "us": 6}[precision]))

    @classmethod
    def format_time(cls, tm: Frame, precision: str = "s", shift_forward=False) -> str:
        """将时间转换成客户端对应的精度，并以 RFC3339 timestamps格式串（即influxdb要求的格式）返回。

        如果这个时间是作为查询的range中的结束时间使用时，由于influx查询的时间范围是左闭右开的，因此如果你需要查询的是一个闭区间，则需要将`end`的时间向前偏移一个精度。通过传入`shift_forward = True`可以完成这种转换。

        Examples:
            >>> # by default, the precision is seconds, and convert a date
            >>> Flux.format_time(datetime.date(2019, 1, 1))
            '2019-01-01T00:00:00Z'

            >>> # set precision to ms, convert a time
            >>> Flux.format_time(datetime.datetime(1978, 7, 8, 12, 34, 56, 123456), precision="ms")
            '1978-07-08T12:34:56.123Z'

            >>> # convert and forward shift
            >>> Flux.format_time(datetime.date(1978, 7, 8), shift_forward = True)
            '1978-07-08T00:00:01Z'

        Args:
            end : 待格式化的时间
            precision: 时间精度，可选值为：'s', 'ms', 'us'
            shift_forward: 如果为True，则将end向前偏移一个精度

        Returns:
            调整后符合influx时间规范的时间（字符串表示）
        """
        timespec = {"s": "seconds", "ms": "milliseconds", "us": "microseconds"}.get(
            precision
        )

        if timespec is None:
            raise ValueError(
                f"precision must be one of 's', 'ms', 'us', but got {precision}"
            )

        tm = arrow.get(tm).naive

        if shift_forward:
            tm = tm + datetime.timedelta(**{timespec: 1})

        return tm.isoformat(sep="T", timespec=timespec) + "Z"

    def tags(self, tags: DefaultDict[str, List[str]]) -> "Flux":
        """给查询添加tags过滤条件

        此查询条件为过滤条件，并非必须。如果查询中没有指定tags，则会返回所有记录。

        在实现上，既可以使用`contains`语法，也可以使用`or`语法(由于一条记录只能属于一个tag，所以，当指定多个tag进行查询时，它们之间的关系应该为`or`)。出于性能考虑，我们使用`contains`语法，这样在查询多个tag时，构建的flux语句会更短，从而节省语句构建时间、传输时间和服务器查询时间（possibly)。

        Raises:
            DuplicateOperationError: 一个查询中只允许执行一次，如果tag filter表达式已经存在，则抛出异常

        Args:
            tags : tags是一个{tagname: Union[str,[tag_values]]}对象。

        Examples:
            >>> flux = Flux()
            >>> flux.tags({"code": ["000001", "000002"], "name": ["浦发银行"]}).expressions["tags"]
            '  |> filter(fn: (r) => r["code"] == "000001" or r["code"] == "000002" or r["name"] == "浦发银行")'


        Returns:
            Flux对象，以便进行管道操作
        """
        if "tags" in self.expressions:
            raise DuplicateOperationError("tags has been set")

        filters = []
        for tag, values in tags.items():
            assert values, f"tag {tag} bind with no value"
            if isinstance(values, str):
                filters.append(f'r["{tag}"] == "{values}"')
            else:
                set_values = ",".join([f'"{v}"' for v in values])
                filters.append(f'contains(value: r["{tag}"], set: [{set_values}])')

        op_expression = " or ".join(filters)

        self.expressions["tags"] = f"  |> filter(fn: (r) => {op_expression})"

        return self

    def fields(self, fields: List) -> "Flux":
        """给查询添加field过滤条件

        此查询条件为过滤条件，用以指定哪些field会出现在查询结果中，并非必须。如果查询中没有指定tags，则会返回所有记录。

        由于一条记录只能属于一个_field，所以，当指定多个_field进行查询时，它们之间的关系应该为`or`。

        Raises:
            DuplicateOperationError: 一个查询中只允许执行一次，如果filed filter表达式已经存在，则抛出异常
        Args:
            fields: 待查询的field列表

        Returns:
            Flux对象，以便进行管道操作
        """
        if "fields" in self.expressions:
            raise DuplicateOperationError("fields has been set")

        self._cols = fields

        filters = [f'r["_field"] == "{name}"' for name in fields]

        self.expressions["fields"] = f"  |> filter(fn: (r) => {' or '.join(filters)})"

        return self

    def pivot(
        self,
        row_keys: List[str] = ["_time"],
        column_keys=["_field"],
        value_column: str = "_value",
    ) -> "Flux":
        """pivot用来将以列为单位的数据转换为以行为单位的数据

        Flux查询返回的结果通常都是以列为单位的数据，增加本pivot条件后，结果将被转换成为以行为单位的数据再返回。

        这里实现的是measurement内的转换，请参考 [pivot](https://docs.influxdata.com/flux/v0.x/stdlib/universe/pivot/#align-fields-within-each-measurement-that-have-the-same-timestamp)


        Args:
            row_keys: 惟一确定输出中一行数据的列名字。
            cols: 待转换的列名称列表, 默认为["_time"]
            column_keys: 列名称列表，默认为["_field"]

        Returns:
            Flux对象，以便进行管道操作
        """
        if "pivot" in self.expressions:
            raise DuplicateOperationError("pivot has been set")

        columns = ",".join([f'"{name}"' for name in column_keys])
        rowkeys = ",".join([f'"{name}"' for name in row_keys])

        self.expressions[
            "pivot"
        ] = f'  |> pivot(columnKey: [{columns}], rowKey: [{rowkeys}], valueColumn: "{value_column}")'

        return self

    def keep(
        self, columns: List[str] = None, reserve_time_stamp: bool = True
    ) -> "Flux":
        """指定查询中的哪些列被保留（即被传回客户端）

        如果columns为None,则使用之前传入的fields,如果fields也为空，则raise Error.

        如果reserve_time_stamp为True,则会在columns之上，再加上_time字段（此字段为缺省字段）。

        Args:
            columns: 待保留的列名称列表
            reserve_time_stamp: 是否保留_time字段，默认为True

        Returns:
            Flux对象，以便进行管道操作
        """
        if "keep" in self.expressions:
            raise DuplicateOperationError("keep has been set")

        self._cols = columns or self._cols
        if self._cols is None:
            raise ValueError("columns和fields必须至少提供一个。")

        if reserve_time_stamp and "_time" not in self._cols:
            self._cols.append("_time")

        columns = ",".join([f'"{name}"' for name in self.cols])

        self.expressions["keep"] = f"  |> keep(columns: [{columns}])"

        return self

    @property
    def cols(self):
        return sorted(self._cols)
