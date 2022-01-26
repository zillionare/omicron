import datetime
from collections import defaultdict
from typing import DefaultDict, List

import arrow
from coretypes import Frame

from omicron.core.errors import DuplicateOperationError


class Flux(object):
    """Helper functions for building flux query expression"""

    def __init__(self):
        self.expressions = defaultdict(list)

    def __str__(self):
        return self._compose()

    def _compose(self):
        """将所有表达式合并为一个表达式"""
        expr = [self.expressions[k] for k in ("bucket", "range", "measurement")]

        if self.expressions.get("tags"):
            expr.append(self.expressions["tags"])

        if self.expressions.get("fields"):
            expr.append(self.expressions["fields"])

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

        由于一条记录只能属于一个tag，所以，当指定多个tag进行查询时，它们之间的关系应该为`or`。

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
            filters.extend([f'r["{tag}"] == "{v}"' for v in values])

        op_expression = " or ".join(filters)

        self.expressions["tags"] = f"  |> filter(fn: (r) => {op_expression})"

        return self

    def fields(self, fields: List) -> "Flux":
        """给查询添加field过滤条件

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

        filters = [f'r["_field"] == "{name}"' for name in fields]

        self.expressions["fields"] = f"  |> filter(fn: (r) => {' or '.join(filters)})"

        return self
