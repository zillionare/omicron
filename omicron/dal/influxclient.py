import datetime
import gzip
import logging
from itertools import chain
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

import arrow
import numpy as np
from aiohttp import ClientSession
from attr import field
from coretypes import Frame
from influxdb_client import InfluxDBClient

from omicron.core.errors import InfluxDBQueryError, InfluxDBWriteError
from omicron.dal.flux import Flux

logger = logging.getLogger(__name__)


class InfluxClient(InfluxDBClient):
    def __init__(
        self,
        url: str,
        token: str,
        bucket: str,
        org: str = None,
        enable_compress=False,
        **kwargs,
    ):
        """[summary]

        Args:
            url ([type]): [description]
            token ([type]): [description]
            bucket ([type]): [description]
            org ([type], optional): [description]. Defaults to None.
            enable_compress ([type], optional): [description]. Defaults to False.
            precision: 支持的时间精度
        """
        super().__init__(url, token, org=org, enable_gzip=enable_compress, **kwargs)

        self._bucket = bucket
        self._enable_compress = enable_compress

        # influxdb 2.0起支持的时间精度有：ns, us, ms, s。本客户端只支持s, ms和us
        self._precision = kwargs.get("precision", "s")
        if self._precision not in ["s", "ms", "us"]:
            raise ValueError("precision must be one of ['s', 'ms', 'us']")

        self._batch_write_size = kwargs.get("batch_write_size", 5000)

        self._write_headers = {
            "Content-Type": "text/plain; charset=utf-8",
            "Authorization": f"Token {token}",
            "Accept": "application/json",
        }

        if self._enable_compress:
            self._write_headers["Content-Encoding"] = "gzip"

        # influx查询结果，在2.1中始终是csv格式
        self._query_headers = {
            "Authorization": f"Token {token}",
            "Content-Type": "application/vnd.flux",
        }

        if self._enable_compress:
            self._query_headers["Accept-Encoding"] = "gzip"

        self._write_url = f"{self.url}/api/v2/write?org={self.org}&bucket={self._bucket}&precision={self._precision}"

        self._query_url = f"{self.url}/api/v2/query?org={self.org}"

    def nparray_to_line_protocol(
        self,
        measurement: str,
        data: np.ndarray,
        tags: Union[set, str] = None,
        field_keys: List[str] = None,
        tm_key: str = None,
        formatters: dict = {},
    ) -> Generator:
        """将由`data`（由numpy structure array表示，同属于**一个series**<即具有相同的tags和retention policy>）的多个数据点转换为line-protocol数据

        `data`为要存储的数据，是一个numpy structured array，其中的每一行都是一个数据点。如果`tm_key`存在，则必须在`data`中存在`tm_key`列，否则，该组数据的timestamp将由服务器决定。

        formatters为一个字典，其中的键为data中的列名，其值为将列值转换为字符串的格式化串。

        Args:
            measurement: measurement(即table/collection)名称
            data: 待存储的数据
            tags: 如果为set，则将从`data`中取对应字段和值构成tags；如果为str,则认为是预先生成好的tags
            field_keys: data中的哪些字段会当作fields存储。如果未提供，则将除`tm_key`以外的全部字段当成fields存储
            tm_key: 时间戳列名
            formatters: 格式化字符串

        Returns:
            line-protocol数据, str
        """
        field_keys = list(field_keys or data.dtype.names)
        if tm_key:
            assert tm_key in data.dtype.names, f"{tm_key} not exist in data"
            if tm_key in field_keys:
                field_keys.remove(tm_key)

        lps = []
        for row in data:
            fields = []
            tm = Flux.to_timestamp(row[tm_key], self._precision) if tm_key else ""

            for key in field_keys:
                fmt = formatters.get(key, "{}")
                fields.append(f"{key}={fmt.format(row[key])}")

            if isinstance(tags, set):
                tags_ = ",".join([f"{k}={row[k]}" for k in tags])
            else:
                tags_ = tags

            lp_row = f"{measurement},{tags_} {','.join(fields)} {tm}"

            lps.append(lp_row)

        return "\n".join(lps)

    async def write(self, line_protocol: str):
        """将line-protocol数组写入influxdb

        Args:
            line_protocol: 待写入的数据，以line-protocol数组形式存在
        """
        if self._enable_compress:
            line_protocol_ = gzip.compress(line_protocol.encode("utf-8"))
        else:
            line_protocol_ = line_protocol

        async with ClientSession() as session:
            async with session.post(
                self._write_url, data=line_protocol_, headers=self._write_headers
            ) as resp:
                if resp.status != 204:
                    err = await resp.json()
                    logger.warning(
                        "influxdb write error when processing: %s, err code: %s, message: %s",
                        {line_protocol[:100]},
                        err["code"],
                        err["message"],
                    )
                    logger.debug("data caused error:%s", line_protocol)
                    raise InfluxDBWriteError(
                        f"influxdb write failed, err: {resp.status}"
                    )

    async def query(self, flux: Union[Flux, str], unserializer: Callable = None) -> Any:
        """flux查询

        flux查询结果是一个以annotated csv格式存储的数据，例如：
        ```
        ,result,table,_time,code,amount,close,factor,high,low,open,volume
        ,_result,0,2019-01-01T00:00:00Z,000001.XSHE,100000000,5.15,1.23,5.2,5,5.1,1000000
        ```

        上述`result`中，事先通过Flux.keep()限制了返回的字段为_time,code,amount,close,factor,high,low,open,volume。influxdb查询返回结果时，字段不会按查询时[keep][omicron.dal.flux.Flux.keep]指定的顺序排列，而总是按照字段名称升序排列。此外，总是会额外地返回_result, table两个字段。

        如果传入了unserializer，则会调用unserializer将其解析成为python对象。否则，返回bytes数据。

        Args:
            flux: flux查询语句
            unserializer: 反序列化函数

        Returns:
            返回查询结果
        """
        if isinstance(flux, Flux):
            flux = str(flux)

        async with ClientSession() as session:
            async with session.post(
                self._query_url, data=flux, headers=self._query_headers
            ) as resp:
                if resp.status != 200:
                    err = await resp.json()
                    logger.warning(
                        f"influxdb query error: {err} when processing {flux[:100]}"
                    )
                    logger.debug("data caused error:%s", flux)
                    raise InfluxDBQueryError(
                        f"influxdb query failed, status code: {resp.status}"
                    )
                else:
                    body = await resp.read()
                    if self._enable_compress:
                        body = gzip.decompress(body)

                    if unserializer:
                        return unserializer(body)
                    else:
                        return body
