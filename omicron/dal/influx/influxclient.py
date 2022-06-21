import datetime
import gzip
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Union

import arrow
import numpy as np
from aiohttp import ClientSession
from pandas import DataFrame

from omicron.core.errors import BadParameterError
from omicron.dal.influx.errors import (
    InfluxDBQueryError,
    InfluxDBWriteError,
    InfluxDeleteError,
    InfluxSchemaError,
)
from omicron.dal.influx.flux import Flux
from omicron.dal.influx.serialize import DataframeSerializer, NumpySerializer

logger = logging.getLogger(__name__)


class InfluxClient:
    def __init__(
        self,
        url: str,
        token: str,
        bucket: str,
        org: str = None,
        enable_compress=False,
        chunk_size: int = 5000,
        precision: str = "s",
    ):
        """[summary]

        Args:
            url ([type]): [description]
            token ([type]): [description]
            bucket ([type]): [description]
            org ([type], optional): [description]. Defaults to None.
            enable_compress ([type], optional): [description]. Defaults to False.
            chunk_size: number of lines to be saved in one request
            precision: 支持的时间精度
        """
        self._url = url
        self._bucket = bucket
        self._enable_compress = enable_compress
        self._org = org
        self._org_id = None  # 需要时通过查询获取，此后不再更新
        self._token = token

        # influxdb 2.0起支持的时间精度有：ns, us, ms, s。本客户端只支持s, ms和us
        self._precision = precision.lower()
        if self._precision not in ["s", "ms", "us"]:  # pragma: no cover
            raise ValueError("precision must be one of ['s', 'ms', 'us']")

        self._chunk_size = chunk_size

        # write
        self._write_url = f"{self._url}/api/v2/write?org={self._org}&bucket={self._bucket}&precision={self._precision}"

        self._write_headers = {
            "Content-Type": "text/plain; charset=utf-8",
            "Authorization": f"Token {token}",
            "Accept": "application/json",
        }

        if self._enable_compress:
            self._write_headers["Content-Encoding"] = "gzip"

        self._query_url = f"{self._url}/api/v2/query?org={self._org}"
        self._query_headers = {
            "Authorization": f"Token {token}",
            "Content-Type": "application/vnd.flux",
            # influx查询结果格式，无论如何指定（或者不指定），在2.1中始终是csv格式
            "Accept": "text/csv",
        }

        if self._enable_compress:
            self._query_headers["Accept-Encoding"] = "gzip"

        self._delete_url = (
            f"{self._url}/api/v2/delete?org={self._org}&bucket={self._bucket}"
        )
        self._delete_headers = {
            "Authorization": f"Token {token}",
            "Content-Type": "application/json",
        }

    async def save(
        self,
        data: Union[np.ndarray, DataFrame],
        measurement: str = None,
        tag_keys: List[str] = [],
        time_key: str = None,
        global_tags: Dict = {},
        chunk_size: int = None,
    ) -> None:
        """save `data` into influxdb

        if `data` is a pandas.DataFrame or numy structured array, it will be converted to line protocol and saved. If `data` is str, use `write` method instead.

        Args:
            data: data to be saved
            measurement: the name of measurement
            tag_keys: which columns name will be used as tags
            chunk_size: number of lines to be saved in one request. if it's -1, then all data will be written in one request. If it's None, then it will be set to `self._chunk_size`

        Raises:
            InfluxDBWriteError: if write failed

        """
        # todo: add more errors raise
        if isinstance(data, DataFrame):
            assert (
                measurement is not None
            ), "measurement must be specified when data is a DataFrame"

            if tag_keys:
                assert set(tag_keys) in set(
                    data.columns.tolist()
                ), "tag_keys must be in data.columns"

            serializer = DataframeSerializer(
                data,
                measurement,
                time_key,
                tag_keys,
                global_tags,
                precision=self._precision,
            )
            if chunk_size == -1:
                chunk_size = len(data)

            for lines in serializer.serialize(chunk_size or self._chunk_size):
                await self.write(lines)
        elif isinstance(data, np.ndarray):
            assert (
                measurement is not None
            ), "measurement must be specified when data is a numpy array"
            assert (
                time_key is not None
            ), "time_key must be specified when data is a numpy array"
            serializer = NumpySerializer(
                data,
                measurement,
                time_key,
                tag_keys,
                global_tags,
                time_precision=self._precision,
            )
            if chunk_size == -1:
                chunk_size = len(data)
            for lines in serializer.serialize(chunk_size or self._chunk_size):
                await self.write(lines)
        else:
            raise TypeError(
                f"data must be pandas.DataFrame, numpy array, got {type(data)}"
            )

    async def write(self, line_protocol: str):
        """将line-protocol数组写入influxdb

        Args:
            line_protocol: 待写入的数据，以line-protocol数组形式存在

        """
        # todo: add raise error declaration
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
                        f"influxdb write failed, err: {err['message']}"
                    )

    async def query(self, flux: Union[Flux, str], deserializer: Callable = None) -> Any:
        """flux查询

        flux查询结果是一个以annotated csv格式存储的数据，例如：
        ```
        ,result,table,_time,code,amount,close,factor,high,low,open,volume
        ,_result,0,2019-01-01T00:00:00Z,000001.XSHE,100000000,5.15,1.23,5.2,5,5.1,1000000
        ```

        上述`result`中，事先通过Flux.keep()限制了返回的字段为_time,code,amount,close,factor,high,low,open,volume。influxdb查询返回结果时，总是按照字段名称升序排列。此外，总是会额外地返回_result, table两个字段。

        如果传入了deserializer，则会调用deserializer将其解析成为python对象。否则，返回bytes数据。

        Args:
            flux: flux查询语句
            deserializer: 反序列化函数

        Returns:
            如果未提供反序列化函数，则返回结果为bytes array(如果指定了compress=True，返回结果为gzip解压缩后的bytes array)，否则返回反序列化后的python对象
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
                        f"influxdb query error: {err} when processing {flux[:500]}"
                    )
                    logger.debug("data caused error:%s", flux)
                    raise InfluxDBQueryError(
                        f"influxdb query failed, status code: {err['message']}"
                    )
                else:
                    # auto-unzip
                    body = await resp.read()
                    if deserializer:
                        try:
                            return deserializer(body)
                        except Exception as e:
                            logger.exception(e)
                            logger.warning(
                                "failed to deserialize data: %s, the query is:%s",
                                body,
                                flux[:500],
                            )
                            raise
                    else:
                        return body

    async def drop_measurement(self, measurement: str):
        """从influxdb中删除一个measurement

        调用此方法后，实际上该measurement仍然存在，只是没有数据。

        """
        # todo: add raise error declaration
        await self.delete(measurement, arrow.now().naive)

    async def delete(
        self,
        measurement: str,
        stop: datetime.datetime,
        tags: Optional[Dict[str, str]] = {},
        start: datetime.datetime = None,
        precision: str = "s",
    ):
        """删除influxdb中指定时间段内的数据

        关于参数，请参见[Flux.delete][omicron.dal.influx.flux.Flux.delete]。

        Args:
            measurement: 指定measurement名字
            stop: 待删除记录的结束时间
            start: 待删除记录的开始时间，如果未指定，则使用EPOCH_START
            tags: 按tag进行过滤的条件
            precision: 用以格式化起始和结束时间。

        Raises:
            InfluxDeleteError: 如果删除失败，则抛出此异常
        """
        # todo: add raise error declaration
        command = Flux().delete(
            measurement, stop, tags, start=start, precision=precision
        )

        async with ClientSession() as session:
            async with session.post(
                self._delete_url, data=json.dumps(command), headers=self._delete_headers
            ) as resp:
                if resp.status != 204:
                    err = await resp.json()
                    logger.warning(
                        "influxdb delete error: %s when processin command %s",
                        err["message"],
                        command,
                    )
                    raise InfluxDeleteError(
                        f"influxdb delete failed, status code: {err['message']}"
                    )

    async def list_buckets(self) -> List[Dict]:
        """列出influxdb中对应token能看到的所有的bucket

        Returns:
            list of buckets, each bucket is a dict with keys:
            ```
            id
            orgID, a 16 bytes hex string
            type, system or user
            description
            name
            retentionRules
            createdAt
            updatedAt
            links
            labels
        ```
        """
        url = f"{self._url}/api/v2/buckets"
        headers = {"Authorization": f"Token {self._token}"}
        async with ClientSession() as session:
            async with session.get(url, headers=headers) as resp:
                if resp.status != 200:
                    err = await resp.json()
                    raise InfluxSchemaError(
                        f"influxdb list bucket failed, status code: {err['message']}"
                    )
                else:
                    return (await resp.json())["buckets"]

    async def delete_bucket(self, bucket_id: str = None):
        """删除influxdb中指定bucket

        Args:
            bucket_id: 指定bucket的id。如果为None，则会删除本client对应的bucket。
        """
        if bucket_id is None:
            buckets = await self.list_buckets()
            for bucket in buckets:
                if bucket["type"] == "user" and bucket["name"] == self._bucket:
                    bucket_id = bucket["id"]
                    break
            else:
                raise BadParameterError(
                    "bucket_id is None, and we can't find bucket with name: %s"
                    % self._bucket
                )

        url = f"{self._url}/api/v2/buckets/{bucket_id}"
        headers = {"Authorization": f"Token {self._token}"}
        async with ClientSession() as session:
            async with session.delete(url, headers=headers) as resp:
                if resp.status != 204:
                    err = await resp.json()
                    logger.warning(
                        "influxdb delete bucket error: %s when processin command %s",
                        err["message"],
                        bucket_id,
                    )
                    raise InfluxSchemaError(
                        f"influxdb delete bucket failed, status code: {err['message']}"
                    )

    async def create_bucket(
        self, description=None, retention_rules: List[Dict] = None, org_id: str = None
    ) -> str:
        """创建influxdb中指定bucket

        Args:
            description: 指定bucket的描述
            org_id: 指定bucket所属的组织id，如果未指定，则使用本client对应的组织id。

        Raises:
            InfluxSchemaError: 当influxdb返回错误时，比如重复创建bucket等，会抛出此异常
        Returns:
            新创建的bucket的id
        """
        if org_id is None:
            org_id = await self.query_org_id()

        url = f"{self._url}/api/v2/buckets"
        headers = {"Authorization": f"Token {self._token}"}
        data = {
            "name": self._bucket,
            "orgID": org_id,
            "description": description,
            "retentionRules": retention_rules,
        }
        async with ClientSession() as session:
            async with session.post(
                url, data=json.dumps(data), headers=headers
            ) as resp:
                if resp.status != 201:
                    err = await resp.json()
                    logger.warning(
                        "influxdb create bucket error: %s when processin command %s",
                        err["message"],
                        data,
                    )
                    raise InfluxSchemaError(
                        f"influxdb create bucket failed, status code: {err['message']}"
                    )
                else:
                    result = await resp.json()
                    return result["id"]

    async def list_organizations(self, offset: int = 0, limit: int = 100) -> List[Dict]:
        """列出本客户端允许查询的所组织

        Args:
            offset : 分页起点
            limit : 每页size

        Raises:
            InfluxSchemaError: influxdb返回的错误

        Returns:
            list of organizations, each organization is a dict with keys:
            ```
            id      : the id of the org
            links
            name    : the name of the org
            description
            createdAt
            updatedAt
            ```
        """
        url = f"{self._url}/api/v2/orgs?offset={offset}&limit={limit}"
        headers = {"Authorization": f"Token {self._token}"}

        async with ClientSession() as session:
            async with session.get(url, headers=headers) as resp:
                if resp.status != 200:
                    err = await resp.json()
                    logger.warning("influxdb query orgs err: %s", err["message"])
                    raise InfluxSchemaError(
                        f"influxdb query orgs failed, status code: {err['message']}"
                    )
                else:
                    return (await resp.json())["orgs"]

    async def query_org_id(self, name: str = None) -> str:
        """通过组织名查找组织id

        只能查的本客户端允许查询的组织。如果name未提供，则使用本客户端创建时传入的组织名。

        Args:
            name: 指定组织名

        Returns:
            组织id
        """
        if name is None:
            name = self._org
        orgs = await self.list_organizations()
        for org in orgs:
            if org["name"] == name:
                return org["id"]

        raise BadParameterError(f"can't find org with name: {name}")
