import asyncio
import datetime
import logging
import re
import time
from typing import Dict, Iterable, List, Union

import arrow
import cfg4py
import ciso8601
import numpy as np
import pandas as pd
from coretypes import Frame, FrameType, SecurityType, bars_cols, bars_dtype

from omicron.core.errors import BadParameterError, DataNotReadyError
from omicron.dal import cache
from omicron.dal.influx.flux import Flux
from omicron.dal.influx.influxclient import InfluxClient
from omicron.dal.influx.serialize import EPOCH, DataframeDeserializer
from omicron.models.timeframe import TimeFrame as tf

logger = logging.getLogger(__name__)
cfg = cfg4py.get_instance()


security_db_dtype = [
    ("frame", "O"),
    ("code", "U16"),
    ("info", "O"),
]

security_info_dtype = [
    ("code", "O"),
    ("alias", "O"),
    ("name", "O"),
    ("ipo", "datetime64[s]"),
    ("end", "datetime64[s]"),
    ("type", "O"),
]

_delta = np.timedelta64(1, "s")
_start = np.datetime64("1970-01-01T00:00:00Z")


def convert_nptime_to_datetime(x):
    # force using CST timezone
    ts = (x - _start) / _delta
    # tz=datetime.timezone.utc  --> UTC string
    _t = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
    return datetime.datetime(_t.year, _t.month, _t.day, _t.hour, _t.minute, _t.second)


class Query:
    """
    ["code", "alias(display_name)", "name", "ipo", "end", "type"]
    """

    def __init__(self, target_date: datetime.date = None, ds=None):
        if target_date is None:
            now = datetime.datetime.now()
            self.target_date = tf.day_shift(now.date(), 0)
        else:
            self.target_date = tf.day_shift(target_date, 0)

        # 名字，显示名，类型过滤器
        self._name_pattern = None  # 字母名字
        self._alias_pattern = None  # 显示名
        self._type_pattern = None  # 不指定则默认为全部，如果传入空值则只选择股票和指数
        # 开关选项
        self._exclude_kcb = False  # 科创板
        self._exclude_cyb = False  # 创业板
        self._exclude_st = False  # ST
        self._include_exit = False  # 是否包含已退市证券(默认包括当天退市的)
        # 下列开关优先级高于上面的
        self._only_kcb = False
        self._only_cyb = False
        self._only_st = False

    def only_cyb(self) -> "Query":
        self._only_cyb = True  # 高优先级
        self._exclude_cyb = False
        self._only_kcb = False
        self._only_st = False
        return self

    def only_st(self) -> "Query":
        self._only_st = True  # 高优先级
        self._exclude_st = False
        self._only_kcb = False
        self._only_cyb = False
        return self

    def only_kcb(self) -> "Query":
        self._only_kcb = True  # 高优先级
        self._exclude_kcb = False
        self._only_cyb = False
        self._only_st = False
        return self

    def exclude_st(self) -> "Query":
        self._exclude_st = True
        self._only_st = False
        return self

    def exclude_cyb(self) -> "Query":
        self._exclude_cyb = True
        self._only_cyb = False
        return self

    def exclude_kcb(self) -> "Query":
        self._exclude_kcb = True
        self._only_kcb = False
        return self

    def include_exit(self) -> "Query":
        self._include_exit = True
        return self

    def types(self, types: List[str]) -> "Query":
        """按类型过滤

        Args:
            # stock: {'index', 'stock'}
            # funds: {'etf', 'fjb', 'mmf', 'reits', 'fja', 'fjm', 'lof'}
        """
        if types is None or isinstance(types, List) is False:
            return self

        if len(types) == 0:
            self._type_pattern = ["index", "stock"]
        else:
            tmp = set(types)
            self._type_pattern = list(tmp)

        return self

    def name(self, name: str):
        if name is None or len(name) == 0:
            self._name_pattern = None
        else:
            self._name_pattern = name

        return self

    def alias(self, display_name: str):
        if display_name is None or len(display_name) == 0:
            self._alias_pattern = None
        else:
            self._alias_pattern = display_name

        return self

    async def eval(self):
        logger.debug("eval, date: %s", self.target_date)
        logger.debug(
            "eval, names and types: %s, %s, %s",
            self._name_pattern,
            self._alias_pattern,
            self._type_pattern,
        )
        logger.debug(
            "eval, exclude and include: %s, %s, %s, %s",
            self._exclude_cyb,
            self._exclude_st,
            self._exclude_kcb,
            self._include_exit,
        )
        logger.debug(
            "eval, only: %s, %s, %s ", self._only_cyb, self._only_st, self._only_kcb
        )

        records = await Security.load_securities_from_db(self.target_date)
        if records is None:
            return None

        t0 = time.time()

        results = []
        now = datetime.datetime.now()
        for record in records:
            if self._type_pattern is not None:
                if record["type"] not in self._type_pattern:
                    continue
            if self._name_pattern is not None:
                if record["name"].startswith(self._name_pattern) is False:
                    continue
            if self._alias_pattern is not None:
                if record["alias"].find(self._alias_pattern) == -1:
                    continue

            # 创业板，科创板，ST暂时限定为股票类型
            if self._only_cyb:
                if (
                    record["type"] != "stock"
                    or record["code"].startswith("300") is False
                ):
                    continue
            if self._only_kcb:
                if (
                    record["type"] != "stock"
                    or record["code"].startswith("688") is False
                ):
                    continue
            if self._only_st:
                if record["type"] != "stock" or record["alias"].find("ST") == -1:
                    continue
            if self._exclude_cyb:
                if record["type"] == "stock" and record["code"].startswith("300"):
                    continue
            if self._exclude_st:
                if record["type"] == "stock" and record["alias"].find("ST") != -1:
                    continue
            if self._exclude_kcb:
                if record["type"] == "stock" and record["code"].startswith("688"):
                    continue

            # 退市暂不限定是否为股票
            if self._include_exit is False:
                d1 = convert_nptime_to_datetime(record["end"]).date()
                if d1 <= now.date():
                    continue

            results.append(record)

        # 返回所有查询到的结果
        t1 = time.time()
        logger.debug("query cost using filters: %s", t1 - t0)
        return results


class Security:
    _securities = []
    _securities_date = None
    _security_types = set()
    _stocks = []

    @classmethod
    async def init(cls):
        # read all securities from redis, 7111 records now
        # {'index', 'stock'}
        # {'fjb', 'mmf', 'reits', 'fja', 'fjm'}
        # {'etf', 'lof'}
        if len(cls._securities) > 100:
            return True

        secs = await cls.load_securities()
        if secs is None or len(secs) == 0:  # pragma: no cover
            raise DataNotReadyError(
                "No securities in cache, make sure you have called omicron.init() first."
            )

        print("init securities done")
        return True

    @classmethod
    async def load_securities(cls):
        """加载所有证券的信息，并缓存到内存中。"""
        secs = await cache.security.lrange("security:all", 0, -1, encoding="utf-8")
        if len(secs) != 0:
            # using np.datetime64[s]
            _securities = np.array(
                [tuple(x.split(",")) for x in secs], dtype=security_info_dtype
            )

            # 更新证券类型列表
            cls._securities = _securities
            cls._security_types = set(_securities["type"])
            cls._stocks = _securities[
                (_securities["type"] == "stock") | (_securities["type"] == "index")
            ]
            logger.info(
                "%d securities loaded, types: %s", len(_securities), cls._security_types
            )

            date_in_cache = await cache.security.get("security:latest_date")
            if date_in_cache is not None:
                cls._securities_date = arrow.get(date_in_cache).date()
            else:
                cls._securities_date = datetime.date.today()

            return _securities
        else:  # pragma: no cover
            return None

    @classmethod
    def get_influx_client(cls):
        cfg = cfg4py.get_instance()
        url = cfg.influxdb.url
        token = cfg.influxdb.token
        org = cfg.influxdb.org
        bucket_name = cfg.influxdb.bucket_name
        return InfluxClient(url, token, bucket=bucket_name, org=org)

    @classmethod
    async def get_security_types(cls):
        if cls._security_types:
            return list(cls._security_types)
        else:
            return None

    @classmethod
    def get_stock(cls, code):
        if len(cls._securities) == 0:
            return None

        tmp = cls._securities[cls._securities["code"] == code]
        if len(tmp) > 0:
            if tmp["type"] in ["stock", "index"]:
                return tmp[0]

        return None

    @classmethod
    def fuzzy_match(cls, query: str):
        query = query.upper()
        if re.match(r"\d+", query):
            return {
                sec["code"]: sec.tolist()
                for sec in cls._securities
                if sec["code"].find(query) != -1
            }
        elif re.match(r"[A-Z]+", query):
            return {
                sec["code"]: sec.tolist()
                for sec in cls._securities
                if sec["name"].startswith(query)
            }
        else:
            return {
                sec["code"]: sec.tolist()
                for sec in cls._securities
                if sec["display_name"].find(query) != -1
            }

    @classmethod
    async def info(cls, code, date=None):
        _obj = await cls.query_security_via_date(code, date)
        if _obj is None:
            return None

        # "_time", "code", "type", "alias", "end", "ipo", "name"
        d1 = convert_nptime_to_datetime(_obj["ipo"]).date()
        d2 = convert_nptime_to_datetime(_obj["end"]).date()
        return {
            "type": _obj["type"],
            "display_name": _obj["alias"],
            "end": d2,
            "start": d1,
            "name": _obj["name"],
        }

    @classmethod
    async def name(cls, code, date=None):
        _security = await cls.query_security_via_date(code, date)
        if _security is None:
            return None
        return _security["name"]

    @classmethod
    async def alias(cls, code, date=None):
        return await cls.display_name(code, date)

    @classmethod
    async def display_name(cls, code, date=None):
        _security = await cls.query_security_via_date(code, date)
        if _security is None:
            return None
        return _security["alias"]

    @classmethod
    async def start_date(cls, code, date=None):
        _security = await cls.query_security_via_date(code, date)
        if _security is None:
            return None
        return convert_nptime_to_datetime(_security["ipo"]).date()

    @classmethod
    async def end_date(cls, code, date=None):
        _security = await cls.query_security_via_date(code, date)
        if _security is None:
            return None
        return convert_nptime_to_datetime(_security["end"]).date()

    @classmethod
    async def security_type(cls, code, date=None) -> SecurityType:
        _security = await cls.query_security_via_date(code, date)
        if _security is None:
            return None
        return _security["type"]

    @classmethod
    async def query_security_via_date(cls, code: str, date: datetime.date = None):
        if date is None:  # 从内存中查找
            date_in_cache = await cache.security.get("security:latest_date")
            if date_in_cache is not None:
                date = arrow.get(date_in_cache).date()
                if date > cls._securities_date:
                    await cls.load_securities()
            results = cls._securities[cls._securities["code"] == code]
        else:  # 从influxdb查找
            date = tf.day_shift(date, 0)
            results = await cls.load_securities_from_db(date, code)

        if results is not None and len(results) > 0:
            return results[0]
        else:
            return None

    @classmethod
    async def select(cls, date: datetime.date = None) -> Query:
        if date is None:
            return Query(target_date=None)
        else:
            return Query(target_date=date)

    @classmethod
    async def update_secs_cache(cls, dt: datetime.date, securities: List[str]):
        # stock: {'index', 'stock'}
        # funds: {'fjb', 'mmf', 'reits', 'fja', 'fjm'}
        # {'etf', 'lof'}
        key = "security:all"
        pipeline = cache.security.pipeline()
        pipeline.delete(key)
        for code, alias, name, start, end, _type in securities:
            pipeline.rpush(key, f"{code},{alias},{name},{start}," f"{end},{_type}")
        await pipeline.execute()
        logger.info("all securities saved to cache %s", key)

        # update latest date info
        await cache.security.set("security:latest_date", dt.strftime("%Y-%m-%d"))

    @classmethod
    async def save_securities(cls, securities: List[str], dt: datetime.date):
        """保存指定的证券信息到缓存中，并且存入influxdb，定时job调用本接口
        Args:
            securities: 证券代码列表。
        """
        # stock: {'index', 'stock'}
        # funds: {'fjb', 'mmf', 'reits', 'fja', 'fjm'}
        # {'etf', 'lof'}
        if dt is None or len(securities) == 0:
            return

        measurement = "security_list"
        client = cls.get_influx_client()

        # code, alias, name, start, end, type
        security_list = np.array(
            [
                (dt, x[0], f"{x[0]},{x[1]},{x[2]},{x[3]},{x[4]},{x[5]}")
                for x in securities
            ],
            dtype=security_db_dtype,
        )
        await client.save(
            security_list, measurement, time_key="frame", tag_keys=["code"]
        )

    @classmethod
    async def load_securities_from_db(
        cls, target_date: datetime.date, code: str = None
    ):
        if target_date is None:
            return None

        client = Security.get_influx_client()
        measurement = "security_list"

        flux = (
            Flux()
            .measurement(measurement)
            .range(target_date, target_date)
            .bucket(client._bucket)
            .fields(["info"])
        )
        if code is not None and len(code) > 0:
            flux.tags({"code": code})

        data = await client.query(flux)
        if len(data) == 2:  # \r\n
            return None

        ds = DataframeDeserializer(
            sort_values="_time",
            usecols=["_time", "code", "info"],
            time_col="_time",
            engine="c",
        )
        actual = ds(data)
        secs = actual.to_records(index=False)

        if len(secs) != 0:
            # "_time", "code", "code, alias, name, start, end, type"
            _securities = np.array(
                [tuple(x["info"].split(",")) for x in secs], dtype=security_info_dtype
            )
            return _securities
        else:
            return None

    @classmethod
    async def get_datescope_from_db(cls):
        client = Security.get_influx_client()
        measurement = "security_list"

        date1 = arrow.get("2005-01-01").date()
        date2 = arrow.now().naive.date()

        flux = (
            Flux()
            .measurement(measurement)
            .range(date1, date2)
            .bucket(client._bucket)
            .tags({"code": "000001.XSHE"})
        )

        data = await client.query(flux)
        print("flux is ", flux)
        if len(data) == 2:  # \r\n
            return None, None

        ds = DataframeDeserializer(
            sort_values="_time",
            usecols=["_time"],
            time_col="_time",
            engine="c",
        )
        actual = ds(data)
        secs = actual.to_records(index=False)

        if len(secs) != 0:
            d1 = convert_nptime_to_datetime(secs[0]["_time"])
            d2 = convert_nptime_to_datetime(secs[len(secs) - 1]["_time"])
            return d1.date(), d2.date()
        else:
            return None, None
