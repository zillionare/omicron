import datetime
import logging
import re
from typing import Dict, List, Tuple

import arrow
import cfg4py
import numpy as np
from coretypes import SecurityType, security_info_dtype
from numpy.typing import NDArray

from omicron.core.errors import DataNotReadyError
from omicron.dal import cache
from omicron.dal.influx.flux import Flux
from omicron.dal.influx.influxclient import InfluxClient
from omicron.dal.influx.serialize import DataframeDeserializer
from omicron.models.timeframe import TimeFrame as tf
from omicron.notify.dingtalk import ding

logger = logging.getLogger(__name__)
cfg = cfg4py.get_instance()


security_db_dtype = [("frame", "O"), ("code", "U16"), ("info", "O")]

xrxd_info_dtype = [
    ("code", "O"),
    ("a_xr_date", "datetime64[s]"),
    ("bonusnote1", "O"),
    ("bonus_ratio", "<f4"),
    ("dividend_ratio", "<f4"),
    ("transfer_ratio", "<f4"),
    ("at_bonus_ratio", "<f4"),
    ("report_date", "datetime64[s]"),
    ("plan_progress", "O"),
    ("bonusnote2", "O"),
    ("bonus_cancel_pub_date", "datetime64[s]"),
]

_delta = np.timedelta64(1, "s")
_start = np.datetime64("1970-01-01 00:00:00")


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

    def __init__(self, target_date: datetime.date = None):
        if target_date is None:
            # 聚宽不一定会及时更新数据，因此db中不存放当天的数据，如果传空，查cache
            self.target_date = None
        else:
            # 如果是交易日，取当天，否则取前一天
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
        """返回结果中只包含创业板股票"""
        self._only_cyb = True  # 高优先级
        self._exclude_cyb = False
        self._only_kcb = False
        self._only_st = False
        return self

    def only_st(self) -> "Query":
        """返回结果中只包含ST类型的证券"""
        self._only_st = True  # 高优先级
        self._exclude_st = False
        self._only_kcb = False
        self._only_cyb = False
        return self

    def only_kcb(self) -> "Query":
        """返回结果中只包含科创板股票"""
        self._only_kcb = True  # 高优先级
        self._exclude_kcb = False
        self._only_cyb = False
        self._only_st = False
        return self

    def exclude_st(self) -> "Query":
        """从返回结果中排除ST类型的股票"""
        self._exclude_st = True
        self._only_st = False
        return self

    def exclude_cyb(self) -> "Query":
        """从返回结果中排除创业板类型的股票"""
        self._exclude_cyb = True
        self._only_cyb = False
        return self

    def exclude_kcb(self) -> "Query":
        """从返回结果中排除科创板类型的股票"""
        self._exclude_kcb = True
        self._only_kcb = False
        return self

    def include_exit(self) -> "Query":
        """从返回结果中排除已退市的证券"""
        self._include_exit = True
        return self

    def types(self, types: List[str]) -> "Query":
        """选择类型在`types`中的证券品种

        Args:
            types: 有效的类型包括: 对股票指数而言是（'index', 'stock'），对基金而言则是（'etf', 'fjb', 'mmf', 'reits', 'fja', 'fjm', 'lof'）
        """
        if types is None or isinstance(types, List) is False:
            return self

        if len(types) == 0:
            self._type_pattern = ["index", "stock"]
        else:
            tmp = set(types)
            self._type_pattern = list(tmp)

        return self

    def name_like(self, name: str) -> "Query":
        """查找股票/证券名称中出现`name`的品种

        注意这里的证券名称并不是其显示名。比如对中国平安000001.XSHE来说，它的名称是ZGPA，而不是“中国平安”。

        Args:
            name: 待查找的名字，比如"ZGPA"

        """
        if name is None or len(name) == 0:
            self._name_pattern = None
        else:
            self._name_pattern = name

        return self

    def alias_like(self, display_name: str) -> "Query":
        """查找股票/证券显示名中出现`display_name的品种

        Args:
            display_name: 显示名，比如“中国平安"
        """
        if display_name is None or len(display_name) == 0:
            self._alias_pattern = None
        else:
            self._alias_pattern = display_name

        return self

    async def eval(self) -> List[str]:
        """对查询结果进行求值，返回code列表

        Returns:
            代码列表
        """
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

        date_in_cache = await cache.security.get("security:latest_date")
        if date_in_cache:  # 无此数据说明omega有某些问题，不处理
            _date = arrow.get(date_in_cache).date()
        else:
            now = datetime.datetime.now()
            _date = tf.day_shift(now, 0)

        # 确定数据源，cache为当天8点之后获取的数据，数据库存放前一日和更早的数据
        if not self.target_date or self.target_date >= _date:
            self.target_date = _date

        records = None
        if self.target_date == _date:  # 从内存中查找，如果缓存中的数据已更新，重新加载到内存
            secs = await cache.security.lrange("security:all", 0, -1, encoding="utf-8")
            if len(secs) != 0:
                # using np.datetime64[s]
                records = np.array(
                    [tuple(x.split(",")) for x in secs], dtype=security_info_dtype
                )
        else:
            records = await Security.load_securities_from_db(self.target_date)
        if records is None:
            return None

        results = []
        for record in records:
            if self._type_pattern is not None:
                if record["type"] not in self._type_pattern:
                    continue
            if self._name_pattern is not None:
                if record["name"].find(self._name_pattern) == -1:
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
                if d1 < self.target_date:
                    continue

            results.append(record["code"])

        # 返回所有查询到的结果
        return results


class Security:
    _securities = []
    _securities_date = None
    _security_types = set()
    _stocks = []

    @classmethod
    async def init(cls):
        """初始化Security.

        一般而言，omicron的使用者无须调用此方法，它会在omicron初始化（通过`omicron.init`）时，被自动调用。

        Raises:
            DataNotReadyError: 如果omicron未初始化，或者cache中未加载最新证券列表，则抛出此异常。
        """
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
        """加载所有证券的信息，并缓存到内存中

        一般而言，omicron的使用者无须调用此方法，它会在omicron初始化（通过`omicron.init`）时，被自动调用。
        """
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
    def _get_influx_client(cls):
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
    def get_stock(cls, code) -> NDArray[security_info_dtype]:
        """根据`code`来查找对应的股票（含指数）对象信息。

        如果您只有股票代码，想知道该代码对应的股票名称、别名（显示名）、上市日期等信息，就可以使用此方法来获取相关信息。

        返回类型为`security_info_dtype`的numpy数组，但仅包含一个元素。您可以象字典一样存取它，比如
        ```python
            item = Security.get_stock("000001.XSHE")
            print(item["alias"])
        ```
        显示为"平安银行"

        Args:
            code: 待查询的股票/指数代码

        Returns:
            类型为`security_info_dtype`的numpy数组，但仅包含一个元素
        """
        if len(cls._securities) == 0:
            return None

        tmp = cls._securities[cls._securities["code"] == code]
        if len(tmp) > 0:
            if tmp["type"] in ["stock", "index"]:
                return tmp[0]

        return None

    @classmethod
    def fuzzy_match_ex(cls, query: str) -> Dict[str, Tuple]:
        # fixme: 此方法与Stock.fuzzy_match重复，并且进行了类型限制，使得其不适合放在Security里，以及作为一个通用方法

        query = query.upper()
        if re.match(r"\d+", query):
            return {
                sec["code"]: sec.tolist()
                for sec in cls._securities
                if sec["code"].find(query) != -1 and sec["type"] == "stock"
            }
        elif re.match(r"[A-Z]+", query):
            return {
                sec["code"]: sec.tolist()
                for sec in cls._securities
                if sec["name"].startswith(query) and sec["type"] == "stock"
            }
        else:
            return {
                sec["code"]: sec.tolist()
                for sec in cls._securities
                if sec["alias"].find(query) != -1 and sec["type"] == "stock"
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
            "alias": _obj["alias"],
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
        if date is None:  # 从内存中查找，如果缓存中的数据已更新，重新加载到内存
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
    def select(cls, date: datetime.date = None) -> Query:
        if date is None:
            return Query(target_date=None)
        else:
            return Query(target_date=date)

    @classmethod
    async def update_secs_cache(cls, dt: datetime.date, securities: List[Tuple]):
        """更新证券列表到缓存数据库中

        Args:
            dt: 证券列表归属的日期
            securities: 证券列表, 元素为元组，分别为代码、别名、名称、IPO日期、退市日和证券类型
        """
        # stock: {'index', 'stock'}
        # funds: {'fjb', 'mmf', 'reits', 'fja', 'fjm'}
        # {'etf', 'lof'}
        key = "security:all"
        pipeline = cache.security.pipeline()
        pipeline.delete(key)
        for code, alias, name, start, end, _type in securities:
            pipeline.rpush(key, f"{code},{alias},{name},{start}," f"{end},{_type}")
        await pipeline.execute()
        logger.info("all securities saved to cache %s, %d secs", key, len(securities))

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
        client = cls._get_influx_client()

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

        client = Security._get_influx_client()
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
        # fixme: 函数名无法反映用途，需要增加文档注释，说明该函数的作用,或者不应该出现在此类中？
        client = Security._get_influx_client()
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
        if len(data) == 2:  # \r\n
            return None, None

        ds = DataframeDeserializer(
            sort_values="_time", usecols=["_time"], time_col="_time", engine="c"
        )
        actual = ds(data)
        secs = actual.to_records(index=False)

        if len(secs) != 0:
            d1 = convert_nptime_to_datetime(secs[0]["_time"])
            d2 = convert_nptime_to_datetime(secs[len(secs) - 1]["_time"])
            return d1.date(), d2.date()
        else:
            return None, None

    @classmethod
    async def _notify_special_bonusnote(cls, code, note, cancel_date):
        # fixme: 这个函数应该出现在omega中？
        default_cancel_date = datetime.date(2099, 1, 1)  # 默认无取消公告
        # report this special event to notify user
        if cancel_date != default_cancel_date:
            ding("security %s, bonus_cancel_pub_date %s" % (code, cancel_date))

        if note.find("流通") != -1:  # 检查是否有“流通股”文字
            ding("security %s, special xrxd note: %s" % (code, note))

    @classmethod
    async def save_xrxd_reports(cls, reports: List[str], dt: datetime.date):
        # fixme: 此函数应该属于omega?

        """保存1年内的分红送股信息，并且存入influxdb，定时job调用本接口

        Args:
            reports: 分红送股公告
        """
        # code(0), a_xr_date, board_plan_bonusnote, bonus_ratio_rmb(3), dividend_ratio, transfer_ratio(5),
        # at_bonus_ratio_rmb(6), report_date, plan_progress, implementation_bonusnote, bonus_cancel_pub_date(10)

        if len(reports) == 0 or dt is None:
            return

        # read reports from db and convert to dict map
        reports_in_db = {}
        dt_start = dt - datetime.timedelta(days=366)  # 往前回溯366天
        dt_end = dt + datetime.timedelta(days=366)  # 往后延长366天
        existing_records = await cls._load_xrxd_from_db(None, dt_start, dt_end)
        for record in existing_records:
            code = record[0]
            if code not in reports_in_db:
                reports_in_db[code] = [record]
            else:
                reports_in_db[code].append(record)

        records = []  # 准备写入db

        for x in reports:
            code = x[0]
            note = x[2]
            cancel_date = x[10]

            existing_items = reports_in_db.get(code, None)
            if existing_items is None:  # 新记录
                record = (
                    x[1],
                    x[0],
                    f"{x[0]}|{x[1]}|{x[2]}|{x[3]}|{x[4]}|{x[5]}|{x[6]}|{x[7]}|{x[8]}|{x[9]}|{x[10]}",
                )
                records.append(record)
                await cls._notify_special_bonusnote(code, note, cancel_date)
            else:
                new_record = True
                for item in existing_items:
                    existing_date = convert_nptime_to_datetime(item[1]).date()
                    if existing_date == x[1]:  # 如果xr_date相同，不更新
                        new_record = False
                        continue
                if new_record:
                    record = (
                        x[1],
                        x[0],
                        f"{x[0]}|{x[1]}|{x[2]}|{x[3]}|{x[4]}|{x[5]}|{x[6]}|{x[7]}|{x[8]}|{x[9]}|{x[10]}",
                    )
                    records.append(record)
                    await cls._notify_special_bonusnote(code, note, cancel_date)

        logger.info("save_xrxd_reports, %d records to be saved", len(records))
        if len(records) == 0:
            return

        measurement = "security_xrxd_reports"
        client = cls._get_influx_client()
        # a_xr_date(_time), code(tag), info
        report_list = np.array(records, dtype=security_db_dtype)
        await client.save(report_list, measurement, time_key="frame", tag_keys=["code"])

    @classmethod
    async def _load_xrxd_from_db(
        cls, code, dt_start: datetime.date, dt_end: datetime.date
    ):
        if dt_start is None or dt_end is None:
            return []

        client = Security._get_influx_client()
        measurement = "security_xrxd_reports"

        flux = (
            Flux()
            .measurement(measurement)
            .range(dt_start, dt_end)
            .bucket(client._bucket)
            .fields(["info"])
        )
        if code is not None and len(code) > 0:
            flux.tags({"code": code})

        data = await client.query(flux)
        if len(data) == 2:  # \r\n
            return []

        ds = DataframeDeserializer(
            sort_values="_time",
            usecols=["_time", "code", "info"],
            time_col="_time",
            engine="c",
        )
        actual = ds(data)
        secs = actual.to_records(index=False)

        if len(secs) != 0:
            _reports = np.array(
                [tuple(x["info"].split("|")) for x in secs], dtype=xrxd_info_dtype
            )
            return _reports
        else:
            return []

    @classmethod
    async def get_xrxd_info(cls, dt: datetime.date, code: str = None):
        if dt is None:
            return None

        # code(0), a_xr_date, board_plan_bonusnote, bonus_ratio_rmb(3), dividend_ratio, transfer_ratio(5),
        # at_bonus_ratio_rmb(6), report_date, plan_progress, implementation_bonusnote, bonus_cancel_pub_date(10)
        reports = await cls._load_xrxd_from_db(code, dt, dt)
        if len(reports) == 0:
            return None

        readable_reports = []
        for report in reports:
            xr_date = convert_nptime_to_datetime(report[1]).date()
            readable_reports.append(
                {
                    "code": report[0],
                    "xr_date": xr_date,
                    "bonus": report[3],
                    "dividend": report[4],
                    "transfer": report[5],
                    "bonusnote": report[2],
                }
            )

        return readable_reports
