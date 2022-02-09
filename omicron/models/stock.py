import asyncio
import datetime
import logging
import re
from typing import Dict, Iterable, List, Union

import arrow
import cfg4py
import numpy as np
import pandas as pd
from coretypes import (
    Frame,
    FrameType,
    SecurityType,
    bars_cols,
    bars_dtype,
    bars_with_limit_cols,
    bars_with_limit_dtype,
)

from omicron.core.errors import DataNotReadyError
from omicron.dal import cache
from omicron.dal.influx.flux import Flux
from omicron.dal.influx.influxclient import InfluxClient
from omicron.dal.influx.serialize import (
    DataframeDeserializer,
    NumpyDeserializer,
    NumpySerializer,
)
from omicron.models.timeframe import TimeFrame

logger = logging.getLogger(__name__)
cfg = cfg4py.get_instance()


class Stock:
    """ "
    Stock对象用于归集某支证券（股票和指数，不包括其它投资品种）的相关信息，比如行情数据（OHLC等）、市值数据、所属概念分类等。
    """

    _stocks = None
    fields_type = [
        ("code", "O"),
        ("display_name", "O"),
        ("name", "O"),
        ("ipo", "O"),
        ("end", "O"),
        ("type", "O"),
    ]

    _cached_frames_start = {
        FrameType.DAY: None,
        FrameType.MIN60: None,
        FrameType.MIN30: None,
        FrameType.MIN15: None,
        FrameType.MIN5: None,
        FrameType.MIN1: None,
    }

    _is_cache_empty = True

    def __init__(self, code: str):
        self._code = code
        stock = self._stocks[self._stocks["code"] == code]
        assert stock, "系统中不存在该code"
        (
            _,
            self._display_name,
            self._name,
            self._start_date,
            self._end_date,
            _type,
        ) = stock[0]
        self._type = SecurityType(_type)

    @classmethod
    async def load_securities(cls):
        """加载所有证券的信息，并缓存到内存中。"""
        secs = await cache.security.lrange("security:stock", 0, -1, encoding="utf-8")
        if len(secs) != 0:
            _stocks = np.array(
                [tuple(x.split(",")) for x in secs], dtype=cls.fields_type
            )

            _stocks = _stocks[
                (_stocks["type"] == "stock") | (_stocks["type"] == "index")
            ]

            _stocks["ipo"] = [arrow.get(x).date() for x in _stocks["ipo"]]
            _stocks["end"] = [arrow.get(x).date() for x in _stocks["end"]]

            return _stocks
        else:  # pragma: no cover
            return None

    @classmethod
    async def init(cls):
        secs = await cls.load_securities()
        if len(secs) != 0:
            cls._stocks = secs
        else:  # pragma: no cover
            raise DataNotReadyError(
                "No securities in cache, make sure you have called omicron.init() first."
            )

    @classmethod
    async def save_securities(cls, securities: List[str]):
        """保存指定的证券到缓存中。

        Args:
            securities: 证券代码列表。
        """
        key = "security:stock"
        pipeline = cache.security.pipeline()
        pipeline.delete(key)
        for code, display_name, name, start, end, _type in securities:
            pipeline.rpush(
                key, f"{code},{display_name},{name},{start}," f"{end},{_type}"
            )
        await pipeline.execute()

    @classmethod
    def choose(
        cls,
        types: List[str] = ["stock", "index"],
        exclude_exit=True,
        exclude_st=True,
        exclude_300=False,
        exclude_688=True,
    ) -> list:
        """选择证券标的

        本函数用于选择部分证券标的。先根据指定的类型(`stock`, `index`等）来加载证券标的，再根
        据其它参数进行排除。

        Args:
            exclude_exit : 是否排除掉已退市的品种. Defaults to True.
            exclude_st : 是否排除掉作ST处理的品种. Defaults to True.
            exclude_300 : 是否排除掉创业板品种. Defaults to False.
            exclude_688 : 是否排除掉科创板品种. Defaults to True.

        Returns:
            筛选出的证券代码列表
        """
        cond = np.array([False] * len(cls._stocks))

        for type_ in types:
            cond |= cls._stocks["type"] == type_

        result = cls._stocks[cond]
        if exclude_exit:
            result = result[result["end"] > arrow.now().date()]
        if exclude_300:
            result = [rec for rec in result if not rec["code"].startswith("300")]
        if exclude_688:
            result = [rec for rec in result if not rec["code"].startswith("688")]
        if exclude_st:
            result = [rec for rec in result if rec["display_name"].find("ST") == -1]
        result = np.array(result, dtype=cls.fields_type)
        return result["code"].tolist()

    @classmethod
    def choose_cyb(cls):
        """选择创业板股票"""
        return [rec["code"] for rec in cls._stocks if rec["code"].startswith("300")]

    @classmethod
    def choose_kcb(cls):
        """选择科创板股票"""
        return [rec["code"] for rec in cls._stocks if rec["code"].startswith("688")]

    @classmethod
    def fuzzy_match(cls, query: str):
        """对股票/指数进行模糊匹配查找"""
        query = query.upper()
        if re.match(r"\d+", query):
            return {
                sec["code"]: sec.tolist()
                for sec in cls._stocks
                if sec["code"].startswith(query)
            }
        elif re.match(r"[A-Z]+", query):
            return {
                sec["code"]: sec.tolist()
                for sec in cls._stocks
                if sec["name"].startswith(query)
            }
        else:
            return {
                sec["code"]: sec.tolist()
                for sec in cls._stocks
                if sec["display_name"].find(query) != -1
            }

    def __str__(self):
        return f"{self.display_name}[{self.code}]"

    @property
    def ipo_date(self) -> datetime.date:
        return self._start_date

    @property
    def display_name(self) -> str:
        return self._display_name

    @property
    def name(self) -> str:
        return self._name

    @property
    def end_date(self) -> datetime.date:
        return self._end_date

    @property
    def code(self) -> str:
        return self._code

    @property
    def sim_code(self) -> str:
        return re.sub(r"\.XSH[EG]", "", self.code)

    @property
    def security_type(self) -> SecurityType:
        """返回证券类型

        Returns:
            SecurityType: [description]
        """
        return self._type

    @staticmethod
    def simplify_code(code) -> str:
        return re.sub(r"\.XSH[EG]", "", code)

    def days_since_ipo(self) -> int:
        """获取上市以来经过了多少个交易日

        由于受交易日历限制（2005年1月4日之前的交易日历没有），对于在之前上市的品种，都返回从2005年1月4日起的日期。

        Returns:
            int: [description]
        """
        epoch_start = arrow.get("2005-01-04").date()
        ipo_day = self.ipo_date if self.ipo_date > epoch_start else epoch_start
        return TimeFrame.count_day_frames(ipo_day, arrow.now().date())

    @staticmethod
    def qfq(bars: np.ndarray) -> np.ndarray:
        """对行情数据执行前复权操作"""
        if bars.size == 0:
            return bars

        last = bars[-1]["factor"]
        for field in ["open", "high", "low", "close", "volume"]:
            bars[field] = (bars[field] / last) * bars["factor"]

        return bars

    @classmethod
    async def __get_bars_in_range(
        cls,
        codes: Union[str, List[str]] = None,
        begin: Frame = None,
        end: Frame = None,
        frame_type: FrameType = FrameType.DAY,
        fq=True,
        n: bool = None,
        unclosed: bool = True,
    ):
        """获取在`[start, stop]`间的行情数据。

        Args:
            begin (Frame): [description]
            end (Frame): [description]
            frame_type (FrameType): [description]
            fq (bool, optional): [description]. Defaults to True.
        """
        codes = [codes] if isinstance(codes, str) else codes
        parts = []
        if not codes:
            codes = list(cls._stocks["code"])
        result = {}
        for code in codes:
            bars = np.empty((0,), dtype=bars_dtype)
            if n:
                part2 = await cls.get_bars(
                    code, n, end=end, frame_type=frame_type, unclosed=unclosed, fq=False
                )
                bars = part2
            else:
                part2 = await cls._get_cached_bars(code, end, 240, frame_type, unclosed)
                parts = []
                early_parts = []
                if len(part2):
                    part2 = await cls.get_bars(
                        code,
                        n=len(part2) or 0,
                        end=end,
                        frame_type=frame_type,
                        unclosed=unclosed,
                        fq=False,
                    )
                    parts = part2[part2["frame"] >= begin]
                    early_parts: np.array = part2[part2["frame"] < begin]
                if not len(early_parts):
                    bars = await cls._get_persisted_bars(
                        code=code,
                        begin=begin,
                        end=end,
                        frame_type=frame_type,
                        dtypes=bars_dtype,
                    )
                    bars = np.concatenate((bars, parts)) if len(parts) else bars
            if fq and len(bars):
                bars = cls.qfq(bars)
            result[code] = bars
        return result

    @classmethod
    async def batch_get_bars_in_range(
        cls,
        codes: Iterable[str],
        frame_type: FrameType,
        begin: Frame,
        end: Frame,
        fq=True,
        unclosed: bool = True,
    ) -> Dict[str, np.ndarray]:
        """获取在`[start, stop]`间的行情数据。

        Args:
            codes: 证券代码列表
            begin (Frame): [description]
            end (Frame): [description]
            frame_type (FrameType): [description]
            fq (bool, optional): [description]. Defaults to True.

        Returns:
            返回一个字典，key为证券代码，value为行情数据。value是一个dtype为`bars_dtype`的一维numpy数组。
        """
        tasks = []

        # make a dummpy empty array
        empty = np.empty((0,), dtype=bars_dtype)

        for code in codes:
            tasks.append(cls._get_cached_bars(code, end, 240, frame_type, unclosed))

        results = await asyncio.gather(*tasks)
        # results are gathered in the order of tasks are created
        part1 = {code: bars for code, bars in zip(codes, results)}

        part2 = await cls._batch_get_persisted_bars(
            frame_type, codes, begin=begin, end=end
        )

        result = {}
        for code in codes:
            part1_bars = part1.get(code, empty)
            part2_bars = part2.get(code, empty)

            bars = np.concatenate([part2_bars, part1_bars])
            if fq:
                bars = cls.qfq(bars)
            result[code] = bars

        return result

    @classmethod
    async def get_bars_in_range(
        cls,
        code: str,
        frame_type: FrameType,
        start: Frame,
        end: Frame,
        fq=True,
        unclosed=True,
    ) -> np.ndarray:
        """获取指定证券（`code`）在[`start`, `end`]期间帧类型为`frame_type`的行情数据。

        Args:
            code : 证券代码
            frame_type : 行情数据的帧类型
            start : 起始时间
            end : 结束时间
            fq : 是否对行情数据执行前复权操作
            unclosed : 是否包含未收盘的数据
        """
        n = TimeFrame.count_frames(start, end, frame_type)
        return await cls.get_bars(code, n, frame_type, end, fq, unclosed)

    @classmethod
    async def batch_get_bars(
        cls,
        codes: List[str],
        n: int,
        frame_type: FrameType,
        end: Frame = None,
        fq: bool = True,
        unclosed: bool = False,
    ) -> Dict[str, np.ndarray]:
        """获取多支股票（指数）的最近的`n`个行情数据。

        停牌数据处理请见[get_bars][omicron.models.stock.Stock.get_bars]。

        结果以dict方式返回，key为传入的股票代码，value为对应的行情数据。

        Args:
            codes (List[str]): 代码列表
            n (int): 返回记录数
            frame_type (FrameType): 帧类型
            end (Frame, optional): 结束时间。如果未指明，则取当前时间。 Defaults to None.
            fq (bool, optional): 是否进行复权，如果是，则进行前复权。Defaults to True.
            unclosed (bool, optional): 是否包含最新一期未收盘数据. Defaults to True.

        Returns:
            返回一个字典，其key为证券代码，其value为dtype为`bars_dtype`的一维numpy数组。
        """
        bars = cls.__get_bars_in_range(
            code=codes,
            fq=fq,
            n=n,
            unclosed=unclosed,
            frame_type=frame_type,
            end=end,
        )
        return bars

    @classmethod
    async def get_bars(
        cls,
        code: str,
        n: int,
        frame_type: FrameType,
        end: Frame = None,
        fq=True,
        unclosed=True,
    ) -> np.ndarray:
        """获取到`end`为止的`n`个行情数据。

        返回的数据是按照时间顺序递增排序的。在遇到停牌的情况时，该时段数据将被跳过，因此返回的记录可能不是交易日连续的。

        如果系统当前没有到指定时间`end`的数据，将尽最大努力返回数据。调用者可以通过判断最后一条数据的时间是否等于`end`来判断是否获取到了全部数据。

        Args:
            code (str): 证券代码
            n (int): 记录数
            frame_type (FrameType): 帧类型
            end (Frame): 截止时间,如果未指明，则取当前时间
            fq (bool, optional): [description]. Defaults to True.
            unclosed (bool, optional): 是否包含最新未收盘的数据？ Defaults to True.

        Returns:
            返回dtype为`coretypes.bars_dtype`的一维numpy数组。
        """
        end = end or arrow.now().naive

        part2 = await cls._get_cached_bars(code, end, n, frame_type, unclosed)

        n2 = len(part2)
        n1 = n - n2
        if n1 > 0:
            # todo: 如果事先计算出begin时间，是否会有性能提升？
            part1 = await cls._get_persisted_bars(
                code, end=end, n=n1, frame_type=frame_type
            )
            part1 = part1[(-n1):]
        else:
            part1 = np.empty((0,), dtype=bars_dtype)

        bars = np.concatenate([part1, part2])
        assert len(bars) == n

        if fq:
            bars = cls.qfq(bars)

        return bars

    @classmethod
    async def _get_persisted_bars(
        cls,
        code: str,
        frame_type: FrameType,
        n: int = None,
        begin: Frame = None,
        end: Frame = None,
    ) -> np.array:
        """从influxdb中获取数据

        `begin`和`n`必须指定至少一个。如果`end`未指定，则取当前时间。当`begin`和`n`都指定时，将只返回在`[begin, end]`范围内的最多`n`条数据。

        返回的数据按`frame`进行升序排列。

        Args:
            code (str): 证券代码
            frame_type: the frame_type to query
            n (int): 返回结果数量
            end (Frame): [description]

        Returns:
            返回dtype为`bars_dtype`的numpy数组
        """
        if n is None:
            assert begin is not None, "must specify `begin` or `n`"
        else:
            begin = begin or Flux.EPOCH_START

        end = end or arrow.now().naive

        keep_cols = bars_cols
        use_cols = list(range(3, 11))

        measurement = f"stock_bars_{frame_type.value}"
        query = (
            Flux()
            .bucket(cfg.influxdb.bucket_name)
            .range(begin, end)
            .measurement(measurement)
            .keep(keep_cols)
            .tags({"code": code})
        )

        if n is not None:
            query.limit(n)

        serializer = NumpyDeserializer(
            bars_dtype,
            sort_values="frame",
            encoding="utf-8",
            skip_rows=1,
            use_cols=use_cols,
        )

        url = cfg.influxdb.url
        token = cfg.influxdb.token
        bucket = cfg.influxdb.bucket_name
        org = cfg.influxdb.org

        client = InfluxClient(url, token, bucket, org)
        return await client.query(query, serializer)

    @classmethod
    async def _batch_get_persisted_bars(
        cls,
        frame_type: FrameType,
        codes: List[str] = None,
        n: int = None,
        begin: Frame = None,
        end: Frame = None,
    ) -> Dict[str, np.array]:
        """获取`codes`指定的一批股票在时间范围内的数据。

        `begin`和`n`必须指定至少一个。如果`end`未指定，则取当前时间。当`begin`和`n`都指定时，将只返回在`[begin, end]`范围内的最多`n`条数据。

        如果`frame_type == FrameType.DAY`，则返回的数据类型是`bars_with_limit_dtype`，否则返回的数据类型是`bars_dtype`。

        返回的数据按`frame`进行升序排列。

        如果`codes`为None,则返回指定时间段内所有股票的数据。

        注意，返回的数据有可能不是等长的。

        Args:
            frame_type : [description]
            code : [description].
            n : [description].
            begin : [description].
            end : [description].

        Returns:
            以`code`为key, 行情数据为value的字典。
        """
        if n is None:
            assert begin is not None, "must specify `begin` or `n`"
        else:
            begin = begin or Flux.EPOCH_START

        end = end or arrow.now().naive

        # influxdb的查询结果格式类似于CSV，其列顺序为_, result_alias, table_seq, _time, tags, fields,其中tags和fields都是升序排列
        if frame_type == FrameType.DAY:
            dtype = bars_with_limit_dtype
            return_cols = bars_with_limit_cols
            keep_cols = bars_with_limit_cols + ["code"]
            names = [
                "_",
                "result",
                "table",
                "frame",
                "code",
                "amount",
                "close",
                "factor",
                "high",
                "high_limit",
                "low",
                "low_limit",
                "open",
                "volume",
            ]
        else:
            dtype = bars_dtype
            return_cols = bars_cols
            keep_cols = bars_cols + ["code"]
            names = [
                "_",
                "result",
                "table",
                "frame",
                "code",
                "amount",
                "close",
                "factor",
                "high",
                "low",
                "open",
                "volume",
            ]

        measurement = f"stock_bars_{frame_type.value}"
        query = (
            Flux()
            .bucket(cfg.influxdb.bucket_name)
            .range(begin, end)
            .measurement(measurement)
            .keep(keep_cols)
        )

        if codes is not None:
            query.tags({"code": codes})
        if n is not None:
            query.limit(n)

        deserializer = DataframeDeserializer(
            names=names, usecols=keep_cols, encoding="utf-8", parse_dates="frame"
        )

        url = cfg.influxdb.url
        token = cfg.influxdb.token
        bucket = cfg.influxdb.bucket_name
        org = cfg.influxdb.org

        client = InfluxClient(url, token, bucket, org, enable_compress=True)
        result_df = await client.query(query, deserializer)

        # 将查询结果转换为dict,并且进行排序
        result = {}
        for code, group in result_df.groupby("code"):
            bars = (
                group[return_cols]
                .sort_values("frame")
                .to_records(index=False)
                .astype(dtype)
            )
            result[code] = bars

        return result

    @classmethod
    async def batch_cache_bars(cls, frame_type: FrameType, bars: Dict[str, np.ndarray]):
        """缓存已收盘的分钟线和日线

        当缓存日线时，仅限于当日收盘后的第一次同步时调用。

        Args:
            frame_type: 帧类型
            bars: 行情数据，其key为股票代码，其value为dtype为`bars_dtype`的一维numpy数组。

        Raises:
            RedisError: 如果在执行过程中发生错误，则抛出以此异常为基类的各种异常，具体参考aioredis相关文档。
        """
        if frame_type == FrameType.DAY:
            await cls.batch_cache_unclosed_bars(frame_type, bars)
            return

        pl = cache.security.pipeline()
        for code, bar in bars.items():
            frame = TimeFrame.time2int(bar["frame"])
            val = [*bar]
            val[0] = frame
            pl.hset(f"bars:{frame_type.value}:{code}", frame, ",".join(map(str, val)))
        await pl.execute()

        cls._set_cached(bars[0]["frame"])

    @classmethod
    async def batch_cache_unclosed_bars(
        cls, frame_type: FrameType, bars: Dict[str, np.ndarray]
    ):
        """缓存未收盘的5、15、30、60分钟线及日线

        Args:
            frame_type: 帧类型
            bars: 行情数据，其key为股票代码，其value为dtype为`bars_dtype`的一维numpy数组。bars不能为None，或者empty。

        Raise:
            RedisError: 如果在执行过程中发生错误，则抛出以此异常为基类的各种异常，具体参考aioredis相关文档。
        """
        pl = cache.security.pipeline()
        key = f"bars:{frame_type.value}:unclosed"

        convert = (
            TimeFrame.time2int
            if frame_type in TimeFrame.minute_level_frames
            else TimeFrame.date2int
        )

        for code, bar in bars.items():
            val = [*bar]
            val[0] = convert(bar["frame"])  # 时间转换
            pl.hset(key, code, ",".join(map(str, val)))
        await pl.execute()

        cls._set_cached(bars[0]["frame"])

    @classmethod
    async def reset_cache(cls):
        """清除缓存的行情数据"""
        try:
            for ft in TimeFrame.minute_level_frames:
                await cache.security.delete(f"bars:{ft.value}:unclosed")
                keys = await cache.security.keys(f"bars:{ft.value}:*")
                if keys:
                    await cache.security.delete(*keys)
        finally:
            cls._is_cache_empty = True

    @classmethod
    def _get_cached_first_frame(cls, frame_type: FrameType) -> Frame:
        """获取各frame_type类型的缓存的第一个时间点。

        我们需要单独记录此信息，是因为未到每个帧的收盘时间，可能缓存中并不存在相应数据。

        Args:
            frame_type : [description]

        Returns:
            [description]
        """
        # todo: rename the func at _x
        if cls._is_cache_empty:
            return None

        return cls._cached_frames_start.get(frame_type, None)

    @classmethod
    def _set_cached(cls, frame: Frame):
        """当某帧数据被缓存时，记录其被缓存的状态。

        Args:
            frame : [description]
        """
        if cls._is_cache_empty:
            dt = arrow.get(frame).date()

            for ft in TimeFrame.minute_level_frames:
                frame = TimeFrame.first_min_frame(dt, ft)
                cls._cached_frames_start[ft] = frame
            cls._cached_frames_start[FrameType.DAY] = dt

            cls._is_cache_empty = False
        else:
            return

    @classmethod
    async def _get_cached_bars(
        cls, code: str, end: Frame, n: int, frame_type: FrameType, unclosed=True
    ) -> np.ndarray:
        """从缓存中获取指定代码的行情数据

        如果行情数据为日线以上级别，则最多只会返回一条数据（也可能没有）。如果行情数据为分钟级别数据，则一次返回当天已缓存的所有数据。

        本接口在如下场景下，性能不是最优的：
        如果cache中存在接近240根分钟线，取截止到9：35分的前5根K线，此实现也会取出全部k线，但只返回前5根。这样会引起不必要的网络通信及反串行化时间。

        args:
            code: the full qualified code of a security or index
            end: the end frame of the bars
            n: the number of bars to return
            frame_type: use this to decide which store to use
            unclosed: whether to return unclosed bars

        returns:
            元素类型为`coretypes.bars_dtype`的一维numpy数组。如果没有数据，则返回空ndarray。
        """
        dtype = bars_with_limit_dtype if frame_type == FrameType.DAY else bars_dtype

        ff = cls._get_cached_first_frame(frame_type)

        if ff is None or end < ff:
            return np.empty((0,), dtype=dtype)

        raw = []
        if frame_type in TimeFrame.day_level_frames:
            convert = TimeFrame.int2date
            if unclosed:
                key = f"bars:{frame_type.value}:unclosed"
                r1 = await cache.security.hget(key, code)
                if r1 is None:
                    return None

                raw.append(r1)
            else:
                assert (
                    False
                ), f"bad parameters: FrameType[{frame_type}] + unclosed[{unclosed}] will always yield no result."
        else:
            convert = TimeFrame.int2time
            key = f"bars:{FrameType.MIN1.value}:{code}"
            end_ = TimeFrame.floor(end, FrameType.MIN1)
            frames = map(str, TimeFrame.get_frames_by_count(end_, 240, FrameType.MIN1))
            r1 = await cache.security.hmget(key, *frames)
            raw.extend(r1)

            if unclosed:
                key = f"bars:{FrameType.MIN1.value}:unclosed"
                r2 = await cache.security.hget(key, code)
                if r2:
                    raw.append(r2)

        recs = []
        for raw_rec in raw:
            if raw_rec is None:
                continue
            f, o, h, l, c, v, m, fac = raw_rec.split(",")
            recs.append(
                (
                    convert(f),
                    float(o),
                    float(h),
                    float(l),
                    float(c),
                    float(v),
                    float(m),
                    float(fac),
                )
            )

        bars = np.array(recs, dtype=bars_dtype)
        if frame_type in TimeFrame.minute_level_frames and frame_type != FrameType.MIN1:
            bars = cls.resample(bars, from_frame=FrameType.MIN1, to_frame=frame_type)[
                -n:
            ]
        bars = bars[-n:]
        if bars[-1]["frame"] > end:
            # 避免取到未来数据
            bars = bars[:-1]

        return bars

    @classmethod
    async def cache_bars(cls, code: str, frame_type: FrameType, bars: np.ndarray):
        """将当期已收盘的行情数据缓存

        行情数据缓存在以`bars:{frame_type.value}:{code}`为key, {frame}为field的hashmap中。

        Args:
            code: the full qualified code of a security or index
            frame_type: frame type of the bars
            bars: the bars to cache, which is a numpy array of dtype `coretypes.bars_dtype`

        Raises:
            RedisError: if redis operation failed, see documentation of aioredis

        """
        today = bars[0]["frame"]
        # 转换时间为int
        convert = (
            TimeFrame.time2int
            if frame_type in TimeFrame.minute_level_frames
            else TimeFrame.date2int
        )

        key = f"bars:{frame_type.value}:{code}"
        pl = cache.security.pipeline()
        for bar in bars:
            val = [*bar]
            val[0] = convert(bar["frame"])
            pl.hset(key, val[0], ",".join(map(str, val)))

        await pl.execute()

        cls._set_cached(today)

    @classmethod
    async def cache_unclosed_bars(
        cls, code: str, frame_type: FrameType, bars: np.ndarray
    ):
        """将未结束的行情数据缓存

        未结束的行情数据缓存在以`bars:{frame_type.value}:unclosed`为key, {code}为field的hashmap中。

        尽管`bars`被声明为np.ndarray，但实际上应该只包含一个元素。

        Args:
            code: the full qualified code of a security or index
            frame_type: frame type of the bars
            bars: the bars to cache, which is a numpy array of dtype `coretypes.bars_dtype`

        Raises:
            RedisError: if redis operation failed, see documentation of aioredis

        """
        today = bars[0]["frame"]
        converter = (
            TimeFrame.time2int
            if frame_type in TimeFrame.minute_level_frames
            else TimeFrame.date2int
        )

        assert len(bars) == 1, "unclosed bars should only have one record"

        key = f"bars:{frame_type.value}:unclosed"
        pl = cache.security.pipeline()
        for bar in bars:
            val = [*bar]
            val[0] = converter(bar["frame"])
            pl.hset(key, code, ",".join(map(str, val)))

        await pl.execute()
        cls._set_cached(today)

    @classmethod
    async def persist_bars(cls, frame_type: FrameType, bars: pd.DataFrame):
        """将行情数据持久化

        bars数据应该属于同一个frame_type,并且列字段由`coretypes.bars_dtype` + ("code", "O")构成。

        Args:
            code: the full qualified code of a security or index
            frame_type: the frame type of the bars
            bars: the bars to be persisted

        Raises:

        """
        client = InfluxClient(
            cfg.influxdb.url,
            cfg.influxdb.token,
            cfg.influxdb.org,
            enable_compress=cfg.influxdb.enable_compress,
        )

        measurement = f"stock_bars_{frame_type.value}"

        await client.save(
            bars,
            measurement,
            tag_keys=["code"],
            time_key="frame",
        )

    @classmethod
    def resample(
        cls, bars: np.ndarray, from_frame: FrameType, to_frame: FrameType
    ) -> np.ndarray:
        """将原来为`from_frame`的行情数据转换为`to_frame`的行情数据

        如果`to_frame`为日线或者分钟级别线，则`from_frame`必须为分钟线；如果`to_frame`为周以上级别线，则`from_frame`必须为日线。其它级别之间的转换不支持。

        如果`from_frame`为1分钟线，则必须从9：31起。

        Args:
            bars (np.ndarray): [description]
            from_frame (FrameType): [description]
            to_frame (FrameType): [description]

        Returns:
            np.ndarray: [description]
        """
        if from_frame == FrameType.MIN1:
            return cls._resample_from_min1(bars, to_frame)
        elif from_frame == FrameType.DAY:  # pragma: no cover
            return cls._resample_from_day(bars, to_frame)
        else:  # pragma: no cover
            raise TypeError(f"unsupported from_frame: {from_frame}")

    @classmethod
    def _resample_from_min1(cls, bars: np.ndarray, to_frame: FrameType) -> np.ndarray:
        """将`bars`从1分钟线转换为`to_frame`的行情数据

        重采样后的数据只包含frame, open, high, low, close, volume, amount, factor，无论传入数据是否还有别的字段，它们都将被丢弃。

        resampling 240根分钟线到5分钟大约需要100微秒。

        TODO： 如果`bars`中包含nan怎么处理？
        """
        if bars[0]["frame"].minute != 31:
            raise ValueError("resampling from 1min must start from 9:31")

        bins_len = {
            FrameType.MIN5: 5,
            FrameType.MIN15: 15,
            FrameType.MIN30: 30,
            FrameType.MIN60: 60,
        }[to_frame]

        bins = len(bars) // bins_len
        npart1 = bins * bins_len

        part1 = bars[:npart1].reshape((-1, bins_len))
        part2 = bars[npart1:]

        open_pos = np.arange(bins) * bins_len
        close_pos = np.arange(1, bins + 1) * bins_len - 1
        if len(bars) > bins_len * bins:
            close_pos = np.append(close_pos, len(bars) - 1)
            resampled = np.empty((bins + 1,), dtype=bars_dtype)
        else:
            resampled = np.empty((bins,), dtype=bars_dtype)

        resampled[:bins]["open"] = bars[open_pos]["open"]

        resampled[:bins]["high"] = np.max(part1["high"], axis=1)
        resampled[:bins]["low"] = np.min(part1["low"], axis=1)

        resampled[:bins]["volume"] = np.sum(part1["volume"], axis=1)
        resampled[:bins]["amount"] = np.sum(part1["amount"], axis=1)

        if len(part2):
            resampled[-1]["open"] = part2["open"][0]
            resampled[-1]["high"] = np.max(part2["high"])
            resampled[-1]["low"] = np.min(part2["low"])

            resampled[-1]["volume"] = np.sum(part2["volume"])
            resampled[-1]["amount"] = np.sum(part2["amount"])

        cols = ["frame", "close", "factor"]
        resampled[cols] = bars[close_pos][cols]

        return resampled

    @classmethod
    def _resample_from_day(cls, bars: np.ndarray, to_frame: FrameType) -> np.ndarray:
        """将`bars`从日线转换成`to_frame`的行情数据

        Args:
            bars (np.ndarray): [description]
            to_frame (FrameType): [description]

        Returns:
            转换后的行情数据
        """
        # todo: 需要实现未收盘的周期合成。
        raise NotImplementedError

    @classmethod
    async def _get_persisted_trade_limit_prices(
        cls, code: str, begin: Frame, end: Frame
    ) -> np.ndarray:
        """从influxdb中获取个股的涨跌停价。

        涨跌停价只有日线数据才有，因此，FrameType固定为FrameType.DAY

        Args:
            code : [description]
            begin : [description]
            end : [description]

        Returns:
            dtype为[('code', 'O'), ('frame', 'datetime64[D]'), ('high_limit', 'f8'), ('low_limit', 'f8')]的numpy数组
        """
        url, token, org, bucket = (
            cfg.influxdb.url,
            cfg.influxdb.token,
            cfg.influxdb.org,
            cfg.influxdb.bucket_name,
        )
        client = InfluxClient(url, token, bucket, org)
        measurement = f"stock_bars_{FrameType.DAY.value}"
        query = (
            Flux()
            .bucket(bucket)
            .measurement(measurement)
            .range(begin, end)
            .fields(["high_limit", "low_limit", "close"])
            .tags({"code": code})
        )

        dtype = [("frame", "datetime64[D]"), ("high_limit", "f8"), ("low_limit", "f8")]
        ds = NumpyDeserializer(
            dtype, sort_values="frame", use_cols=["_time", "high_limit", "low_limit"]
        )

        result = await client.query(query, ds)
        return result

    @classmethod
    async def get_limits_in_range(
        cls, code: str, begin: datetime.date, end: datetime.date
    ) -> np.array:
        """获取股票的涨跌停价"""
        now = datetime.datetime.now().date()
        end = min(now, end)
        assert begin < now, "begin time should NOT be great than now"
        assert begin < end, "begin time should NOT be great than end time"

        part1 = await cls._get_persisted_trade_limit_prices(code, begin, end)

        if end == now:
            pl = cache._security_.pipeline()
            pl.hget("high_low_limit", f"{code}.high_limit")
            pl.hget("high_low_limit", f"{code}.low_limit")

            hl, ll = await pl.execute()
            if hl or ll:
                part2 = np.array([(end, hl, ll)], dtype=part1.dtype)
                return np.concatenate((part1, part2))

            else:
                return part1
