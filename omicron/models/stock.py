import asyncio
import datetime
import logging
import re
from typing import Dict, Iterable, List, Union

import arrow
import cfg4py
import ciso8601
import numpy as np
import pandas as pd
from coretypes import Frame, FrameType, SecurityType, bars_cols, bars_dtype

from omicron.core.constants import TRADE_PRICE_LIMITS
from omicron.core.errors import BadParameterError, DataNotReadyError
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


def ciso8601_parse_date(x):
    return ciso8601.parse_datetime(x).date()


def ciso8601_parse_naive(x):
    return ciso8601.parse_datetime_as_naive(x)


class Stock:
    """ "
    Stock对象用于归集某支证券（股票和指数，不包括其它投资品种）的相关信息，比如行情数据（OHLC等）、市值数据、所属概念分类等。
    """

    _stocks = None
    stock_info_dtype = [
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
                [tuple(x.split(",")) for x in secs], dtype=cls.stock_info_dtype
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
        result = np.array(result, dtype=cls.stock_info_dtype)
        return result["code"].tolist()

    @classmethod
    def choose_listed(cls, dt: datetime.date, types: List[str] = ["stock", "index"]):
        cond = np.array([False] * len(cls._stocks))

        for type_ in types:
            cond |= cls._stocks["type"] == type_
        result = cls._stocks[cond]
        result = result[result["end"] > dt]
        result = result[result["ipo"] <= dt]

        result = np.array(result, dtype=cls.stock_info_dtype)
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
        # todo: 这里可以优化
        if bars.size == 0:
            return bars

        last = bars[-1]["factor"]
        for field in ["open", "high", "low", "close", "volume"]:
            bars[field] = bars[field] * (bars["factor"] / last)

        return bars

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
        closed_end = TimeFrame.floor(end, frame_type)
        n = TimeFrame.count_frames(begin, closed_end, frame_type)
        if closed_end < end and unclosed:
            n += 1

        return await cls.batch_get_bars(codes, n, frame_type, end, fq, unclosed)

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
        closed_end = TimeFrame.floor(end, frame_type)
        n = TimeFrame.count_frames(start, closed_end, frame_type)

        if closed_end != end and unclosed:
            n += 1

        return await cls.get_bars(code, n, frame_type, end, fq, unclosed)

    @classmethod
    async def batch_get_bars(
        cls,
        codes: Iterable[str],
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
            codes: 代码列表
            n: 返回记录数
            frame_type: 帧类型
            end: 结束时间。如果未指明，则取当前时间。 Defaults to None.
            fq: 是否进行复权，如果是，则进行前复权。Defaults to True.
            unclosed: 是否包含最新一期未收盘数据. Defaults to True.

        Returns:
            返回一个字典，其key为证券代码，其value为dtype为`bars_dtype`的一维numpy数组。
        """
        end = end or arrow.now().floor("minute").naive

        if frame_type in TimeFrame.minute_level_frames:
            part2_start = TimeFrame.first_min_frame(end, frame_type)
            part2_closed = TimeFrame.floor(end, frame_type)

            n2 = TimeFrame.count_frames(part2_start, part2_closed, frame_type)
            if end != part2_closed and unclosed:
                # 如果end指定了非帧对齐时间，且不是属于最后一帧，这里的逻辑可能有问题。但这种场景不应该出现。
                n2 += 1

            part2 = await cls._batch_get_cached_bars(
                codes, end, n2, frame_type, unclosed
            )
            n2 = len(max(part2.values(), key=len))
        elif frame_type == FrameType.DAY and unclosed:
            return await cls._batch_get_cached_bars(codes, end, 1, frame_type, unclosed)
        else:
            part2 = {}
            part2_closed = end
            n2 = 0

        if n2 == n:
            if fq:
                return {code: cls.qfq(bars) for code, bars in part2.items()}
            else:
                return part2

        # part2可能部分品种有数据，部分没有。因此，part1的长度仍设置为n，且从end开始起向前取数据。这样，两部分加起来的结果可能大于n,需要在返回前进行截断。
        begin = TimeFrame.shift(part2_closed, -n, frame_type)
        part1 = await cls._batch_get_persisted_bars(
            codes, frame_type, begin=begin, n=n, end=end
        )

        result = {}
        for code in codes:
            part1_bars = part1.get(code, np.empty((0,), dtype=bars_dtype))
            part2_bars = part2.get(code, np.empty((0,), dtype=bars_dtype))

            bars = np.concatenate([part1_bars, part2_bars])[-n:]
            if fq:
                bars = cls.qfq(bars)
            result[code] = bars

        return result

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
        end = end or arrow.now().floor("minute").naive
        close_end = TimeFrame.floor(end, frame_type)

        part2 = await cls._get_cached_bars(code, end, n, frame_type, unclosed)

        if part2.size == n:
            part1 = np.empty((0,), dtype=bars_dtype)
        elif part2.size > 0:
            n2 = part2.size
            n1 = n - n2

            if n1 > 0:
                # 可能多查询一个bar，但返回前通过limit进行了限制
                part1_end = TimeFrame.shift(part2[0]["frame"], -1, frame_type)
                part1_begin = TimeFrame.shift(part1_end, -n1 + 1, frame_type)
                part1 = await cls._get_persisted_bars(
                    code, begin=part1_begin, end=part1_end, n=n1, frame_type=frame_type
                )
        else:  # part2 is empty
            n1 = n
            part1_end = close_end
            part1_begin = TimeFrame.shift(part1_end, -n1 + 1, frame_type)
            part1 = await cls._get_persisted_bars(
                code, begin=part1_begin, end=part1_end, n=n1, frame_type=frame_type
            )

        bars = np.concatenate([part1, part2])

        if fq:
            bars = cls.qfq(bars)

        return bars

    @classmethod
    async def _get_persisted_bars(
        cls,
        code: str,
        frame_type: FrameType,
        begin: Frame,
        end: Frame = None,
        n: int = None,
    ) -> np.array:
        """从influxdb中获取数据

        如果`end`未指定，则取当前时间。当`n`指定时，将只返回在`[begin, end]`范围内的最多`n`条数据,且为递增排序的最后`n`条数据（即最接近于`end`的数据）。

        返回的数据按`frame`进行升序排列。

        Args:
            code (str): 证券代码
            frame_type: the frame_type to query
            n (int): 返回结果数量
            end (Frame): [description]

        Returns:
            返回dtype为`bars_dtype`的numpy数组
        """
        assert begin is not None, "must specify `begin`"
        # check is needed since tags accept List as well
        assert isinstance(code, str), "`code` must be a string"

        end = end or arrow.now().naive

        keep_cols = ["_time"] + bars_cols[1:]

        measurement = cls._measurement_name(frame_type)
        flux = (
            Flux(no_sys_cols=False)
            .bucket(cfg.influxdb.bucket_name)
            .range(begin, end)
            .measurement(measurement)
            .keep(keep_cols)
            .tags({"code": code})
        )

        if n is not None:
            flux.latest(n)
        else:
            flux.sort("_time")

        if frame_type in TimeFrame.day_level_frames:
            _time_converter = ciso8601_parse_date
        else:
            _time_converter = ciso8601_parse_naive
        serializer = NumpyDeserializer(
            bars_dtype,
            # sort at server side
            # sort_values="frame",
            encoding="utf-8",
            skip_rows=1,
            use_cols=keep_cols,
            converters={
                "_time": _time_converter,
            },
        )

        url = cfg.influxdb.url
        token = cfg.influxdb.token
        bucket = cfg.influxdb.bucket_name
        org = cfg.influxdb.org

        client = InfluxClient(url, token, bucket, org)
        return await client.query(flux, serializer)

    @classmethod
    async def _batch_get_persisted_bars(
        cls,
        codes: List[str],
        frame_type: FrameType,
        begin: Frame,
        n: int = None,
        end: Frame = None,
    ) -> Dict[str, np.array]:
        """从持久化存储中获取`codes`指定的一批股票在时间范围内的数据。

        如果`end`未指定，则取当前时间。当`n`指定时，将只返回在`[begin, end]`范围内的最多`n`条数据,且为递增排序的最后`n`条数据（即最接近于`end`的数据）。

        返回的数据按`frame`进行升序排列。

        如果`codes`为None,则返回指定时间段内所有股票的数据。

        注意，返回的数据有可能不是等长的。

        Args:
            codes : 证券代码列表
            frame_type : the frame_type to query
            begin : begin timestamp of returned results
            n : 返回结果数量
            end : end timestamp of returned results

        Returns:
            以`code`为key, 行情数据（dtype为bars_dtype的numpy数组）为value的字典
        """
        if n is None:
            assert begin is not None, "must specify `begin` or `n`"
        else:
            begin = begin or Flux.EPOCH_START

        end = end or arrow.now().naive

        # influxdb的查询结果格式类似于CSV，其列顺序为_, result_alias, table_seq, _time, tags, fields,其中tags和fields都是升序排列
        return_cols = bars_cols
        keep_cols = bars_cols + ["code"]
        names = ["_", "result", "table", "frame", "code"]

        # influxdb will return fields in the order of name ascending parallel
        names.extend(sorted(bars_cols[1:]))

        measurement = cls._measurement_name(frame_type)
        flux = (
            Flux(no_sys_cols=False)
            .bucket(cfg.influxdb.bucket_name)
            .range(begin, end)
            .measurement(measurement)
            .keep(keep_cols)
        )

        if len(codes) > 0:
            flux.tags({"code": codes})
        if n is not None:
            flux.latest(n)
        else:
            flux.sort("_time")

        deserializer = DataframeDeserializer(
            names=names, usecols=keep_cols, encoding="utf-8", time_col="frame"
        )

        client = cls._get_influx_client()
        result_df = await client.query(flux, deserializer)

        # 将查询结果转换为dict,并且进行排序
        result = {}
        if frame_type in TimeFrame.day_level_frames:
            convertor = cls._pd_timestamp_to_date
        else:
            convertor = cls._pd_timestamp_to_datetime

        for code, group in result_df.groupby("code"):
            df = group[return_cols].sort_values("frame")
            bars = df.to_records(index=False).astype(bars_dtype)
            bars["frame"] = [convertor(x) for x in df["frame"]]
            result[code] = bars

        return result

    @classmethod
    def _pd_timestamp_to_date(cls, ts: pd.Timestamp) -> datetime.date:
        """将pd.Timestamp转换为date"""
        return ts.to_pydatetime().date()

    @classmethod
    def _pd_timestamp_to_datetime(cls, ts: pd.Timestamp) -> datetime.datetime:
        return ts.to_pydatetime()

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
        for code, bars in bars.items():
            key = f"bars:{frame_type.value}:{code}"
            for bar in bars:
                frame = TimeFrame.time2int(bar["frame"])
                val = [*bar]
                val[0] = frame
                pl.hset(key, frame, ",".join(map(str, val)))
        await pl.execute()

    @classmethod
    async def batch_cache_unclosed_bars(
        cls, frame_type: FrameType, bars: Dict[str, np.ndarray]
    ):  # pragma: no cover
        """缓存未收盘的5、15、30、60分钟线及日线

        Note:
            同[cache_unclosed_bars][omicron.models.stock.Stock.cache_unclosed_bars]一样，考虑到resample的性能较高，所以当前并没有将未结束的行情数据缓存 -- 这需要有一个程序以每分钟一次的频率，对全市场数据进行resample并缓存。

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
            val = [*bar[0]]
            val[0] = convert(bar["frame"][0])  # 时间转换
            pl.hset(key, code, ",".join(map(str, val)))

        await pl.execute()

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
    def _deserialize_cached_bars(cls, raw: List[str], ft: FrameType) -> np.ndarray:
        """从redis中反序列化缓存的数据

        Args:
            raw: redis中的缓存数据
            ft: 帧类型

        Returns:
            [description]
        """
        if ft in TimeFrame.minute_level_frames:
            convert = TimeFrame.int2time
        else:
            convert = TimeFrame.int2date
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

        return np.array(recs, dtype=bars_dtype)

    @classmethod
    async def _batch_get_cached_bars(
        cls, codes: List[str], end: Frame, n: int, frame_type: FrameType, unclosed=True
    ) -> Dict[str, np.ndarray]:
        """批量获取在cache中截止`end`的`n`个bars。

        Args:
            codes: 证券代码列表
            end : 截止时间
            n : 返回记录条数
            frame_type : 时间帧类型
            unclosed : 是否包含未结束数据

        Raises:

        Returns:
            key为code, value为行情数据的字典
        """
        tasks = [cls._get_cached_bars(c, end, n, frame_type, unclosed) for c in codes]
        results = await asyncio.gather(*tasks)

        return {c: r for c, r in zip(codes, results)}

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
        if frame_type == FrameType.DAY and unclosed:
            # only available after 15:00, 如果在盘中，则都从1m数据resample
            raw = await cache.security.hmget(f"bars:{frame_type.value}:unclosed", code)
            bars = cls._deserialize_cached_bars(raw, FrameType.DAY)
            if len(bars) != 0:
                return bars

        ff = TimeFrame.combine_time(end, 9, 31)
        end = TimeFrame.floor(end, FrameType.MIN1)

        if ff is None or end < ff:  # cache 中还没有分钟线数据，或者结束时间早于cache
            return np.empty((0,), dtype=bars_dtype)

        if (
            frame_type in TimeFrame.minute_level_frames
            or frame_type == FrameType.DAY
            and unclosed
        ):
            # 取1分钟数据，再进行resample
            key = f"bars:{FrameType.MIN1.value}:{code}"
            start = TimeFrame.combine_time(end, 9, 31)
            frames = map(str, TimeFrame.get_frames(start, end, FrameType.MIN1))
            r1 = await cache.security.hmget(key, *frames)

            min_bars = cls._deserialize_cached_bars(r1, FrameType.MIN1)

            if min_bars.size == 0:
                return min_bars

            if frame_type == FrameType.MIN1:
                return min_bars[-n:]

            bars = cls.resample(
                min_bars, from_frame=FrameType.MIN1, to_frame=frame_type
            )

            if not unclosed:
                last_frame = bars[-1]["frame"]
                if TimeFrame.floor(last_frame, frame_type) != last_frame:
                    bars = bars[:-1]

            return bars[-n:]
        else:
            logger.warning("no closed data for %s in cache", frame_type)
            return np.empty((0,), bars_dtype)

    @classmethod
    async def cache_bars(cls, code: str, frame_type: FrameType, bars: np.ndarray):
        """将当期已收盘的行情数据缓存

        Note:
            当前只缓存1分钟数据。其它分钟数据，都在调用时，通过resample临时合成。

        行情数据缓存在以`bars:{frame_type.value}:{code}`为key, {frame}为field的hashmap中。

        Args:
            code: the full qualified code of a security or index
            frame_type: frame type of the bars
            bars: the bars to cache, which is a numpy array of dtype `coretypes.bars_dtype`

        Raises:
            RedisError: if redis operation failed, see documentation of aioredis

        """
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

    @classmethod
    async def cache_unclosed_bars(
        cls, code: str, frame_type: FrameType, bars: np.ndarray
    ):  # pragma: no cover
        """将未结束的行情数据缓存

        Note:
            考虑到resample的性能高，所以当前并没有将未结束的行情数据缓存 -- 这需要有一个程序以每分钟一次的频率，对全市场数据进行resample并缓存。因此这个函数当前并没有被使用。

        未结束的行情数据缓存在以`bars:{frame_type.value}:unclosed`为key, {code}为field的hashmap中。

        尽管`bars`被声明为np.ndarray，但实际上应该只包含一个元素。

        Args:
            code: the full qualified code of a security or index
            frame_type: frame type of the bars
            bars: the bars to cache, which is a numpy array of dtype `coretypes.bars_dtype`

        Raises:
            RedisError: if redis operation failed, see documentation of aioredis

        """
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

    @classmethod
    def _get_influx_client(cls):
        client = InfluxClient(
            cfg.influxdb.url,
            cfg.influxdb.token,
            cfg.influxdb.bucket_name,
            cfg.influxdb.org,
            enable_compress=cfg.influxdb.enable_compress,
        )

        return client

    @classmethod
    async def persist_bars(
        cls,
        frame_type: FrameType,
        bars: Union[Dict[str, np.ndarray], np.ndarray, pd.DataFrame],
    ):
        """将行情数据持久化

        如果`bars`类型为Dict,则key为`code`，value为`bars`。如果其类型为np.ndarray或者pd.DataFrame，则`bars`各列字段应该为`coretypes.bars_dtype` + ("code", "O")构成。

        Args:
            frame_type: the frame type of the bars
            bars: the bars to be persisted

        Raises:
            InfluxDBWriteError: if influxdb write failed
        """
        client = cls._get_influx_client()

        measurement = cls._measurement_name(frame_type)

        if isinstance(bars, dict):
            for code, value in bars.items():
                await client.save(
                    value, measurement, global_tags={"code": code}, time_key="frame"
                )
        else:
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
    def _measurement_name(cls, frame_type):
        return f"stock_bars_{frame_type.value}"

    @classmethod
    def _resample_from_min1(cls, bars: np.ndarray, to_frame: FrameType) -> np.ndarray:
        """将`bars`从1分钟线转换为`to_frame`的行情数据

        重采样后的数据只包含frame, open, high, low, close, volume, amount, factor，无论传入数据是否还有别的字段，它们都将被丢弃。

        resampling 240根分钟线到5分钟大约需要100微秒。

        TODO： 如果`bars`中包含nan怎么处理？
        """
        if bars[0]["frame"].minute != 31:
            raise ValueError("resampling from 1min must start from 9:31")

        if to_frame not in (
            FrameType.MIN5,
            FrameType.MIN15,
            FrameType.MIN30,
            FrameType.MIN60,
        ):
            raise ValueError(f"unsupported to_frame: {to_frame}")

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
    async def _get_persisted_trade_price_limits(
        cls, code: str, begin: Frame, end: Frame
    ) -> np.ndarray:
        """从influxdb中获取个股在[begin, end]之间的涨跌停价。

        涨跌停价只有日线数据才有，因此，FrameType固定为FrameType.DAY

        Args:
            code : 个股代码
            begin : 开始日期
            end : 结束日期

        Returns:
            dtype为[('frame', 'O'), ('high_limit', 'f8'), ('low_limit', 'f8')]的numpy数组
        """
        client = cls._get_influx_client()
        measurement = cls._measurement_name(FrameType.DAY)
        flux = (
            Flux(no_sys_cols=False)
            .bucket(client._bucket)
            .measurement(measurement)
            .range(begin, end)
            .fields(["high_limit", "low_limit"])
            .tags({"code": code})
            .keep(["_time", "high_limit", "low_limit"])
            .sort("_time")
        )

        dtype = [("frame", "O"), ("high_limit", "f8"), ("low_limit", "f8")]
        ds = NumpyDeserializer(
            dtype,
            use_cols=["_time", "high_limit", "low_limit"],
            converters={
                "_time": lambda x: ciso8601.parse_datetime(x).date(),
            },
            # since we ask parse date in convertors, so we have to disable parse_date
            parse_date=None,
        )

        result = await client.query(flux, ds)
        return result

    @classmethod
    async def get_trade_price_limits(
        cls, code: str, begin: datetime.date, end: datetime.date
    ) -> np.array:
        """获取股票在`[begin, end]`期间的涨跌停价

        Args:
            code: 股票代码
            begin: 开始日期,必须指定为交易日
            end: 结束日期,必须指定为交易日

        Returns:
            dtype为[('frame', 'O'), ('high_limit', 'f8'), ('low_limit', 'f8')]的numpy数组

        """
        now = TimeFrame.day_shift(arrow.now(), 0)
        end = min(now, end)
        assert begin <= end, "begin time should NOT be great than end time"

        part1 = await cls._get_persisted_trade_price_limits(code, begin, end)

        if end == now:  # 当天的数据在缓存中
            pl = cache._security_.pipeline()
            pl.hget(TRADE_PRICE_LIMITS, f"{code}.high_limit")
            pl.hget(TRADE_PRICE_LIMITS, f"{code}.low_limit")

            hl, ll = await pl.execute()
            if hl or ll:
                part2 = np.array([(end, hl, ll)], dtype=part1.dtype)
                return np.concatenate((part1, part2))

        else:
            return part1

    @classmethod
    async def save_trade_price_limits(cls, price_limits: np.ndarray, to_cache: bool):
        """保存涨跌停价

        Args:
            price_limits: numpy structured array of dtype [('frame', 'O'), ('code', 'O'), ('high_limit', 'f4'), ('low_limit', 'f4')]
            to_cache: 是保存到缓存中，还是保存到持久化存储中
        """
        if len(price_limits) == 0:
            return

        if to_cache:
            pl = cache._security_.pipeline()
            for row in price_limits:
                # .item convert np.float64 to python float
                pl.hset(
                    TRADE_PRICE_LIMITS,
                    f"{row['code']}.high_limit",
                    row["high_limit"].item(),
                )
                pl.hset(
                    TRADE_PRICE_LIMITS,
                    f"{row['code']}.low_limit",
                    row["low_limit"].item(),
                )
            await pl.execute()

        else:
            # to influxdb
            client = cls._get_influx_client()
            await client.save(
                price_limits,
                cls._measurement_name(FrameType.DAY),
                tag_keys="code",
                time_key="frame",
            )
