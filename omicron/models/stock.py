import asyncio
import datetime
import logging
import re
from typing import Dict, Iterable, List, Tuple, Union

import arrow
import cfg4py
import ciso8601
import numpy as np
import pandas as pd
from coretypes import Frame, FrameType, SecurityType, bars_cols, bars_dtype

from omicron import tf
from omicron.core.constants import TRADE_PRICE_LIMITS
from omicron.core.errors import BadParameterError, DataNotReadyError
from omicron.dal import cache
from omicron.dal.influx.flux import Flux
from omicron.dal.influx.influxclient import InfluxClient
from omicron.dal.influx.serialize import DataframeDeserializer, NumpyDeserializer
from omicron.extensions.np import array_math_round, array_price_equal
from omicron.models.security import Security, convert_nptime_to_datetime

logger = logging.getLogger(__name__)
cfg = cfg4py.get_instance()


def ciso8601_parse_date(x):
    return ciso8601.parse_datetime(x).date()


def ciso8601_parse_naive(x):
    return ciso8601.parse_datetime_as_naive(x)


class Stock(Security):
    """ "
    Stock对象用于归集某支证券（股票和指数，不包括其它投资品种）的相关信息，比如行情数据（OHLC等）、市值数据、所属概念分类等。
    """

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
        self._stock = self.get_stock(code)
        assert self._stock, "系统中不存在该code"
        (
            _,
            self._display_name,
            self._name,
            ipo,
            end,
            _type,
        ) = self._stock
        self._start_date = convert_nptime_to_datetime(ipo).date()
        self._end_date = convert_nptime_to_datetime(end).date()
        self._type = SecurityType(_type)

    @classmethod
    def choose_listed(cls, dt: datetime.date, types: List[str] = ["stock", "index"]):
        cond = np.array([False] * len(cls._stocks))
        dt = datetime.datetime.combine(dt, datetime.time())

        for type_ in types:
            cond |= cls._stocks["type"] == type_
        result = cls._stocks[cond]
        result = result[result["end"] > dt]
        result = result[result["ipo"] <= dt]
        # result = np.array(result, dtype=cls.stock_info_dtype)
        return result["code"].tolist()

    @classmethod
    def fuzzy_match(cls, query: str) -> Dict[str, Tuple]:
        """对股票/指数进行模糊匹配查找

        query可以是股票/指数代码，也可以是字母（按name查找），也可以是汉字（按显示名查找）

        Args:
            query (str): 查询字符串

        Returns:
            Dict[str, Tuple]: 查询结果，其中Tuple为(code, display_name, name, start, end, type)
        """
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
                if sec["alias"].find(query) != -1
            }

    @classmethod
    def fuzzy_match_ex(cls, query: str):
        """对股票/指数进行模糊匹配查找"""
        query = query.upper()
        if re.match(r"\d+", query):
            return {
                sec["code"]: sec.tolist()
                for sec in cls._stocks
                if sec["code"].find(query) != -1
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
                if sec["alias"].find(query) != -1
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
        return tf.count_day_frames(ipo_day, arrow.now().date())

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
        closed_end = tf.floor(end, frame_type)
        n = tf.count_frames(begin, closed_end, frame_type)
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
        closed_end = tf.floor(end, frame_type)
        n = tf.count_frames(start, closed_end, frame_type)

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
        unclosed: bool = True,
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
        tasks = [cls.get_bars(c, n, frame_type, end, fq, unclosed) for c in codes]
        results = await asyncio.gather(*tasks)

        return {c: r for c, r in zip(codes, results)}

    @classmethod
    async def _get_day_level_bars(
        cls, code: str, n: int, frame_type: FrameType, end: datetime.date
    ) -> np.ndarray:
        """获取日线及日线以上级别的行情数据

        日线级别以上的数据，如果unclosed为False，则只需要从持久化存储中获取；如果为True，则需要先获取日线、再重采样为更高级别的数据。

        对于日线数据，因为次日凌晨才写入持久化储存，因此在此之前都认为是unclosed

        Args:
            code (str): 证券代码
            n (int): 记录数
            frame_type (FrameType): 帧类型
            end (Frame): 截止时间,如果未指明，则取当前时间
            fq (bool, optional): [description]. Defaults to True.
            unclosed (bool, optional): 是否包含最新未收盘的数据？ Defaults to True.

        Returns:
            一维numpy array, dtype为bars_dtype。如果数据不存在，则返回空数组。
        """
        if n <= 0:
            raise BadParameterError(f"n must be positive, got {n}")

        begin = tf.shift(end, -n + 1, frame_type)
        part1 = await cls._get_persisted_bars(code, frame_type, begin, end, n)

        if part1.size == n or (part1.size > 0 and part1[-1]["frame"] == end):
            return part1

        # need get data from cache
        cached_day_bar = await cls._get_cached_day_bar(code)

        if frame_type == FrameType.DAY:
            return np.concatenate([part1, cached_day_bar])[-n:]

        # 其它周期的数据，需要从日线级别来拼。注意，此时可能某个周期的数据已收盘，但还未到周期校准同步的时间，因此即使是取已收盘数据，也需要从日线来拼。
        day_start = tf.day_shift(tf.floor(end, frame_type), 1)

        m = tf.count_frames(day_start, end, FrameType.DAY)
        if m <= 0:
            persisted_day_bars = np.array([], dtype=bars_dtype)
        else:
            persisted_day_bars = await cls._get_persisted_bars(
                code, FrameType.DAY, day_start, end, m
            )

        day_bars = np.concatenate([persisted_day_bars, cached_day_bar])
        if day_bars.size >= 2 and day_bars[-1]["frame"] == day_bars[-2]["frame"]:
            # overlapped, should never happen
            logger.warning(
                "both persisted db and cache has bars for %s at %s", code, end
            )
            day_bars = day_bars[:-1]

        if day_bars.size > 0:
            unsynced = cls._resample_from_day(day_bars, frame_type)
        else:
            unsynced = np.array([], dtype=bars_dtype)

        bars = np.concatenate([part1, unsynced])
        return bars[-n:]

    @classmethod
    async def _get_cached_day_bar(cls, code: str) -> np.ndarray:
        """获取最新日线数据

        Args:
            code : 证券代码（聚宽格式）

        Returns:
            dtype为bars_dtype的一维numpy数组
        """
        key = "bars:1d:unclosed"

        raw = await cache.security.hget(key, code)
        if raw is not None:
            return cls._deserialize_cached_bars([raw], FrameType.DAY)

        mbars = await cls._get_cached_bars(code, arrow.now().naive, 240, FrameType.MIN1)
        if len(mbars) == 0:
            return np.array([], dtype=bars_dtype)

        return cls.resample(mbars, FrameType.MIN1, FrameType.DAY)

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
        try:
            if frame_type in tf.day_level_frames:
                end = arrow.get(end).date() if end else arrow.now().date()
                bars = await cls._get_day_level_bars(code, n, frame_type, end)

                if not unclosed and not tf.is_bar_closed(bars[-1]["frame"], frame_type):
                    bars = bars[:-1]

                if fq:
                    bars = cls.qfq(bars)
                return bars

            # frame_type is minute level
            end = end or arrow.now().floor("minute").naive
            close_end = tf.floor(end, frame_type)

            part2 = await cls._get_cached_bars(code, end, n, frame_type)

            if not unclosed and not tf.is_bar_closed(part2[-1]["frame"], frame_type):
                part2 = part2[:-1]

            if part2.size == n:
                part1 = np.array([], dtype=bars_dtype)
            elif part2.size == 0:
                n1 = n
                part1_end = close_end
                part1_begin = tf.shift(part1_end, -n1 + 1, frame_type)
                part1 = await cls._get_persisted_bars(
                    code, begin=part1_begin, end=part1_end, n=n1, frame_type=frame_type
                )
            else:
                n1 = n - part2.size

                # 可能多查询一个bar，但返回前通过limit进行了限制
                part1_end = tf.shift(part2[0]["frame"], -1, frame_type)
                part1_begin = tf.shift(part1_end, -n1 + 1, frame_type)
                part1 = await cls._get_persisted_bars(
                    code, begin=part1_begin, end=part1_end, n=n1, frame_type=frame_type
                )

            bars = np.concatenate([part1, part2])

            if fq:
                bars = cls.qfq(bars)

            return bars
        except Exception as e:
            logger.exception(e)
            logger.warning(
                "failed to get bars for %s, %s, %s, %s", code, n, frame_type, end
            )
            raise

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
            Flux()
            .bucket(cfg.influxdb.bucket_name)
            .range(begin, end)
            .measurement(measurement)
            .fields(keep_cols)
            .tags({"code": code})
        )

        if n is not None:
            flux.latest(n)
        else:
            flux.sort("_time")

        if frame_type in tf.day_level_frames:
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
            parse_date=None,
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
            Flux()
            .bucket(cfg.influxdb.bucket_name)
            .range(begin, end)
            .measurement(measurement)
            .fields(keep_cols)
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
        if frame_type in tf.day_level_frames:
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
                frame = tf.time2int(bar["frame"])
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

        convert = tf.time2int if frame_type in tf.minute_level_frames else tf.date2int

        for code, bar in bars.items():
            val = [*bar[0]]
            val[0] = convert(bar["frame"][0])  # 时间转换
            pl.hset(key, code, ",".join(map(str, val)))

        await pl.execute()

    @classmethod
    async def reset_cache(cls):
        """清除缓存的行情数据"""
        try:
            await cache.security.delete("bars:1d:unclosed")
            for ft in tf.minute_level_frames:
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
        if ft in tf.minute_level_frames:
            convert = tf.int2time
        else:
            convert = tf.int2date
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
        cls, codes: List[str], end: Frame, n: int, frame_type: FrameType
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
        tasks = [cls._get_cached_bars(c, end, n, frame_type) for c in codes]
        results = await asyncio.gather(*tasks)

        return {c: r for c, r in zip(codes, results)}

    @classmethod
    async def _get_cached_bars(
        cls, code: str, end: Frame, n: int, frame_type: FrameType
    ) -> np.ndarray:
        """从缓存中获取指定代码的行情数据

        本接口在如下场景下，性能不是最优的：
        如果cache中存在接近240根分钟线，取截止到9：35分的前5根K线，此实现也会取出全部k线，但只返回前5根。这样会引起不必要的网络通信及反串行化时间。

        args:
            code: the full qualified code of a security or index
            end: the end frame of the bars
            n: the number of bars to return
            frame_type: 帧类型。只能为分钟类型和日线类型。

        returns:
            元素类型为`coretypes.bars_dtype`的一维numpy数组。如果没有数据，则返回空ndarray。
        """
        # 只有日线及以下级别的数据，才能直接从缓存中获取
        assert frame_type in tf.minute_level_frames or frame_type == FrameType.DAY

        # 如果传入的end为日期型（如果是日线，则常常会传入datetime.date类型），则转换为datetime型
        if getattr(end, "date", None) is None:
            end = tf.combine_time(end, 15)

        # ff = tf.day_shift(arrow.now().date(), 0)
        start = tf.combine_time(arrow.now().date(), 9, 31)
        end = tf.floor(end, FrameType.MIN1)

        if end < start:
            return np.array([], dtype=bars_dtype)

        # 取1分钟数据，再进行resample
        key = f"bars:{FrameType.MIN1.value}:{code}"

        frames = map(str, tf.get_frames(start, end, FrameType.MIN1))
        r1 = await cache.security.hmget(key, *frames)

        min_bars = cls._deserialize_cached_bars(r1, FrameType.MIN1)

        if min_bars.size == 0:
            return min_bars

        if frame_type == FrameType.MIN1:
            return min_bars[-n:]

        bars = cls.resample(min_bars, from_frame=FrameType.MIN1, to_frame=frame_type)

        return bars[-n:]

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
        convert = tf.time2int if frame_type in tf.minute_level_frames else tf.date2int

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
        converter = tf.time2int if frame_type in tf.minute_level_frames else tf.date2int

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
            bars (np.ndarray): 行情数据
            from_frame (FrameType): 转换前的FrameType
            to_frame (FrameType): 转换后的FrameType

        Returns:
            np.ndarray: 转换后的行情数据
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
            FrameType.DAY,
        ):
            raise ValueError(f"unsupported to_frame: {to_frame}")

        bins_len = {
            FrameType.MIN5: 5,
            FrameType.MIN15: 15,
            FrameType.MIN30: 30,
            FrameType.MIN60: 60,
            FrameType.DAY: 240,
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

        if to_frame == FrameType.DAY:
            resampled["frame"] = bars[-1]["frame"].date()

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
        rules = {
            "frame": "last",
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "amount": "sum",
            "factor": "last",
        }

        if to_frame == FrameType.WEEK:
            freq = "W-Fri"
        elif to_frame == FrameType.MONTH:
            freq = "M"
        elif to_frame == FrameType.QUARTER:
            freq = "Q"
        elif to_frame == FrameType.YEAR:
            freq = "A"
        else:
            raise ValueError(f"unsupported to_frame: {to_frame}")

        df = pd.DataFrame(bars)
        df.index = pd.to_datetime(bars["frame"])
        df = df.resample(freq).agg(rules)
        bars = np.array(df.to_records(index=False), dtype=bars_dtype)

        # filter out data like (None, nan, ...)
        return bars[np.isfinite(bars["close"])]

    @classmethod
    async def get_trade_price_limits(
        cls, code: str, begin: Frame, end: Frame
    ) -> np.ndarray:
        """从influxdb中获取个股在[begin, end]之间的涨跌停价。

        涨跌停价只有日线数据才有，因此，FrameType固定为FrameType.DAY

        Args:
            code : 个股代码
            begin : 开始日期
            end : 结束日期

        Returns:
            dtype为[('frame', 'O'), ('high_limit', 'f4'), ('low_limit', 'f4')]的numpy数组
        """
        cols = ["_time", "high_limit", "low_limit"]
        dtype = [
            ("frame", "O"),
            ("high_limit", "f4"),
            ("low_limit", "f4"),
        ]

        client = cls._get_influx_client()
        measurement = cls._measurement_name(FrameType.DAY)
        flux = (
            Flux()
            .bucket(client._bucket)
            .measurement(measurement)
            .range(begin, end)
            .tags({"code": code})
            .fields(cols)
            .sort("_time")
        )

        ds = NumpyDeserializer(
            dtype,
            use_cols=cols,
            converters={
                "_time": lambda x: ciso8601.parse_datetime(x).date(),
            },
            # since we ask parse date in convertors, so we have to disable parse_date
            parse_date=None,
        )

        result = await client.query(flux, ds)
        return result

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

    @classmethod
    async def trade_price_limit_flags(
        cls, code: str, start: datetime.date, end: datetime.date
    ) -> Tuple[List[bool]]:
        """获取个股在[start, end]之间的涨跌停标志

        Args:
            code: 个股代码
            start: 开始日期
            end: 结束日期

        Returns:
            涨跌停标志列表(buy, sell)
        """
        cols = ["_time", "close", "high_limit", "low_limit"]
        client = cls._get_influx_client()
        measurement = cls._measurement_name(FrameType.DAY)
        flux = (
            Flux()
            .bucket(client._bucket)
            .measurement(measurement)
            .range(start, end)
            .tags({"code": code})
            .fields(cols)
            .sort("_time")
        )

        dtype = [
            ("frame", "O"),
            ("close", "f4"),
            ("high_limit", "f4"),
            ("low_limit", "f4"),
        ]
        ds = NumpyDeserializer(
            dtype,
            use_cols=["_time", "close", "high_limit", "low_limit"],
            converters={
                "_time": lambda x: ciso8601.parse_datetime(x).date(),
            },
            # since we ask parse date in convertors, so we have to disable parse_date
            parse_date=None,
        )

        result = await client.query(flux, ds)
        if result.size == 0:
            return np.array([], dtype=dtype)

        return (
            array_price_equal(result["close"], result["high_limit"]),
            array_price_equal(result["close"], result["low_limit"]),
        )
