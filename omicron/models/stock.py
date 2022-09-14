import datetime
import itertools
import logging
import re
from typing import Dict, Generator, Iterable, List, Tuple, Union

import arrow
import cfg4py
import ciso8601
import numpy as np
import pandas as pd
from coretypes import (
    BarsArray,
    BarsPanel,
    Frame,
    FrameType,
    LimitPriceOnlyBarsArray,
    SecurityType,
    bars_cols,
    bars_dtype,
    bars_dtype_with_code,
)

from omicron import tf
from omicron.core.constants import (
    TRADE_LATEST_PRICE,
    TRADE_PRICE_LIMITS,
    TRADE_PRICE_LIMITS_DATE,
)
from omicron.core.errors import BadParameterError
from omicron.dal import cache
from omicron.dal.influx.flux import Flux
from omicron.dal.influx.influxclient import InfluxClient
from omicron.dal.influx.serialize import DataframeDeserializer, NumpyDeserializer
from omicron.extensions.np import array_price_equal, numpy_append_fields
from omicron.models.security import Security, convert_nptime_to_datetime

logger = logging.getLogger(__name__)
cfg = cfg4py.get_instance()

INFLUXDB_MAX_QUERY_SIZE = 250 * 200


def ciso8601_parse_date(x):
    return ciso8601.parse_datetime(x).date()


def ciso8601_parse_naive(x):
    return ciso8601.parse_datetime_as_naive(x)


class Stock(Security):
    """ "
    Stock对象用于归集某支证券（股票和指数，不包括其它投资品种）的相关信息，比如行情数据（OHLC等）、市值数据、所属概念分类等。
    """

    _is_cache_empty = True

    def __init__(self, code: str):
        self._code = code
        self._stock = self.get_stock(code)
        assert self._stock, "系统中不存在该code"
        (_, self._display_name, self._name, ipo, end, _type) = self._stock
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
    def qfq(bars: BarsArray) -> BarsArray:
        """对行情数据执行前复权操作"""
        # todo: 这里可以优化
        if bars.size == 0:
            return bars

        last = bars[-1]["factor"]
        for field in ["open", "high", "low", "close", "volume"]:
            bars[field] = bars[field] * (bars["factor"] / last)

        return bars

    @classmethod
    async def batch_get_min_level_bars_in_range(
        cls,
        codes: List[str],
        frame_type: FrameType,
        start: Frame,
        end: Frame,
        fq: bool = True,
    ) -> Generator[Dict[str, BarsArray], None, None]:
        """获取多支股票（指数）在[start, end)时间段内的行情数据

        如果要获取的行情数据是分钟级别（即1m, 5m, 15m, 30m和60m)，使用本接口。

        停牌数据处理请见[get_bars][omicron.models.stock.Stock.get_bars]。

        本函数返回一个迭代器，使用方法示例：
        ```
        async for code, bars in Stock.batch_get_min_level_bars_in_range(...):
            print(code, bars)
        ```

        如果`end`不在`frame_type`所属的边界点上，那么，如果`end`大于等于当前缓存未收盘数据时间，则将包含未收盘数据；否则，返回的记录将截止到`tf.floor(end, frame_type)`。

        Args:
            codes: 股票/指数代码列表
            frame_type: 帧类型
            start: 起始时间
            end: 结束时间。如果未指明，则取当前时间。
            fq: 是否进行复权，如果是，则进行前复权。Defaults to True.

        Returns:
            Generator[Dict[str, BarsArray], None, None]: 迭代器，每次返回一个字典，其中key为代码，value为行情数据
        """
        closed_end = tf.floor(end, frame_type)
        n = tf.count_frames(start, closed_end, frame_type)
        max_query_size = min(cfg.influxdb.max_query_size, INFLUXDB_MAX_QUERY_SIZE)
        batch_size = max(1, max_query_size // n)
        ff = tf.first_min_frame(datetime.datetime.now(), frame_type)

        for i in range(0, len(codes), batch_size):
            batch_codes = codes[i : i + batch_size]

            if end < ff:
                part1 = await cls._batch_get_persisted_bars_in_range(
                    batch_codes, frame_type, start, end
                )
                part2 = pd.DataFrame([], columns=bars_dtype_with_code.names)
            elif start >= ff:
                part1 = pd.DataFrame([], columns=bars_dtype_with_code.names)
                n = tf.count_frames(start, closed_end, frame_type) + 1
                cached = await cls._batch_get_cached_bars_n(
                    frame_type, n, end, batch_codes
                )
                cached = cached[cached["frame"] >= start]
                part2 = pd.DataFrame(cached, columns=bars_dtype_with_code.names)
            else:
                part1 = await cls._batch_get_persisted_bars_in_range(
                    batch_codes, frame_type, start, ff
                )
                n = tf.count_frames(start, closed_end, frame_type) + 1
                cached = await cls._batch_get_cached_bars_n(
                    frame_type, n, end, batch_codes
                )
                part2 = pd.DataFrame(cached, columns=bars_dtype_with_code.names)

            df = pd.concat([part1, part2])

            for code in batch_codes:
                filtered = df[df["code"] == code][bars_cols]
                bars = filtered.to_records(index=False).astype(bars_dtype)
                if fq:
                    bars = cls.qfq(bars)

                yield code, bars

    @classmethod
    async def batch_get_day_level_bars_in_range(
        cls,
        codes: List[str],
        frame_type: FrameType,
        start: Frame,
        end: Frame,
        fq: bool = True,
    ) -> Generator[Dict[str, BarsArray], None, None]:
        """获取多支股票（指数）在[start, end)时间段内的行情数据

        如果要获取的行情数据是日线级别（即1d, 1w, 1M)，使用本接口。

        停牌数据处理请见[get_bars][omicron.models.stock.Stock.get_bars]。

        本函数返回一个迭代器，使用方法示例：
        ```
        async for code, bars in Stock.batch_get_day_level_bars_in_range(...):
            print(code, bars)
        ```

        如果`end`不在`frame_type`所属的边界点上，那么，如果`end`大于等于当前缓存未收盘数据时间，则将包含未收盘数据；否则，返回的记录将截止到`tf.floor(end, frame_type)`。

        Args:
            codes: 代码列表
            frame_type: 帧类型
            start: 起始时间
            end: 结束时间
            fq: 是否进行复权，如果是，则进行前复权。Defaults to True.

        Returns:
            Generator[Dict[str, BarsArray], None, None]: 迭代器，每次返回一个字典，其中key为代码，value为行情数据
        """
        today = datetime.datetime.now().date()
        # 日线，end不等于最后交易日，此时已无缓存
        if frame_type == FrameType.DAY and end == tf.floor(today, frame_type):
            from_cache = True
        elif frame_type != FrameType.DAY and start > tf.floor(today, frame_type):
            from_cache = True
        else:
            from_cache = False

        n = tf.count_frames(start, end, frame_type)
        max_query_size = min(cfg.influxdb.max_query_size, INFLUXDB_MAX_QUERY_SIZE)
        batch_size = max(max_query_size // n, 1)

        for i in range(0, len(codes), batch_size):
            batch_codes = codes[i : i + batch_size]
            persisted = await cls._batch_get_persisted_bars_in_range(
                batch_codes, frame_type, start, end
            )

            if from_cache:
                cached = await cls._batch_get_cached_bars_n(
                    frame_type, 1, end, batch_codes
                )
                cached = pd.DataFrame(cached, columns=bars_dtype_with_code.names)

                df = pd.concat([persisted, cached])
            else:
                df = persisted

            for code in batch_codes:
                filtered = df[df["code"] == code][bars_cols]
                bars = filtered.to_records(index=False).astype(bars_dtype)
                if fq:
                    bars = cls.qfq(bars)

                yield code, bars

    @classmethod
    async def get_bars_in_range(
        cls,
        code: str,
        frame_type: FrameType,
        start: Frame,
        end: Frame = None,
        fq=True,
        unclosed=True,
    ) -> BarsArray:
        """获取指定证券（`code`）在[`start`, `end`]期间帧类型为`frame_type`的行情数据。

        Args:
            code : 证券代码
            frame_type : 行情数据的帧类型
            start : 起始时间
            end : 结束时间,如果为None，则表明取到当前时间。
            fq : 是否对行情数据执行前复权操作
            unclosed : 是否包含未收盘的数据
        """
        now = datetime.datetime.now()
        if frame_type in tf.day_level_frames:
            end = end or now.date()
            if unclosed and tf.day_shift(end, 0) == now.date():
                part2 = await cls._get_cached_bars_n(code, 1, frame_type)
            else:
                part2 = np.array([], dtype=bars_dtype)

            # get rest from persisted
            part1 = await cls._get_persisted_bars_in_range(code, frame_type, start, end)
            bars = np.concatenate((part1, part2))
        else:
            end = end or now
            closed_end = tf.floor(end, frame_type)
            ff = tf.first_min_frame(now, frame_type)
            if end < ff:
                part1 = await cls._get_persisted_bars_in_range(
                    code, frame_type, start, end
                )
                part2 = np.array([], dtype=bars_dtype)
            elif start >= ff:  # all in cache
                part1 = np.array([], dtype=bars_dtype)
                n = tf.count_frames(start, closed_end, frame_type) + 1
                part2 = await cls._get_cached_bars_n(code, n, frame_type, end)
                part2 = part2[part2["frame"] >= start]
            else:  # in both cache and persisted
                part1 = await cls._get_persisted_bars_in_range(
                    code, frame_type, start, ff
                )
                n = tf.count_frames(ff, closed_end, frame_type) + 1
                part2 = await cls._get_cached_bars_n(code, n, frame_type, end)

            if not unclosed:
                part2 = part2[part2["frame"] <= closed_end]
            bars = np.concatenate((part1, part2))

        if fq:
            return cls.qfq(bars)
        else:
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
    ) -> BarsArray:
        """获取到`end`为止的`n`个行情数据。

        返回的数据是按照时间顺序递增排序的。在遇到停牌的情况时，该时段数据将被跳过，因此返回的记录可能不是交易日连续的，并且可能不足`n`个。

        如果系统当前没有到指定时间`end`的数据，将尽最大努力返回数据。调用者可以通过判断最后一条数据的时间是否等于`end`来判断是否获取到了全部数据。

        Args:
            code: 证券代码
            n: 记录数
            frame_type: 帧类型
            end: 截止时间,如果未指明，则取当前时间
            fq: 是否对返回记录进行复权。如果为`True`的话，则进行前复权。Defaults to True.
            unclosed: 是否包含最新未收盘的数据？ Defaults to True.

        Returns:
            返回dtype为`coretypes.bars_dtype`的一维numpy数组。
        """
        now = datetime.datetime.now()
        try:
            if frame_type in tf.day_level_frames:
                if end is None:
                    end = now.date()
                elif type(end) == datetime.datetime:
                    end = end.date()
                n0 = n
                if unclosed:
                    cached = await cls._get_cached_bars_n(code, 1, frame_type)
                    if cached.size > 0:
                        # 如果缓存的未收盘日期 > end，则该缓存不是需要的
                        if cached[0]["frame"].item().date() > end:
                            cached = np.array([], dtype=bars_dtype)
                        else:
                            n0 = n - 1
            else:
                end = end or now
                closed_frame = tf.floor(end, frame_type)

                # fetch one more bar, in case we should discard unclosed bar
                cached = await cls._get_cached_bars_n(code, n + 1, frame_type, end)
                if not unclosed:
                    cached = cached[cached["frame"] <= closed_frame]

                # n bars we need fetch from persisted db
                n0 = n - cached.size
            if n0 > 0:
                if cached.size > 0:
                    end0 = cached[0]["frame"].item()
                else:
                    end0 = end

                bars = await cls._get_persisted_bars_n(code, frame_type, n0, end0)
                merged = np.concatenate((bars, cached))
                bars = merged[-n:]
            else:
                bars = cached[-n:]

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
    async def _get_persisted_bars_in_range(
        cls, code: str, frame_type: FrameType, start: Frame, end: Frame = None
    ) -> BarsArray:
        """从持久化数据库中获取介于[`start`, `end`]间的行情记录

        如果`start`到`end`区间某支股票停牌，则会返回空数组。

        Args:
            code: 证券代码
            frame_type: 帧类型
            start: 起始时间
            end: 结束时间，如果未指明，则取当前时间

        Returns:
            返回dtype为`coretypes.bars_dtype`的一维numpy数组。
        """
        end = end or datetime.datetime.now()

        keep_cols = ["_time"] + list(bars_cols[1:])

        measurement = cls._measurement_name(frame_type)
        flux = (
            Flux()
            .bucket(cfg.influxdb.bucket_name)
            .range(start, end)
            .measurement(measurement)
            .fields(keep_cols)
            .tags({"code": code})
        )

        serializer = DataframeDeserializer(
            encoding="utf-8",
            names=[
                "_",
                "table",
                "result",
                "frame",
                "code",
                "amount",
                "close",
                "factor",
                "high",
                "low",
                "open",
                "volume",
            ],
            engine="c",
            skiprows=0,
            header=0,
            usecols=bars_cols,
            parse_dates=["frame"],
        )

        url = cfg.influxdb.url
        token = cfg.influxdb.token
        bucket = cfg.influxdb.bucket_name
        org = cfg.influxdb.org

        client = InfluxClient(url, token, bucket, org)
        result = await client.query(flux, serializer)
        return result.to_records(index=False).astype(bars_dtype)

    @classmethod
    async def _get_persisted_bars_n(
        cls, code: str, frame_type: FrameType, n: int, end: Frame = None
    ) -> BarsArray:
        """从持久化数据库中获取截止到`end`的`n`条行情记录

        如果`end`未指定，则取当前时间。

        基于influxdb查询的特性，在查询前，必须先根据`end`和`n`计算出起始时间，但如果在此期间某些股票有停牌，则无法返回的数据将小于`n`。而如果起始时间设置得足够早，虽然能满足返回数据条数的要求，但会带来性能上的损失。因此，我们在计算起始时间时，不是使用`n`来计算，而是使用了`min(n * 2, n + 20)`来计算起始时间，这样多数情况下，能够保证返回数据的条数为`n`条。

        返回的数据按`frame`进行升序排列。

        Args:
            code: 证券代码
            frame_type: 帧类型
            n: 返回结果数量
            end: 结束时间，如果未指明，则取当前时间

        Returns:
            返回dtype为`bars_dtype`的numpy数组
        """
        # check is needed since tags accept List as well
        assert isinstance(code, str), "`code` must be a string"

        end = end or datetime.datetime.now()
        closed_end = tf.floor(end, frame_type)
        start = tf.shift(closed_end, -min(2 * n, n + 20), frame_type)

        keep_cols = ["_time"] + list(bars_cols[1:])

        measurement = cls._measurement_name(frame_type)
        flux = (
            Flux()
            .bucket(cfg.influxdb.bucket_name)
            .range(start, end)
            .measurement(measurement)
            .fields(keep_cols)
            .tags({"code": code})
            .latest(n)
        )

        serializer = DataframeDeserializer(
            encoding="utf-8",
            names=[
                "_",
                "table",
                "result",
                "frame",
                "code",
                "amount",
                "close",
                "factor",
                "high",
                "low",
                "open",
                "volume",
            ],
            engine="c",
            skiprows=0,
            header=0,
            usecols=bars_cols,
            parse_dates=["frame"],
        )

        url = cfg.influxdb.url
        token = cfg.influxdb.token
        bucket = cfg.influxdb.bucket_name
        org = cfg.influxdb.org

        client = InfluxClient(url, token, bucket, org)
        result = await client.query(flux, serializer)
        return result.to_records(index=False).astype(bars_dtype)

    @classmethod
    async def _batch_get_persisted_bars_n(
        cls, codes: List[str], frame_type: FrameType, n: int, end: Frame = None
    ) -> pd.DataFrame:
        """从持久化存储中获取`codes`指定的一批股票截止`end`时的`n`条记录。

        返回的数据按`frame`进行升序排列。如果不存在满足指定条件的查询结果，将返回空的DataFrame。

        基于influxdb查询的特性，在查询前，必须先根据`end`和`n`计算出起始时间，但如果在此期间某些股票有停牌，则无法返回的数据将小于`n`。如果起始时间设置的足够早，虽然能满足返回数据条数的要求，但会带来性能上的损失。因此，我们在计算起始时间时，不是使用`n`来计算，而是使用了`min(n * 2, n + 20)`来计算起始时间，这样多数情况下，能够保证返回数据的条数为`n`条。

        Args:
            codes: 证券代码列表。
            frame_type: 帧类型
            n: 返回结果数量
            end: 结束时间，如果未指定，则使用当前时间

        Returns:
            DataFrame, columns为`code`, `frame`, `open`, `high`, `low`, `close`, `volume`, `amount`, `factor`

        """
        max_query_size = min(cfg.influxdb.max_query_size, INFLUXDB_MAX_QUERY_SIZE)

        if len(codes) * min(n + 20, 2 * n) > max_query_size:
            raise BadParameterError(
                f"codes的数量和n的乘积超过了influxdb的最大查询数量限制{max_query_size}"
            )

        end = end or datetime.datetime.now()
        close_end = tf.floor(end, frame_type)
        begin = tf.shift(close_end, -1 * min(n + 20, n * 2), frame_type)

        # influxdb的查询结果格式类似于CSV，其列顺序为_, result_alias, table_seq, _time, tags, fields,其中tags和fields都是升序排列
        keep_cols = ["code"] + list(bars_cols)
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
            .latest(n)
        )

        if codes is not None:
            assert isinstance(codes, list), "`codes` must be a list or None"
            flux.tags({"code": codes})

        deserializer = DataframeDeserializer(
            names=names,
            usecols=keep_cols,
            encoding="utf-8",
            time_col="frame",
            engine="c",
        )

        client = cls._get_influx_client()
        return await client.query(flux, deserializer)

    @classmethod
    async def _batch_get_persisted_bars_in_range(
        cls, codes: List[str], frame_type: FrameType, begin: Frame, end: Frame = None
    ) -> pd.DataFrame:
        """从持久化存储中获取`codes`指定的一批股票在`begin`和`end`之间的记录。

        返回的数据将按`frame`进行升序排列。
        注意，返回的数据有可能不是等长的，因为有的股票可能停牌。

        Args:
            codes: 证券代码列表。
            frame_type: 帧类型
            begin: 开始时间
            end: 结束时间

        Returns:
            DataFrame, columns为`code`, `frame`, `open`, `high`, `low`, `close`, `volume`, `amount`, `factor`

        """
        end = end or datetime.datetime.now()

        n = tf.count_frames(begin, end, frame_type)
        max_query_size = min(cfg.influxdb.max_query_size, INFLUXDB_MAX_QUERY_SIZE)
        if len(codes) * n > max_query_size:
            raise BadParameterError(
                f"asked records is {len(codes) * n}, which is too large than {max_query_size}"
            )

        # influxdb的查询结果格式类似于CSV，其列顺序为_, result_alias, table_seq, _time, tags, fields,其中tags和fields都是升序排列
        keep_cols = ["code"] + list(bars_cols)
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

        flux.tags({"code": codes})

        deserializer = DataframeDeserializer(
            names=names,
            usecols=keep_cols,
            encoding="utf-8",
            time_col="frame",
            engine="c",
        )

        client = cls._get_influx_client()
        df = await client.query(flux, deserializer)
        return df

    @classmethod
    async def batch_cache_bars(cls, frame_type: FrameType, bars: Dict[str, BarsArray]):
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
                frame = tf.time2int(bar["frame"].item())
                val = [*bar]
                val[0] = frame
                pl.hset(key, frame, ",".join(map(str, val)))
        await pl.execute()

    @classmethod
    async def batch_cache_unclosed_bars(
        cls, frame_type: FrameType, bars: Dict[str, BarsArray]
    ):  # pragma: no cover
        """缓存未收盘的5、15、30、60分钟线及日线、周线、月线

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
            val[0] = convert(bar["frame"][0].item())  # 时间转换
            pl.hset(key, code, ",".join(map(str, val)))

        await pl.execute()

    @classmethod
    async def reset_cache(cls):
        """清除缓存的行情数据"""
        try:
            for ft in itertools.chain(tf.minute_level_frames, tf.day_level_frames):
                keys = await cache.security.keys(f"bars:{ft.value}:*")
                if keys:
                    await cache.security.delete(*keys)
        finally:
            cls._is_cache_empty = True

    @classmethod
    def _deserialize_cached_bars(cls, raw: List[str], ft: FrameType) -> BarsArray:
        """从redis中反序列化缓存的数据

        如果`raw`空数组或者元素为`None`，则返回空数组。

        Args:
            raw: redis中的缓存数据
            ft: 帧类型
            sort: 是否需要重新排序，缺省为False

        Returns:
            BarsArray: 行情数据
        """
        if ft in tf.minute_level_frames:
            convert = tf.int2time
        else:
            convert = tf.int2date
        recs = []
        # it's possible to treat raw as csv and use pandas to parse, however, the performance is 10 times worse than this method
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
    async def _batch_get_cached_bars_n(
        cls, frame_type: FrameType, n: int, end: Frame = None, codes: List[str] = None
    ) -> BarsPanel:
        """批量获取在cache中截止`end`的`n`个bars。

        如果`end`不在`frame_type`所属的边界点上，那么，如果`end`大于等于当前缓存未收盘数据时间，则将包含未收盘数据；否则，返回的记录将截止到`tf.floor(end, frame_type)`。

        Args:
            frame_type: 时间帧类型
            n: 返回记录条数
            codes: 证券代码列表
            end: 截止时间, 如果为None

        Returns:
            BarsPanel: 行情数据
        """
        # 调用者自己保证end在缓存中
        cols = list(bars_dtype_with_code.names)
        if frame_type in tf.day_level_frames:
            key = f"bars:{frame_type.value}:unclosed"
            if codes is None:
                recs = await cache.security.hgetall(key)
                codes = list(recs.keys())
                recs = recs.values()
            else:
                recs = await cache.security.hmget(key, *codes)

            barss = cls._deserialize_cached_bars(recs, frame_type)
            if barss.size > 0:
                if len(barss) != len(codes):
                    # issue 39, 如果某支票当天停牌，则缓存中将不会有它的记录，此时需要移除其代码
                    codes = [
                        codes[i] for i, item in enumerate(recs) if item is not None
                    ]
                barss = numpy_append_fields(barss, "code", codes, [("code", "O")])
                return barss[cols].astype(bars_dtype_with_code)
            else:
                return np.array([], dtype=bars_dtype_with_code)
        else:
            end = end or datetime.datetime.now()
            close_end = tf.floor(end, frame_type)
            all_bars = []
            if codes is None:
                keys = await cache.security.keys(
                    f"bars:{frame_type.value}:*[^unclosed]"
                )
                codes = [key.split(":")[-1] for key in keys]
            else:
                keys = [f"bars:{frame_type.value}:{code}" for code in codes]

            if frame_type != FrameType.MIN1:
                unclosed = await cache.security.hgetall(
                    f"bars:{frame_type.value}:unclosed"
                )
            else:
                unclosed = {}

            pl = cache.security.pipeline()
            frames = tf.get_frames_by_count(close_end, n, frame_type)
            for key in keys:
                pl.hmget(key, *frames)

            all_closed = await pl.execute()
            for code, raw in zip(codes, all_closed):
                raw.append(unclosed.get(code))
                barss = cls._deserialize_cached_bars(raw, frame_type)
                barss = numpy_append_fields(
                    barss, "code", [code] * len(barss), [("code", "O")]
                )
                barss = barss[cols].astype(bars_dtype_with_code)
                all_bars.append(barss[barss["frame"] <= end][-n:])

            try:
                return np.concatenate(all_bars)
            except ValueError as e:
                logger.exception(e)
                return np.array([], dtype=bars_dtype_with_code)

    @classmethod
    async def _get_cached_bars_n(
        cls, code: str, n: int, frame_type: FrameType, end: Frame = None
    ) -> BarsArray:
        """从缓存中获取指定代码的行情数据

        存取逻辑是，从`end`指定的时间向前取`n`条记录。`end`不应该大于当前系统时间，并且根据`end`和`n`计算出来的起始时间应该在缓存中存在。否则，两种情况下，返回记录数都将小于`n`。

        如果`end`不处于`frame_type`所属的边界结束位置，且小于当前已缓存的未收盘bar时间，则会返回前一个已收盘的数据，否则，返回的记录中还将包含未收盘的数据。

        args:
            code: 证券代码，比如000001.XSHE
            n: 返回记录条数
            frame_type: 帧类型
            end: 结束帧，如果为None，则取当前时间

        returns:
            元素类型为`coretypes.bars_dtype`的一维numpy数组。如果没有数据，则返回空ndarray。
        """
        # 50 times faster than arrow.now().floor('second')
        end = end or datetime.datetime.now().replace(second=0, microsecond=0)

        if frame_type in tf.minute_level_frames:
            cache_start = tf.first_min_frame(end.date(), frame_type)
            closed = tf.floor(end, frame_type)

            frames = (tf.get_frames(cache_start, closed, frame_type))[-n:]
            if len(frames) == 0:
                return np.empty(shape=(0,), dtype=bars_dtype)

            key = f"bars:{frame_type.value}:{code}"
            recs = await cache.security.hmget(key, *frames)
            recs = cls._deserialize_cached_bars(recs, frame_type)

            if closed < end:
                # for unclosed
                key = f"bars:{frame_type.value}:unclosed"
                unclosed = await cache.security.hget(key, code)
                unclosed = cls._deserialize_cached_bars([unclosed], frame_type)

                if end < unclosed[0]["frame"].item():
                    # 如果unclosed为9:36, 调用者要求取9:29的5m数据，则取到的unclosed不合要求，抛弃。似乎没有更好的方法检测end与unclosed的关系
                    return recs[-n:]
                else:
                    bars = np.concatenate((recs, unclosed))
                    return bars[-n:]
            else:
                return recs[-n:]
        else:  # 日线及以上级别，仅在缓存中存在未收盘数据
            key = f"bars:{frame_type.value}:unclosed"
            rec = await cache.security.hget(key, code)
            return cls._deserialize_cached_bars([rec], frame_type)

    @classmethod
    async def cache_bars(cls, code: str, frame_type: FrameType, bars: BarsArray):
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
            val[0] = convert(bar["frame"].item())
            pl.hset(key, val[0], ",".join(map(str, val)))

        await pl.execute()

    @classmethod
    async def cache_unclosed_bars(
        cls, code: str, frame_type: FrameType, bars: BarsArray
    ):  # pragma: no cover
        """将未结束的行情数据缓存

        未结束的行情数据缓存在以`bars:{frame_type.value}:unclosed`为key, {code}为field的hashmap中。

        尽管`bars`被声明为BarsArray，但实际上应该只包含一个元素。

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
        bar = bars[0]
        val = [*bar]
        val[0] = converter(bar["frame"].item())
        await cache.security.hset(key, code, ",".join(map(str, val)))

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
        bars: Union[Dict[str, BarsArray], BarsArray, pd.DataFrame],
    ):
        """将行情数据持久化

        如果`bars`类型为Dict,则key为`code`，value为`bars`。如果其类型为BarsArray或者pd.DataFrame，则`bars`各列字段应该为`coretypes.bars_dtype` + ("code", "O")构成。

        Args:
            frame_type: the frame type of the bars
            bars: the bars to be persisted

        Raises:
            InfluxDBWriteError: if influxdb write failed
        """
        client = cls._get_influx_client()

        measurement = cls._measurement_name(frame_type)
        logger.info("persisting bars to influxdb: %s, %d secs", measurement, len(bars))

        if isinstance(bars, dict):
            for code, value in bars.items():
                await client.save(
                    value, measurement, global_tags={"code": code}, time_key="frame"
                )
        else:
            await client.save(bars, measurement, tag_keys=["code"], time_key="frame")

    @classmethod
    def resample(
        cls, bars: BarsArray, from_frame: FrameType, to_frame: FrameType
    ) -> BarsArray:
        """将原来为`from_frame`的行情数据转换为`to_frame`的行情数据

        如果`to_frame`为日线或者分钟级别线，则`from_frame`必须为分钟线；如果`to_frame`为周以上级别线，则`from_frame`必须为日线。其它级别之间的转换不支持。

        如果`from_frame`为1分钟线，则必须从9：31起。

        Args:
            bars (BarsArray): 行情数据
            from_frame (FrameType): 转换前的FrameType
            to_frame (FrameType): 转换后的FrameType

        Returns:
            BarsArray: 转换后的行情数据
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
    def _resample_from_min1(cls, bars: BarsArray, to_frame: FrameType) -> BarsArray:
        """将`bars`从1分钟线转换为`to_frame`的行情数据

        重采样后的数据只包含frame, open, high, low, close, volume, amount, factor，无论传入数据是否还有别的字段，它们都将被丢弃。

        resampling 240根分钟线到5分钟大约需要100微秒。

        TODO： 如果`bars`中包含nan怎么处理？
        """
        if bars[0]["frame"].item().minute != 31:
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
            resampled["frame"] = bars[-1]["frame"].item().date()

        return resampled

    @classmethod
    def _resample_from_day(cls, bars: BarsArray, to_frame: FrameType) -> BarsArray:
        """将`bars`从日线转换成`to_frame`的行情数据

        Args:
            bars (BarsArray): [description]
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
    async def _get_price_limit_in_cache(
        cls, code: str, begin: datetime.date, end: datetime.date
    ):
        date_str = await cache._security_.get(TRADE_PRICE_LIMITS_DATE)
        if date_str:
            date_in_cache = arrow.get(date_str).date()
            if date_in_cache < begin or date_in_cache > end:
                return None
        else:
            return None

        dtype = [("frame", "O"), ("high_limit", "f4"), ("low_limit", "f4")]
        hp = await cache._security_.hget(TRADE_PRICE_LIMITS, f"{code}.high_limit")
        lp = await cache._security_.hget(TRADE_PRICE_LIMITS, f"{code}.low_limit")
        if hp is None or lp is None:
            return None
        else:
            return np.array([(date_in_cache, hp, lp)], dtype=dtype)

    @classmethod
    async def get_trade_price_limits(
        cls, code: str, begin: Frame, end: Frame
    ) -> BarsArray:
        """从influxdb和cache中获取个股在[begin, end]之间的涨跌停价。

        涨跌停价只有日线数据才有，因此，FrameType固定为FrameType.DAY，
        当天的数据存放于redis，如果查询日期包含当天（交易日），从cache中读取并追加到结果中

        Args:
            code : 个股代码
            begin : 开始日期
            end : 结束日期

        Returns:
            dtype为[('frame', 'O'), ('high_limit', 'f4'), ('low_limit', 'f4')]的numpy数组
        """
        cols = ["_time", "high_limit", "low_limit"]
        dtype = [("frame", "O"), ("high_limit", "f4"), ("low_limit", "f4")]

        if isinstance(begin, datetime.datetime):
            begin = begin.date()  # 强制转换为date
        if isinstance(end, datetime.datetime):
            end = end.date()  # 强制转换为date

        data_in_cache = await cls._get_price_limit_in_cache(code, begin, end)

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
            converters={"_time": lambda x: ciso8601.parse_datetime(x).date()},
            # since we ask parse date in convertors, so we have to disable parse_date
            parse_date=None,
        )

        result = await client.query(flux, ds)
        if data_in_cache:
            result = np.concatenate([result, data_in_cache])
        return result

    @classmethod
    async def reset_price_limits_cache(cls, cache_only: bool, dt: datetime.date = None):
        if cache_only is False:
            date_str = await cache._security_.get(TRADE_PRICE_LIMITS_DATE)
            if not date_str:
                return  # skip clear action if date not found in cache
            date_in_cache = arrow.get(date_str).date()
            if dt is None or date_in_cache != dt:  # 更新的时间和cache的时间相同，则清除cache
                return  # skip clear action

        await cache._security_.delete(TRADE_PRICE_LIMITS)
        await cache._security_.delete(TRADE_PRICE_LIMITS_DATE)

    @classmethod
    async def save_trade_price_limits(
        cls, price_limits: LimitPriceOnlyBarsArray, to_cache: bool
    ):
        """保存涨跌停价

        Args:
            price_limits: 要保存的涨跌停价格数据。
            to_cache: 是保存到缓存中，还是保存到持久化存储中
        """
        if len(price_limits) == 0:
            return

        if to_cache:  # 每个交易日上午9点更新两次
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

            dt = price_limits[-1]["frame"]
            pl.set(TRADE_PRICE_LIMITS_DATE, dt.strftime("%Y-%m-%d"))
            await pl.execute()
        else:
            # to influxdb， 每个交易日的第二天早上2点保存
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
            converters={"_time": lambda x: ciso8601.parse_datetime(x).date()},
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

    @classmethod
    async def get_latest_price(cls, codes: Iterable[str]) -> List[str]:
        """获取多支股票的最新价格（交易日当天），暂不包括指数

        价格数据每5秒更新一次，接受多只股票查询，返回最后缓存的价格

        Args:
            codes: 代码列表

        Returns:
            返回一个List，价格是字符形式的浮点数。
        """
        if not codes:
            return []

        _raw_code_list = []
        for code_str in codes:
            code, _ = code_str.split(".")
            _raw_code_list.append(code)

        _converted_data = []
        raw_data = await cache.feature.hmget(TRADE_LATEST_PRICE, *_raw_code_list)
        for _data in raw_data:
            if _data is None:
                _converted_data.append(_data)
            else:
                _converted_data.append(float(_data))
        return _converted_data
