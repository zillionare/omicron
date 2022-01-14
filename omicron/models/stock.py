import datetime
import logging
import re
from typing import List

import arrow
import numpy as np
import pandas as pd

from omicron.core.errors import DataNotReadyError
from omicron.core.types import (
    Frame,
    FrameType,
    MarketType,
    SecurityType,
    stock_bars_dtype,
)
from omicron.dal import cache, influxdb
from omicron.models.calendar import Calendar as cal

logger = logging.getLogger(__name__)


class Stock:
    """ "
    Stock对象用于归集某支证券（股票和指数，不包括其它投资品种）的相关信息，比如行情数据（OHLC等）、市值数据、所属概念分类等。
    """

    _stocks = None
    fileds_type = [
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
        (
            _,
            self._display_name,
            self._name,
            self._start_date,
            self._end_date,
            _type,
        ) = self.all_stocks[code]
        self._type = SecurityType(_type)

    @classmethod
    async def load_securities(cls):
        """
        加载所有证券的信息，并缓存到内存中。
        """
        secs = await cache.security.lrange("security:stock", 0, -1, encoding="utf-8")
        if len(secs) != 0:
            _stocks = np.array(
                [tuple(x.split(",")) for x in secs], dtype=cls.fileds_type
            )

            _stocks = _stocks[
                (_stocks["type"] == "stock") | (_stocks["type"] == "index")
            ]

            _stocks["ipo"] = [arrow.get(x).date() for x in _stocks["ipo"]]
            _stocks["end"] = [arrow.get(x).date() for x in _stocks["end"]]

            return _stocks
        else:
            return None

    @classmethod
    async def init(cls):
        secs = await cls.load_securities()
        if len(secs) != 0:
            cls._stocks = secs
        else:
            raise DataNotReadyError(
                "No securities in cache, make sure you have called omicron.init() first."
            )

    @classmethod
    async def save_securities(cls, securities: List[str]):
        """
        保存指定的证券到缓存中。
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
        result = np.array(result, dtype=cls.fileds_type)
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
    def type(self) -> SecurityType:
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

        Returns:
            int: [description]
        """
        epoch_start = arrow.get("2005-01-04").date()
        ipo_day = self.ipo_date if self.ipo_date > epoch_start else epoch_start
        return cal.count_day_frames(ipo_day, arrow.now().date())

    @staticmethod
    def parse_security_type(code: str) -> SecurityType:
        """
        通过证券代码全称，解析出A股MarketType, 证券类型SecurityType和简码
        Args:
            code:

        Returns:

        """
        market = MarketType(code[-4:])
        _type = SecurityType.UNKNOWN
        s1, s2, s3 = code[0], code[0:2], code[0:3]
        if market == MarketType.XSHG:
            if s1 == "6":
                _type = SecurityType.STOCK
            elif s3 in {"000", "880", "999"}:
                _type = SecurityType.INDEX
            elif s2 == "51":
                _type = SecurityType.ETF
            elif s3 in {"129", "100", "110", "120"}:
                _type = SecurityType.BOND
            else:
                _type = SecurityType.UNKNOWN

        else:
            if s2 in {"00", "30"}:
                _type = SecurityType.STOCK
            elif s2 == "39":
                _type = SecurityType.INDEX
            elif s2 == "15":
                _type = SecurityType.ETF
            elif s2 in {"10", "11", "12", "13"}:
                _type = SecurityType.BOND
            elif s2 == "20":
                _type = SecurityType.STOCK_B

        return _type

    @staticmethod
    def qfq(bars) -> np.ndarray:
        """对行情数据执行前复权操作"""
        last = bars[-1]["factor"]
        for field in ["open", "high", "low", "close", "volume"]:
            bars[field] = (bars[field] / last) * bars["factor"]

        return bars

    @classmethod
    async def get_bars_in_range(
        cls, begin: Frame, end: Frame, frame_type: FrameType, fq=True
    ):
        """获取在`[start, stop]`间的行情数据。

        Args:
            begin (Frame): [description]
            end (Frame): [description]
            frame_type (FrameType): [description]
            fq (bool, optional): [description]. Defaults to True.
        """
        raise NotImplementedError

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
        """获取最近的`n`个行情数据。

        返回的数据包含以下字段：

        frame, open, high, low, close, volume, amount, high_limit

        返回数据格式为numpy strucutre array，每一行对应一个bar,可以通过下标访问，如`bars['frame'][-1]`

        返回的数据是按照时间顺序递增排序的。在遇到停牌的情况时，该时段数据将被跳过，因此返回的记录可能不是交易日连续的。

        如果系统当前没有到指定时间`end`的数据，将尽最大努力返回数据。调用者可以通过判断最后一条数据的时间是否等于`end`来判断是否获取到了全部数据。

        Args:
            code (str): 证券代码
            n (int): 记录数
            frame_type (FrameType): 帧类型
            end (Frame): 截止时间,如果未指明，则取当前时间
            fq (bool, optional): [description]. Defaults to True.
            unclosed (bool, optional): 是否包含最新未收盘的数据？ Defaults to True.
        """
        end = end or arrow.now().naive

        part2 = await cls._get_cached_bars(code, end, n, frame_type, unclosed)

        n2 = len(part2)
        n1 = n - n2
        if n1 > 0:
            pend = cal.shift(cls.get_cached_first_frame(frame_type), -1, frame_type)
            part1 = await cls._get_persited_bars(code, pend, n1, frame_type, unclosed)
        else:
            part1 = np.empty((0,), dtype=stock_bars_dtype)

        bars = np.concatenate([part1, part2])
        assert len(bars) == n

        if fq:
            bars = cls.qfq(bars)

        return bars

    @classmethod
    async def _get_persited_bars(
        cls, code: str, n: int, frame_type: FrameType, end: Frame
    ):
        """从influxdb中获取数据

        Args:
            code (str): [description]
            n (int): [description]
            frame_type (FrameType): [description]
            end (Frame): [description]

        Raises:
            NotImplemented: [description]
        """
        raise NotImplementedError

    @classmethod
    async def batch_get_bars(
        cls,
        codes: List[str],
        n: int,
        frame_type: FrameType,
        end: Frame = None,
        fq=True,
        closed=True,
        skip_paused=True,
    ) -> dict:
        """获取多支股票（指数）的最近的`n`个行情数据。

        停牌数据处理请见 `get_bars`

        结果以dict方式返回，key为传入的股票代码，value为对应的行情数据。

        Args:
            codes (List[str]): [description]
            n (int): [description]
            frame_type (FrameType): [description]
            end (Frame, optional): 结束时间。如果未指明，则取当前时间。 Defaults to None.
            fq (bool, optional): [description]. Defaults to True.
            closed (bool, optional): [description]. Defaults to True.

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError

    @classmethod
    async def batch_cache_bars(cls, frame_type: FrameType, bars: np.ndarray):
        """缓存已收盘的分钟线和日线

        当缓存日线时，仅限于当日收盘后的第一次同步时调用。

        bars的数据结构为：
        （[“frame", "open", ..., "factor", "code”)]

        bars中可能存在同一`code`的多条数据，这些数据在时间上按增序排列
        """
        if frame_type == FrameType.DAY:
            await cls.batch_cache_unclosed_bars(frame_type, bars)
            return

        pl = cache.security.pipeline()
        for bar in bars:
            code = bar["code"]
            frame = cal.time2int(bar["frame"])
            val = [*bar][:-1]
            val[0] = frame
            pl.hset(f"bars:{frame_type.value}:{code}", frame, ",".join(map(str, val)))
        await pl.execute()

        cls.set_cached(bars[0]["frame"])

    @classmethod
    async def batch_cache_unclosed_bars(cls, frame_type: FrameType, bars: np.ndarray):
        """缓存未收盘的5、15、30、60分钟线及日线

        `bars`数据结构同`batch_cache_bars`方法。 `bars`中不应该存在同一code的多条数据。
        """
        pl = cache.security.pipeline()
        key = f"bars:{frame_type.value}:unclosed"

        convert = (
            cal.time2int if frame_type in cal.minute_level_frames else cal.time2date
        )

        for bar in bars:
            code = bar["code"]
            val = [*bar[:-1]]  # 去掉code
            val[0] = convert(bar["frame"])  # 时间转换
            pl.hset(key, code, ",".join(map(str, val)))
        await pl.execute()

        cal.set_cached(bars[0]["frame"])

    @classmethod
    async def reset_cache(cls):
        """清除缓存"""
        try:
            for ft in cal.minute_level_frames:
                await cache.security.delete(f"bars:{ft.value}:unclosed")
                keys = await cache.security.keys(f"bars:{ft.value}:*")
                if keys:
                    await cache.security.delete(*keys)
        finally:
            cls._is_cache_empty = True

    @classmethod
    def get_cached_first_frame(cls, frame_type: FrameType) -> Frame:
        if cls._is_cache_empty:
            return None

        return cls._cached_frames_start.get(frame_type, None)

    @classmethod
    def set_cached(cls, frame: Frame):
        if cls._is_cache_empty:
            dt = arrow.get(frame).date()

            for ft in cal.minute_level_frames:
                frame = cal.first_min_frame(dt, ft)
                cls._cached_frames_start[ft] = frame
            cls._cached_frames_start[FrameType.DAY] = dt

            cls._is_cache_empty = False
        else:
            return

    @classmethod
    async def _get_cached_bars(
        cls,
        code: str,
        end: Frame,
        n: int,
        frame_type: FrameType,
        unclosed=True,
        fq=True,
    ) -> np.ndarray:
        """从缓存中获取指定代码的行情数据

        如果行情数据为日线以上级别，则最多只会返回一条数据（也可能没有）。如果行情数据为分钟级别数据，则一次返回当天已缓存的所有数据。

        本接口在如下场景下，性能不是最优的：
        如果cache中存在接近240根分钟线，取截止到9：35分的前5根K线，此段实现也会取出全部k线，但只返回前5根。这样会引起不必要的网络通信及反串行化时间。

        args:
            code: the full qualified code of a security or index
            end: the end frame of the bars
            frame_type: use this to decide which store to use
        """
        ff = cls.get_cached_first_frame(frame_type)

        if ff is None or end < ff:
            return np.empty((0,), dtype=stock_bars_dtype)

        raw = []
        if frame_type in cal.day_level_frames:
            convert = cal.int2date
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
            convert = cal.int2time
            key = f"bars:{frame_type.value}:{code}"
            end_ = cal.floor(end, frame_type)
            frames = map(str, cal.get_frames_by_count(end_, n, frame_type))
            r1 = await cache.security.hmget(key, *frames)
            raw.extend(r1)

            if unclosed:
                key = f"bars:{frame_type.value}:unclosed"
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

        bars = np.array(recs, dtype=stock_bars_dtype)[-n:]
        if bars[-1]["frame"] > end:
            # 避免取到未来数据
            bars = bars[:-1]
        if fq:
            return cls.qfq(bars)
        else:
            return bars

    @classmethod
    async def cache_bars(cls, code: str, frame_type: FrameType, bars: np.ndarray):
        """将行情数据缓存"""
        today = bars[0]["frame"]
        # 转换时间为int
        convert = (
            cal.time2int if frame_type in cal.minute_level_frames else cal.date2int
        )

        key = f"bars:{frame_type.value}:{code}"
        pl = cache.security.pipeline()
        for bar in bars:
            val = [*bar]
            val[0] = convert(bar["frame"])
            pl.hset(key, val[0], ",".join(map(str, val)))

        await pl.execute()

        cls.set_cached(today)

    @classmethod
    async def cache_unclosed_bars(
        cls, code: str, frame_type: FrameType, bars: np.ndarray
    ):
        """将未结束的行情数据缓存"""
        today = bars[0]["frame"]
        converter = (
            cal.time2int if frame_type in cal.minute_level_frames else cal.date2int
        )

        assert len(bars) == 1, "unclosed bars should only have one record"

        key = f"bars:{frame_type.value}:unclosed"
        pl = cache.security.pipeline()
        for bar in bars:
            val = [*bar]
            val[0] = converter(bar["frame"])
            pl.hset(key, code, ",".join(map(str, val)))

        await pl.execute()
        cls.set_cached(today)

    @classmethod
    async def persist_bars(cls, frame_type: FrameType, bars: np.ndarray):
        """将行情数据持久化"""
        df = pd.DataFrame(data=bars, columns=bars.dtype.names)
        df["frame_type"] = frame_type.to_int()
        df.index = df["frame"]
        await influxdb.write("zillionare", df, "stock", ["frame", "frame_type", "code"])

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
        elif from_frame == FrameType.DAY:
            return cls._resample_from_day(bars, to_frame)
        else:
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
            resampled = np.empty((bins + 1,), dtype=stock_bars_dtype)
        else:
            resampled = np.empty((bins,), dtype=stock_bars_dtype)

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
            np.ndarray: [description]
        """
        raise NotImplementedError

    @classmethod
    async def get_limits_in_range(cls, code: str, begin: Frame, end: Frame) -> np.array:
        """获取股票的涨跌停价"""
        now = datetime.datetime.now()
        assert begin > now, "begin time can't gt now()"
        assert begin > end, "begin time can't gt end time"
        df = pd.DataFrame(
            columns=["code", "frame", "frame_type", "high_limit", "low_limit", "close"],
        )
        if begin < now:
            df = await influxdb.get_limit_in_date_range(
                bucket="zillionare", code=code, begin=begin, end=end
            )
            df = df.sort_values(
                by=[
                    "frame",
                ],
                ascending=[
                    False,
                ],
            )
        df1 = pd.DataFrame(
            columns=["code", "frame", "frame_type", "high_limit", "low_limit"],
        )
        if end == now:
            high_limit = cache._security_.hget("high_low_limit", f"{code}.high_limit")
            low_limit = cache._security_.hget("high_low_limit", f"{code}.low_limit")
            items = [
                {
                    "code": code,
                    "frame": now,
                    "frame_type": FrameType.DAY.to_int(),
                    "high_limit": high_limit,
                    "low_limit": low_limit,
                }
            ]
            df1 = pd.DataFrame(
                items,
                columns=["code", "frame", "frame_type", "high_limit", "low_limit"],
            )
        if df1.empty:
            if df.empty:
                return df.to_numpy()
            else:
                stock = Stock(code)
                if code.startswith("300"):
                    if stock.display_name.startswith("ST"):
                        percent = 0.1
                    else:
                        percent = 0.2
                else:
                    if stock.display_name.startswith("ST"):
                        percent = 0.05
                    else:
                        percent = 0.1
                close = df.iloc[1, 5]
                high_limit = close + close * percent
                low_limit = close - close * percent
                s = pd.Series(
                    {
                        "code": code,
                        "frame": now,
                        "frame_type": FrameType.DAY.to_int(),
                        "high_limit": high_limit,
                        "low_limit": low_limit,
                    }
                )
                df.append(s)
        else:
            df = pd.concat(df, df1)
        df = df[["code", "frame", "frame_type", "high_limit", "low_limit"]]
        dtypes = [
            ("code", "O"),
            ("frame", "O"),
            ("frame_type", "O"),
            ("high_limit", "f4"),
            ("low_limit", "f4"),
        ]
        return np.array(np.rec.fromrecords(df.values), dtype=dtypes)
