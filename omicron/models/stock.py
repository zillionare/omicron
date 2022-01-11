import datetime
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
        cls, exclude_exit=True, exclude_st=True, exclude_300=False, exclude_688=True
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

        cond = cls._stocks["type"] == "stock"

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
        return self._type

    @staticmethod
    def simplify_code(code) -> str:
        return re.sub(r"\.XSH[EG]", "", code)

    def days_since_ipo(self):
        """
        获取上市以来经过了多少个交易日
        Returns:

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
        for field in ["open", "high", "low", "close"]:
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
        n: int,
        frame_type: FrameType,
        end: Frame = None,
        fq=True,
        closed=True,
        skip_paused=True,
    ) -> np.ndarray:
        """获取最近的`n`个行情数据。

        返回的数据包含以下字段：

        frame, open, high, low, close, volume, amount, high_limit, low_limit, pre_close

        返回数据格式为numpy strucutre array，每一行对应一个bar,可以通过下标访问，如bars['frame'][-1]

        返回的数据是按照时间顺序递增排序的。在遇到停牌的情况时，如果skip_paused未指定为False，则返回数据包含停牌时间段，其`frame`字段有意义，但其它字段，特别是close取值为None。如果指定为True，则返回数据不包含停牌时间段数据。此时返回的数据条数将少于`n`。

        如果系统当前没有到指定时间`end`的数据，将尽最大努力返回数据。调用者可以通过判断最后一条数据的时间是否等于`end`来判断是否获取到了全部数据。

        Args:
            n (int): 记录数
            end (Frame): 截止时间,如果未指明，则取当前时间
            frame_type (FrameType): [description]
            fq (bool, optional): [description]. Defaults to True.
            closed (bool, optional): 是否包含未收盘的数据。 Defaults to True.
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
    async def reset_cache(cls):
        """清除缓存"""
        for ft in cal.minute_level_frames:
            await cache.security.delete(f"bars:{ft.value}:unclosed")
            keys = await cache.security.keys(f"bars:{ft.value}:*")
            if keys:
                await cache.security.delete(*keys)

    @classmethod
    async def _get_cached_bars(
        cls, code: str, frame_type: FrameType, unclosed=True
    ) -> np.ndarray:
        """从缓存中获取指定代码的行情数据

        如果行情数据为日线以上级别，则最多只会返回一条数据（也可能没有）。如果行情数据为分钟级别数据，则一次返回当天已缓存的所有数据。

        args:
            code: the full qualified code of a security or index
            end: the end frame of the bars
            frame_type: use this to decide which store to use
        """
        raw = []
        if frame_type in cal.day_level_frames:
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
            key = f"bars:{frame_type.value}:{code}"
            r1 = await cache.security.lrange(key, 0, -1)
            raw.extend(r1)

            if unclosed:
                key = f"bars:{frame_type.value}:unclosed"
                r2 = await cache.security.hget(key, code)
                raw.append(r2)

        # convert
        frame_convertor = (
            cal.int2time if frame_type in cal.minute_level_frames else cal.int2date
        )

        recs = []
        for raw_rec in raw:
            f, o, h, l, c, v, m, a, hl, ll, pc, fac = raw_rec.split(",")
            recs.append(
                (
                    frame_convertor(f),
                    float(o),
                    float(h),
                    float(l),
                    float(c),
                    float(v),
                    float(m),
                    float(a),
                    float(hl),
                    float(ll),
                    float(pc),
                    float(fac),
                )
            )

        return np.array(recs, dtype=stock_bars_dtype)

    @classmethod
    async def cache_bars(cls, code: str, frame_type: FrameType, bars: np.ndarray):
        """将行情数据缓存"""
        bars = bars.copy()
        # 转换时间为int
        if frame_type in cal.day_level_frames:
            bars["frame"] = [cal.date2int(x) for x in bars["frame"]]
        else:
            bars["frame"] = [cal.time2int(x) for x in bars["frame"]]

        pl = cache.security.pipeline()
        for bar in bars:
            pl.rpush(f"bars:{frame_type.value}:{code}", ",".join(map(str, bar)))
        await pl.execute()

    @classmethod
    async def cache_unclosed_bars(
        cls, code: str, frame_type: FrameType, bars: np.ndarray
    ):
        """将未结束的行情数据缓存"""
        bars = bars.copy()
        # 转换时间为int
        if frame_type in cal.day_level_frames:
            bars["frame"] = [cal.date2int(x) for x in bars["frame"]]
        else:
            bars["frame"] = [cal.time2int(x) for x in bars["frame"]]

        assert len(bars) == 1, "unclosed bars should only have one record"

        await cache.security.hset(
            f"bars:{frame_type.value}:unclosed", code, ",".join(map(str, bars[0]))
        )

    @classmethod
    async def persist_bars(cls, frame_type: FrameType, bars: np.ndarray):
        """将行情数据持久化"""
        df = pd.DataFrame(data=bars, columns=bars.dtype.names)
        df["frame_type"] = frame_type.to_int()
        df.index = df["frame"]
        await influxdb.write("zillionare", df, "stock", ["frame", "frame_type", "code"])
