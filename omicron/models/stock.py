import datetime
from typing import List
from omicron.core.errors import DataNotReadyError
from omicron.core.types import MarketType, SecurityType, Frame, FrameType
from omicron.dal import cache
import numpy as np
import arrow
import re
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
    async def load_stocks(cls):
        secs = await cache.get_securities()
        if len(secs) != 0:
            _stocks = np.array(
                [tuple(x.split(",")) for x in secs], dtype=cls.fileds_type
            )

            cls._stocks = _stocks[(_stocks.type == "stock") | (_stocks.type == "index")]

        raise DataNotReadyError(
            "No securities in cache, make sure you have called omicron.init() first."
        )

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
