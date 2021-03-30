# -*- coding: utf-8 -*-

import asyncio
import datetime
import logging
import re
from collections import ChainMap
from typing import AsyncIterator, List

import arrow
import numpy as np
import numpy.lib.recfunctions as rfn

import omicron.core.accelerate as accl
from omicron import cache
from omicron.client.quotes_fetcher import get_bars, get_bars_batch
from omicron.core.timeframe import TimeFrame, tf
from omicron.core.types import Frame, FrameType, MarketType, SecurityType
from omicron.models.securities import Securities
from omicron.models.valuation import Valuation

logger = logging.getLogger(__name__)


class Security(object):
    def __init__(self, code: str):
        self._code = code

        (
            _,
            self._display_name,
            self._name,
            self._start_date,
            self._end_date,
            _type,
        ) = Securities()[code]
        self._type = SecurityType(_type)
        self._bars = None

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

    @property
    def bars(self):
        return self._bars

    def to_canonical_code(self, simple_code: str) -> str:
        """
        将简码转换(比如 000001) 转换成为规范码 (i.e., 000001.XSHE)
        Args:
            simple_code:

        Returns:

        """
        if not re.match(r"^\d{6}$", simple_code):
            raise ValueError(f"Bad simple code: {simple_code}")

        if simple_code.startswith("6"):
            return f"{simple_code}.XSHG"
        else:
            return f"{simple_code}.XSHE"

    def days_since_ipo(self):
        """
        获取上市以来经过了多少个交易日
        Returns:

        """
        epoch_start = arrow.get("2005-01-04").date()
        ipo_day = self.ipo_date if self.ipo_date > epoch_start else epoch_start
        return tf.count_day_frames(ipo_day, arrow.now().date())

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

    def __getitem__(self, key):
        return self._bars[key]

    def qfq(self) -> np.ndarray:
        last = self._bars[-1]["factor"]
        for field in ["open", "high", "low", "close"]:
            self._bars[field] = (self._bars[field] / last) * self._bars["factor"]

        return self._bars

    async def load_bars(
        self,
        start: Frame,
        stop: datetime.datetime,
        frame_type: FrameType,
        fq=True,
        turnover=False,
    ) -> np.ndarray:
        """
        加载[`start`, `stop`]间的行情数据到`Security`对象中，并返回行情数据。

        这里`start`可以等于`stop`。

        为加快速度，对分钟级别的turnover数据，均使用当前周期的成交量除以最新报告期的流通股本数,
        注意这样得到的是一个近似值。如果近期有解禁股，则同样的成交量，解禁后的换手率应该小于解
        禁前。
        Args:
            start:
            stop:
            frame_type:
            fq: 是否进行复权处理
            turnover: 是否包含turnover数据。

        Returns:

        """
        self._bars = None
        start = tf.floor(start, frame_type)
        _stop = tf.floor(stop, frame_type)

        assert start <= _stop
        head, tail = await cache.get_bars_range(self.code, frame_type)

        if not all([head, tail]):
            # not cached at all, ensure cache pointers are clear
            await cache.clear_bars_range(self.code, frame_type)

            n = tf.count_frames(start, _stop, frame_type)
            if stop > _stop:
                self._bars = await get_bars(self.code, stop, n + 1, frame_type)
            else:
                self._bars = await get_bars(self.code, _stop, n, frame_type)

            if fq:
                self.qfq()

            if turnover:
                await self._add_turnover(frame_type)

            return self._bars

        if start < head:
            n = tf.count_frames(start, head, frame_type)
            if n > 0:
                _end = tf.shift(head, -1, frame_type)
                self._bars = await get_bars(self.code, _end, n, frame_type)

        if _stop > tail:
            n = tf.count_frames(tail, _stop, frame_type)
            if n > 0:
                await get_bars(self.code, _stop, n, frame_type)

        # now all closed bars in [start, _stop] should exist in cache
        n = tf.count_frames(start, _stop, frame_type)
        self._bars = await cache.get_bars(self.code, _stop, n, frame_type)

        if arrow.get(stop) > arrow.get(_stop):
            bars = await get_bars(self.code, stop, 2, frame_type)
            if len(bars) == 2 and bars[0]["frame"] == self._bars[-1]["frame"]:
                self._bars = np.append(self._bars, bars[1])

        if fq:
            self.qfq()

        if turnover:
            await self._add_turnover(frame_type)

        return self._bars

    async def _add_turnover(self, frame_type: FrameType):
        # 从bars对应的frame中提取天数
        if frame_type in TimeFrame.minute_level_frames:
            dates = sorted(set(map(lambda x: x.date(), self._bars["frame"])))
        else:
            dates = sorted(set(self._bars["frame"]))
        date = dates[-1]
        n = len(dates)

        cc_recs = await Valuation.get_circulating_cap(self.code, date, n)

        # align circulating_cap with self.bars
        if frame_type != FrameType.DAY:
            tmp_bars = accl.numpy_append_fields(
                self._bars,
                "join_key",
                [x.date() for x in self._bars["frame"]],
                dtypes=[("join_key", "O")],
            )
            cc_recs.dtype.names = ["join_key", "circulating_cap"]
            tmp_bars = accl.join_by_left("join_key", tmp_bars, cc_recs)

        else:
            # rename 'date' to frame, so we can align self._bars with circulating_cap
            cc_recs.dtype.names = "frame", "circulating_cap"
            tmp_bars = accl.join_by_left("frame", self._bars, cc_recs)
        # todo: 对非股票类证券，circulating_cap的单位不一定是手（即100股），此处需要调查
        self._bars = rfn.rec_append_fields(
            self._bars,
            "turnover",
            tmp_bars["volume"] / tmp_bars["circulating_cap"] / 100,
            [("<f4")],
        )
        return self._bars

    async def price_change(
        self, start: Frame, end: Frame, frame_type: FrameType, return_max: False
    ):
        bars = await self.load_bars(start, end, frame_type)
        if return_max:
            return np.max(bars["close"][1:]) / bars["close"][0] - 1
        else:
            return bars["close"][-1] / bars["close"][0] - 1

    @classmethod
    async def _load_bars_batch(
        cls, codes: List[str], end: Frame, n: int, frame_type: FrameType
    ):
        batch = 1000 // n
        tasks = []
        for i in range(0, batch + 1):
            if i * batch > len(codes):
                break

            task = asyncio.create_task(
                get_bars_batch(codes[i * batch : (i + 1) * batch], end, n, frame_type)
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        return dict(ChainMap(*results))

    @classmethod
    async def _get_bars(cls, code, start, stop, frame_type):
        sec = Security(code)
        bars = await sec.load_bars(start, stop, frame_type)
        return code, bars

    @classmethod
    async def load_bars_batch(
        cls, codes: List[str], end: Frame, n: int, frame_type: FrameType
    ) -> AsyncIterator:
        """为一批证券品种加载行情数据

        examples:
        ```
        codes = ["000001.XSHE", "000001.XSHG"]

        end = arrow.get("2020-08-27").datetime
        async for code, bars in Security.load_bars_batch(codes, end, 5, FrameType.DAY):
            print(code, bars[-2:])
            self.assertEqual(5, len(bars))
            self.assertEqual(bars[-1]["frame"], end.date())
            if code == "000001.XSHG":
                self.assertAlmostEqual(3350.11, bars[-1]["close"], places=2)
        ```

        Args:
            codes : 证券列表
            end : 结束帧
            n : 周期数
            frame_type : 帧类型

        Returns:
            [description]

        Yields:
            [description]
        """
        assert type(end) in (datetime.date, datetime.datetime)
        closed_frame = tf.floor(end, frame_type)

        if end == closed_frame:
            start = tf.shift(closed_frame, -n + 1, frame_type)

            cached = [
                asyncio.create_task(
                    cls._get_bars(code, start, closed_frame, frame_type)
                )
                for code in codes
            ]
            for fut in asyncio.as_completed(cached):
                rec = await fut
                yield rec
        else:
            start = tf.shift(closed_frame, -n + 2, frame_type)

            cached = [
                asyncio.create_task(
                    cls._get_bars(code, start, closed_frame, frame_type)
                )
                for code in codes
            ]
            recs1 = await asyncio.gather(*cached)
            recs2 = await cls._load_bars_batch(codes, end, 1, frame_type)

            for code, bars in recs1:
                _bars = recs2.get(code)
                if _bars is None or len(_bars) != 1:
                    logger.warning("wrong/emtpy records for %s", code)
                    continue

                yield code, np.append(bars, _bars)
