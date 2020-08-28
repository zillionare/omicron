#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors:

"""
import asyncio
import datetime
import logging
import re
from collections import ChainMap
from typing import List, AsyncIterator

import arrow
import numpy as np

from ..core.quotes_fetcher import get_bars, get_bars_batch
from ..core.timeframe import tf
from ..core.types import SecurityType, MarketType, FrameType, Frame
from ..dal import security_cache
from ..models.securities import Securities

logger = logging.getLogger(__name__)


class Security(object):
    def __init__(self, code: str):
        self._code = code

        _, self._display_name, self._name, self._start_date, self._end_date, _type = \
            Securities()[code]
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
        epoch_start = arrow.get('2005-01-04').date()
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
            if s1 == '6':
                _type = SecurityType.STOCK
            elif s3 in {'000', '880', '999'}:
                _type = SecurityType.INDEX
            elif s2 == '51':
                _type = SecurityType.ETF
            elif s3 in {'129', '100', '110', '120'}:
                _type = SecurityType.BOND
            else:
                _type = SecurityType.UNKNOWN

        else:
            if s2 in {'00', '30'}:
                _type = SecurityType.STOCK
            elif s2 == '39':
                _type = SecurityType.INDEX
            elif s2 == '15':
                _type = SecurityType.ETF
            elif s2 in {'10', '11', '12', '13'}:
                _type = SecurityType.BOND
            elif s2 == '20':
                _type = SecurityType.STOCK_B

        return _type

    def __getitem__(self, key):
        return self._bars[key]

    def qfq(self) -> np.ndarray:
        last = self._bars[-1]['factor']
        for field in ['open', 'high', 'low', 'close']:
            self._bars[field] = (self._bars[field] / last) * self._bars['factor']

        return self._bars

    async def load_bars(self, start: Frame, stop: Frame, frame_type: FrameType,
                        fq=True) -> np.ndarray:
        """
        取时间位于[start, stop]之间的行情数据，这里start可以等于stop。取数据的过程中先利用redis
        缓存，如果遇到缓存中不存在的数据，则从quotes_fetcher服务器取
        Args:
            start:
            stop:
            frame_type:
            fq:

        Returns:

        """
        self._bars = None
        start = tf.floor(start, frame_type)
        _stop = tf.floor(stop, frame_type)

        assert (start <= _stop)
        head, tail = await security_cache.get_bars_range(self.code, frame_type)

        if not all([head, tail]):
            # not cached at all, ensure cache pointers are clear
            await security_cache.clear_bars_range(self.code, frame_type)

            n = tf.count_frames(start, _stop, frame_type)
            if stop > _stop:
                self._bars = await get_bars(self.code, stop, n + 1, frame_type)
            else:
                self._bars = await get_bars(self.code, _stop, n, frame_type)

            return self.qfq() if fq else self._bars

        if start < head:
            n = tf.count_frames(start, head, frame_type)
            if n > 0:
                _end = tf.shift(head, -1, frame_type)
                self._bars = await get_bars(self.code, _end, n, frame_type)

        if _stop > tail:
            n = tf.count_frames(tail, _stop, frame_type)
            if n > 0:
                self._bars = await get_bars(self.code, _stop, n, frame_type)

        # now all closed bars in [start, _stop] should exist in cache
        n = tf.count_frames(start, _stop, frame_type)
        self._bars = await security_cache.get_bars(self.code, _stop, n, frame_type)

        if arrow.get(stop) > arrow.get(_stop):
            bars = await get_bars(self.code, stop, 2, frame_type)
            if len(bars) == 2 and bars[0]['frame'] == self._bars[-1]['frame']:
                self._bars = np.append(self._bars, bars[1])

        return self.qfq() if fq else self._bars

    async def price_change(self, start: Frame, end: Frame, frame_type: FrameType,
                           return_max: False):
        bars = await self.load_bars(start, end, frame_type)
        if return_max:
            return np.max(bars['close'][1:]) / bars['close'][0] - 1
        else:
            return bars['close'][-1] / bars['close'][0] - 1

    @classmethod
    async def _load_bars_batch(cls, codes: List[str], end: Frame, n: int,
                               frame_type: FrameType):
        batch = 1000 // n
        tasks = []
        for i in range(0, batch + 1):
            if i * batch > len(codes):
                break

            task = asyncio.create_task(get_bars_batch(codes[i * batch:(i + 1) * batch],
                                                      end, n,
                                                      frame_type))
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        return dict(ChainMap(*results))

    @classmethod
    async def _get_bars(cls, code, start, stop, frame_type):
        sec = Security(code)
        bars = await sec.load_bars(start, stop, frame_type)
        return code, bars

    @classmethod
    async def load_bars_batch(cls, codes: List[str], end: Frame, n: int,
                              frame_type: FrameType)->AsyncIterator:
        closed_frame = tf.floor(end, frame_type)
        start = tf.shift(closed_frame, -n + 1, frame_type)

        load_alone_tasks = [
            asyncio.create_task(cls._get_bars(code, start, closed_frame, frame_type))
            for code in codes
        ]

        if end == closed_frame:
            for fut in asyncio.as_completed(load_alone_tasks):
                rec = await fut
                yield rec
        else:
            recs1 = await asyncio.gather(*load_alone_tasks)
            recs2 = await cls._load_bars_batch(codes, end, 1, frame_type)

            for code, bars in recs1.items():
                _bars = recs2.get(code)
                if _bars is None or len(_bars) != 1:
                    logger.warning("wrong/emtpy records for %s", code)
                    continue

                yield code, np.append(bars, _bars)