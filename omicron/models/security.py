#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors:

"""
import datetime
import logging
import re

import arrow
import numpy as np
from arrow import Arrow
from omega.remote.fetchquotes import FetchQuotes

from ..core.timeframe import tf
from ..core.types import SecurityType, MarketType, FrameType
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
    def display_name(self) -> datetime.date:
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

    async def load_bars(self, start: Arrow, offset: int, frame_type: FrameType,
                        fq=True) -> np.ndarray:
        self._bars = None
        start = tf.shift(start, 0, frame_type)
        exclude_edge = -1 if offset > 0 else 1
        end = tf.shift(start, offset + exclude_edge, frame_type)
        start, end = (start, end) if start < end else (end, start)

        offset = abs(offset)

        head, tail = await security_cache.get_bars_range(self.code, frame_type)

        if not all([head, tail]):
            # not cached at all, ensure cache pointers are clear
            await security_cache.clear_bars_range(self.code, frame_type)

            self._bars = await FetchQuotes(self.code, end, offset, frame_type).invoke()
            return self.qfq() if fq else self._bars

        if start < head:
            n = tf.count_frames(start, head, frame_type)
            if n > 0:
                await FetchQuotes(self.code, tf.shift(head, -1, frame_type), n,
                                  frame_type).invoke()

        if end > tail:
            n = tf.count_frames(tail, end, frame_type)
            if n > 0:
                await FetchQuotes(self.code, end, n, frame_type).invoke()

        # now all bars in [start, end] should exist in cache
        self._bars = await security_cache.get_bars(self.code, end, offset, frame_type)
        return self.qfq() if fq else self._bars
