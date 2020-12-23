#!/usr/bin/env python
# -*- coding: utf-8 -*-
import datetime
from enum import Enum
from typing import Union

Frame = Union[datetime.date, datetime.datetime]


class FrameType(Enum):
    DAY = "1d"
    MIN60 = "60m"
    MIN30 = "30m"
    MIN15 = "15m"
    MIN5 = "5m"
    MIN1 = "1m"
    WEEK = "1w"
    MONTH = "1M"
    QUARTER = "1Q"
    YEAR = "1Y"

    def to_int(self) -> int:
        mapping = {
            FrameType.MIN1: 1,
            FrameType.MIN5: 2,
            FrameType.MIN15: 3,
            FrameType.MIN30: 4,
            FrameType.MIN60: 5,
            FrameType.DAY: 6,
            FrameType.WEEK: 7,
            FrameType.MONTH: 8,
            FrameType.QUARTER: 9,
            FrameType.YEAR: 10,
        }
        return mapping[self]

    @staticmethod
    def from_int(frame_type: int) -> "FrameType":
        mapping = {
            1: FrameType.MIN1,
            2: FrameType.MIN5,
            3: FrameType.MIN15,
            4: FrameType.MIN30,
            5: FrameType.MIN60,
            6: FrameType.DAY,
            7: FrameType.WEEK,
            8: FrameType.MONTH,
            9: FrameType.QUARTER,
            10: FrameType.YEAR,
        }

        return mapping[frame_type]

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.to_int() < other.to_int()
        return NotImplemented

    def __le__(self, other) -> bool:
        if self.__class__ is other.__class__:
            return self.to_int() <= other.to_int()

        return NotImplemented

    def __ge__(self, other) -> bool:
        if self.__class__ is other.__class__:
            return self.to_int() >= other.to_int()

        return NotImplemented

    def __gt__(self, other) -> bool:
        if self.__class__ is other.__class__:
            return self.to_int() > other.to_int()

        return NotImplemented


class SecurityType(Enum):
    STOCK = "stock"
    INDEX = "index"
    ETF = "etf"
    FUND = "fund"
    LOF = "lof"
    FJA = "fja"
    FJB = "fjb"
    FUTURES = "futures"
    BOND = "bond"
    STOCK_B = "stock_b"
    UNKNOWN = "unknown"


class MarketType(Enum):
    XSHG = "XSHG"
    XSHE = "XSHE"


bars_dtype = [
    ("frame", "O"),
    ("open", "f4"),
    ("high", "f4"),
    ("low", "f4"),
    ("close", "f4"),
    ("volume", "f8"),
    ("amount", "f8"),
    ("factor", "f4"),
]
