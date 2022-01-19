#!/usr/bin/env python
# -*- coding: utf-8 -*-
import datetime
from enum import Enum
from typing import Union

Frame = Union[datetime.date, datetime.datetime]


class SecurityType(Enum):
    """支持的证券品种类型定义

    |     类型                   | 值         | 说明    |
    | ------------------------ | --------- | ----- |
    |     SecurityType.STOCK   | stock     | 股票类型  |
    |     SecurityType.INDEX   | index     | 指数类型  |
    |     SecurityType.ETF     | etf       | ETF基金 |
    |     SecurityType.FUND    | fund      | 基金    |
    |     SecurityType.LOF     | lof，LOF基金 |       |
    |     SecurityType.FJA     | fja       | 分级A基金 |
    |     SecurityType.FJB     | fjb       | 分级B基金 |
    |     SecurityType.BOND    | bond      | 债券基金  |
    |     SecurityType.STOCK_B | stock_b   | B股    |
    |     SecurityType.UNKNOWN | unknown   | 未知品种  |
    """

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
    """市场类型。当前支持的类型为上交所`XSHG`和`XSHE`"""

    XSHG = "XSHG"
    XSHE = "XSHE"


stock_bars_dtype = [
    ("frame", "O"),
    ("open", "f4"),
    ("high", "f4"),
    ("low", "f4"),
    ("close", "f4"),
    ("volume", "f8"),
    ("amount", "f8"),
    ("factor", "f4"),
]

bars_with_limit_dtype = [
    ("frame", "O"),
    ("open", "f4"),
    ("high", "f4"),
    ("low", "f4"),
    ("close", "f4"),
    ("volume", "f8"),
    ("amount", "f8"),
    ("high_limit", "f4"),
    ("low_limit", "O"),
    ("factor", "f4"),
    ("code", "O"),
    ("frame_type", "O"),
]
