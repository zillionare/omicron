#!/usr/bin/env python
# -*- coding: utf-8 -*-


from enum import Enum


class FrameType(Enum):
    DAY = '1d'
    MIN60 = '60m'
    MIN30 = '30m'
    MIN15 = '15m'
    MIN5 = '5m'
    MIN1 = '1m'
    WEEK = '1w'
    MONTH = '1M'
    QUARTER = '1Q'
    YEAR = '1Y'


class SecurityType(Enum):
    STOCK = 'stock'
    INDEX = 'index'
    ETF = 'etf'
    FUND = 'fund'
    LOF = 'lof'
    FJA = 'fja'
    FJB = 'fjb'
    FUTURES = 'futures'
    BOND = 'bond'
    STOCK_B = 'stock_b'
    UNKNOWN = 'unknown'


class MarketType(Enum):
    XSHG = 'XSHG'
    XSHE = 'XSHE'
