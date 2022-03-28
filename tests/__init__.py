"""Unit test package for omicron."""
import json
import logging
import os
from typing import Union

import aioredis
import arrow
import cfg4py
import numpy as np
from coretypes import FrameType, bars_dtype
from numpy.testing import assert_array_almost_equal, assert_array_equal

import omicron
from omicron.models.timeframe import TimeFrame

cfg = cfg4py.get_instance()
logger = logging.getLogger(__name__)


async def clear_cache(dsn):
    redis = await aioredis.create_redis(dsn)
    await redis.flushall()
    redis.close()
    await redis.wait_closed()


async def set_security_data(redis):
    # set example securities
    stocks = [
        ("000001.XSHE", "平安银行", "PAYH", "1991-04-03", "2200-01-01", "stock"),
        ("000001.XSHG", "上证指数", "SZZS", "1991-07-15", "2200-01-01", "index"),
        ("000406.XSHE", "石油大明", "SYDM", "1996-06-28", "2006-04-20", "stock"),
        ("000005.XSHE", "ST星源", "STXY", "1990-12-10", "2200-01-01", "stock"),
        ("300001.XSHE", "特锐德", "TRD", "2009-10-30", "2200-01-01", "stock"),
        ("600000.XSHG", "浦发银行", "PFYH", "1999-11-10", "2200-01-01", "stock"),
        ("688001.XSHG", "华兴源创", "HXYC", "2019-07-22", "2200-01-01", "stock"),
        ("000007.XSHE", "*ST全新", "*STQX", "1992-04-13", "2200-01-01", "stock"),
    ]

    pl = redis.pipeline()
    key = "security:stock"
    await redis.delete(key)
    for s in stocks:
        pl.rpush(key, ",".join(s))
    await pl.execute()


async def set_calendar_data(redis):
    # set calendar
    os.environ[cfg4py.envar] = "DEV"
    _dir = os.path.dirname(omicron.__file__)
    file = os.path.join(_dir, "config/calendar.json")
    with open(file, "r") as f:
        data = json.load(f)

        for ft, frame_name in zip(
            [
                FrameType.DAY,
                FrameType.WEEK,
                FrameType.MONTH,
                FrameType.QUARTER,
                FrameType.YEAR,
            ],
            [
                "day_frames",
                "week_frames",
                "month_frames",
                "quarter_frames",
                "year_frames",
            ],
        ):
            key = f"calendar:{ft.value}"
            pl = redis.pipeline()
            pl.delete(key)
            frames = data[frame_name]
            pl.rpush(key, *frames)
            await pl.execute()


async def init_test_env():
    os.environ[cfg4py.envar] = "DEV"
    src_dir = os.path.dirname(__file__)
    config_path = os.path.join(src_dir, "../omicron/config")

    root = logging.getLogger()
    root.handlers.clear()
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])

    cfg = cfg4py.init(config_path, False)

    redis = await aioredis.create_redis(cfg.redis.dsn, db=1)

    try:
        await set_calendar_data(redis)
        await set_security_data(redis)
    finally:
        if redis:
            redis.close()
            await redis.wait_closed()

    return cfg


def assert_bars_equal(exp, actual):
    assert_array_equal(exp["frame"], actual["frame"])

    for field, _ in bars_dtype:
        if field == "frame":
            continue

        decimal = 2 if field != "volume" else 0
        # 经过前复权后，成交量可能略有误差
        assert_array_almost_equal(exp[field], actual[field], decimal)


def test_dir():
    # return path to tests/
    home = os.path.dirname(__file__)
    return home


def lines2bars(lines, is_date):
    """将CSV记录转换为Bar对象

    header: date,open,high,low,close,money,volume,factor
    lines: 2022-02-10 10:06:00,16.87,16.89,16.87,16.88,4105065.000000,243200.000000,121.719130

    """
    if isinstance(lines, str):
        lines = [lines]

    def parse_date(x):
        return arrow.get(x).date()

    def parse_naive(x):
        return arrow.get(x).naive

    if is_date:
        convert = parse_date
    else:
        convert = parse_naive

    data = []
    for line in lines:
        fields = line.split(",")
        data.append(
            (
                convert(fields[0]),
                float(fields[1]),
                float(fields[2]),
                float(fields[3]),
                float(fields[4]),
                float(fields[5]),
                float(fields[6]),
                float(fields[7]),
            )
        )

    return np.array(data, dtype=bars_dtype)


def read_csv(fname, start=None, end=None):
    """start, end是行计数，从1开始，以便于与编辑器展示的相一致。
    返回[start, end]之间的行
    """
    path = os.path.join(test_dir(), "data", fname)
    with open(path, "r") as f:
        lines = f.readlines()

    if start is None:
        start = 1  # skip header
    else:
        start -= 1

    if end is None:
        end = len(lines)

    return lines[start:end]


def bars_from_csv(
    code: str, ft: Union[str, FrameType], start_line: int = None, end_line: int = None
):
    ft = FrameType(ft)

    fname = f"{code}.{ft.value}.csv"

    if ft in TimeFrame.minute_level_frames:
        is_date = False
    else:
        is_date = True

    return lines2bars(read_csv(fname, start_line, end_line), is_date)
