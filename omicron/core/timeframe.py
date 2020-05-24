#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import itertools
import logging
from typing import Optional, Union

import arrow
import numpy as np
import pytz
from arrow import Arrow

import omicron.core.accelerate as accl
from omicron.config import calendar
from .types import FrameType

logger = logging.getLogger(__file__)


class TimeFrame:
    _tz = pytz.timezone('Asia/Chongqing')
    back_test_mode = False
    _now: Optional[Arrow] = None
    minute_level_frames = [FrameType.MIN1, FrameType.MIN5, FrameType.MIN15,
                           FrameType.MIN30, FrameType.MIN60]
    day_level_frames = [FrameType.DAY, FrameType.WEEK, FrameType.MONTH, FrameType.YEAR]

    ticks = {
        FrameType.MIN1:
                         [i for i in itertools.chain(range(571, 691), range(781, 901))],
        FrameType.MIN5:
                         [i for i in
                          itertools.chain(range(575, 695, 5), range(785, 905, 5))],
        FrameType.MIN15: [i for i in
                          itertools.chain(range(585, 705, 15), range(795, 915, 15))],
        FrameType.MIN30: [int(s[:2]) * 60 + int(s[2:]) for s in ["1000", "1030", "1100",
                                                                 "1130", "1330", "1400",
                                                                 "1430", "1500"]],
        FrameType.MIN60: [int(s[:2]) * 60 + int(s[2:]) for s in ["1030", "1130",
                                                                 "1400", "1500"]]
    }

    day_frames = np.array(calendar.day_frames)
    week_frames = np.array(calendar.week_frames)
    month_frames = np.array(calendar.month_frames)

    @classmethod
    async def update_calendar(cls):
        """

        """
        from ..dal import security_cache

        for name in ['day_frames', 'week_frames', 'month_frames']:
            frames = await security_cache.load_calendar(name)
            if len(frames):
                setattr(cls, name, np.array(frames))

    @classmethod
    def int2time(cls, num: int) -> datetime.datetime:
        """
        convert special formatted integer like 202005011500 into datetime(2020,5,1,15)
        Args:
            num:

        Returns:

        """
        s = str(num)
        # its 8 times faster than arrow.get()
        return cls._tz.localize(datetime.datetime(int(s[:4]), int(s[4:6]), int(s[6:8]),
                                                  int(s[8:10]),
                                                  int(s[10:12])))

    @classmethod
    def time2int(cls, tm: Arrow) -> int:
        """
        convert datetime into special int format, for example, from datetime(2020, 5,
        1, 15) to 202005011500
        Args:
            tm:

        Returns:

        """
        return int(f"{tm.year:04}{tm.month:02}{tm.day:02}{tm.hour:02}{tm.minute:02}")

    @classmethod
    def date2int(cls, d: datetime.date) -> int:
        """
        convert date into a special formatted int, for example, from date(2020,5,
        1) to 20200501
        Args:
            d:

        Returns:

        """
        return int(f"{d.year:04}{d.month:02}{d.day:02}")

    @classmethod
    def int2date(cls, d: Union[int, str]) -> datetime.date:
        """
        convert a special formatted int to date, for example, from 20200501 to date(
        2020,5,1)
        Args:
            d:

        Returns:

        """
        s = str(d)
        # it's 8 times faster than arrow.get
        return datetime.date(int(s[:4]), int(s[4:6]), int(s[6:]))

    @classmethod
    def day_shift(cls, start: datetime.date, offset: int) -> datetime.date:
        """
        如果 n == 0，则返回d对应的交易日（如果是非交易日，则返回刚结束的一个交易日）
        如果 n > 0，则返回d对应的交易日后第 n 个交易日
        如果 n < 0，则返回d对应的交易日前第 n 个交易日
        Args:
            start: the origin day
            offset: days to shift, can be negative

        Returns:

        """
        # accelerated from 0.12 to 0.07, per 10000 loop, type conversion time included
        start = cls.date2int(start)

        return cls.int2date(accl.shift(cls.day_frames, start, offset))

    @classmethod
    def week_shift(cls, start: datetime.date, offset: int) -> datetime.date:
        """
        返回start对应的那一周结束的日期，这个日期就是那个frame的id
        """
        start = cls.date2int(start)
        return cls.int2date(accl.shift(cls.week_frames, start, offset))

    @classmethod
    def month_shift(cls, start: datetime.date, offset: int) -> datetime.date:
        """
        返回start对应的那一个月的结束日期。这个日期就是那一个frame的id
        """
        start = cls.date2int(start)
        return cls.int2date(accl.shift(cls.month_frames, start, offset))

    @classmethod
    def shift(cls, start: Union[Arrow, datetime.date, datetime.datetime], n: int,
              frame_type: FrameType) -> Union[
        datetime.date, datetime.datetime]:
        if frame_type == FrameType.DAY:
            return cls.day_shift(start, n)

        elif frame_type == FrameType.WEEK:
            return cls.week_shift(start, n)
        elif frame_type == FrameType.MONTH:
            return cls.month_shift(start, n)
        elif frame_type in [FrameType.MIN1, FrameType.MIN5, FrameType.MIN15,
                            FrameType.MIN30, FrameType.MIN60]:
            tm = start.hour * 60 + start.minute
            if tm not in cls.ticks[frame_type]:
                raise ValueError(f"{start} is not aligned with unit {frame_type}")

            new_tick_pos = cls.ticks[frame_type].index(tm) + n
            days = new_tick_pos // len(cls.ticks[frame_type])
            min_part = new_tick_pos % len(cls.ticks[frame_type])

            date_part = cls.day_shift(start.date(), days)
            return cls._tz.localize(datetime.datetime(date_part.year, date_part.month,
                                                      date_part.day) +
                                    datetime.timedelta(
                                        minutes=cls.ticks[frame_type][min_part]))
        else:
            raise ValueError(f"{frame_type} is not supported.")

    @classmethod
    def count_day_frames(cls, start: Union[datetime.date, Arrow],
                         end: Union[datetime.date, Arrow]) -> int:
        """
        calc trade days between start and end in close-to-close way. if start == end,
        this will returns 1. Both start/end will be aligned to open trade day before
        calculation.


        :param start:
        :param end:
        :return:
        """
        start = cls.date2int(start)
        end = cls.date2int(end)
        return accl.count_between(cls.day_frames, start, end)

    @classmethod
    def count_week_frames(cls, start: datetime.date, end: datetime.date) -> int:
        """
        calc trade weeks between start and end in close-to-close way. Both start and
        end will be aligned to open trade day before calculation. After that, if start
         == end, this will returns 1
        :param
        """
        start = cls.date2int(start)
        end = cls.date2int(end)
        return accl.count_between(cls.week_frames, start, end)

    @classmethod
    def count_month_frames(cls, start: datetime.date, end: datetime.date) -> int:
        """
        cacl trade months between start and end date in close-to-close way. Both
        start and end will be aligned to open trade day before calculation. After that,
        if start == end, this will returns 1.
        Args:
            start:
            end:

        Returns:

        """
        start = cls.date2int(start)
        end = cls.date2int(end)

        return accl.count_between(cls.month_frames, start, end)

    @classmethod
    def count_frames(cls, start: Union[datetime.date, datetime.datetime, Arrow],
                     end: Union[datetime.date, datetime.datetime, Arrow],
                     frame_type) -> int:
        if frame_type == FrameType.DAY:
            return cls.count_day_frames(start, end)
        elif frame_type == FrameType.WEEK:
            return cls.count_week_frames(start, end)
        elif frame_type == FrameType.MONTH:
            return cls.count_month_frames(start, end)
        elif frame_type in [FrameType.MIN1, FrameType.MIN5, FrameType.MIN15,
                            FrameType.MIN30, FrameType.MIN60]:
            tm_start = start.hour * 60 + start.minute
            tm_end = end.hour * 60 + end.minute
            days = cls.count_day_frames(start.date(), end.date()) - 1

            tm_start_pos = cls.ticks[frame_type].index(tm_start)
            tm_end_pos = cls.ticks[frame_type].index(tm_end)

            min_bars = tm_end_pos - tm_start_pos + 1

            return days * len(cls.ticks[frame_type]) + min_bars
        else:
            raise ValueError(f"{frame_type} is not supported yet")

    @classmethod
    def set_backtest_mode(cls, now: Arrow):
        cls.back_test_mode = True
        cls._now = now

    @classmethod
    def now(cls) -> Arrow:
        if cls.back_test_mode:
            return cls._now
        else:
            return arrow.now()

    @classmethod
    def is_trade_day(cls, dt: Arrow) -> bool:
        return cls.date2int(dt) in cls.day_frames

    @classmethod
    def is_open_time(cls, tm: Optional[Arrow] = None) -> bool:
        if tm is None:
            tm = cls.now()

        if not cls.is_trade_day(tm):
            return False

        tick = f"{tm.hour:02}{tm.minute:02}"
        return tick in cls.ticks[FrameType.MIN1]

    @classmethod
    def is_opening_call_auction_time(cls, tm: Optional[Arrow] = None) -> bool:
        if tm is None:
            tm = cls.now()

        if not cls.is_trade_day(tm):
            return False

        minutes = tm.hour * 60 + tm.minute
        return 9 * 60 + 15 < minutes < 9 * 60 + 25

    @classmethod
    def is_closing_call_auction_time(cls, tm: Optional[Arrow] = None) -> bool:
        tm = tm or cls.now()

        if not cls.is_trade_day(tm):
            return False

        minutes = tm.hour * 60 + tm.minute
        return 15 * 60 - 3 < minutes < 16 * 60

    @classmethod
    def minutes_left(cls, tm: Arrow) -> int:
        pass

    @classmethod
    def minutes_elapsed(cls, tm: Arrow) -> int:
        pass

    @classmethod
    def round_up(cls, tm: Arrow, frame_type: FrameType) -> Arrow:
        pass

    @classmethod
    def round_down(cls, tm: Arrow, frame_type: FrameType) -> Arrow:
        pass

    @classmethod
    def get_start_frame(cls, frame_type: FrameType) -> Union[datetime.date,
                                                             datetime.datetime]:
        if frame_type == FrameType.DAY:
            return cls.int2date(cls.day_frames[0])
        elif frame_type == FrameType.WEEK:
            return cls.int2date(cls.week_frames[0])
        elif frame_type == FrameType.MONTH:
            return cls.int2date(cls.month_frames[0])
        elif frame_type == FrameType.MIN1:
            day = cls.int2date(cls.day_frames[0])
            return datetime.datetime(day.year, day.month, day.day, hour=9, minute=31)
        elif frame_type == FrameType.MIN5:
            day = cls.int2date(cls.day_frames[0])
            return datetime.datetime(day.year, day.month, day.day, hour=9, minute=35)
        elif frame_type == FrameType.MIN15:
            day = cls.int2date(cls.day_frames[0])
            return datetime.datetime(day.year, day.month, day.day, hour=9, minute=45)
        elif frame_type == FrameType.MIN30:
            day = cls.int2date(cls.day_frames[0])
            return datetime.datetime(day.year, day.month, day.day, hour=10)
        elif frame_type == FrameType.MIN60:
            day = cls.int2date(cls.day_frames[0])
            return datetime.datetime(day.year, day.month, day.day, hour=10, minute=30)
        else:
            raise ValueError(f"{frame_type} not supported")


tf = TimeFrame()
__all__ = ['tf']
