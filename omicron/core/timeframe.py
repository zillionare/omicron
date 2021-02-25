#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import itertools
import logging
from typing import List, Optional, Union

import arrow
import numpy as np
from arrow import Arrow
from dateutil import tz

import omicron.core.accelerate as accl
from omicron.config import calendar
from omicron.core.types import Frame, FrameType

logger = logging.getLogger(__file__)


class TimeFrame:
    _tz = tz.gettz("Asia/Shanghai")
    back_test_mode = False
    _now: Optional[Arrow] = None
    minute_level_frames = [
        FrameType.MIN1,
        FrameType.MIN5,
        FrameType.MIN15,
        FrameType.MIN30,
        FrameType.MIN60,
    ]
    day_level_frames = [FrameType.DAY, FrameType.WEEK, FrameType.MONTH, FrameType.YEAR]

    ticks = {
        FrameType.MIN1: [i for i in itertools.chain(range(571, 691), range(781, 901))],
        FrameType.MIN5: [
            i for i in itertools.chain(range(575, 695, 5), range(785, 905, 5))
        ],
        FrameType.MIN15: [
            i for i in itertools.chain(range(585, 705, 15), range(795, 915, 15))
        ],
        FrameType.MIN30: [
            int(s[:2]) * 60 + int(s[2:])
            for s in ["1000", "1030", "1100", "1130", "1330", "1400", "1430", "1500"]
        ],
        FrameType.MIN60: [
            int(s[:2]) * 60 + int(s[2:]) for s in ["1030", "1130", "1400", "1500"]
        ],
    }

    day_frames = np.array(calendar.day_frames)
    week_frames = np.array(calendar.week_frames)
    month_frames = np.array(calendar.month_frames)

    @classmethod
    async def update_calendar(cls):
        """"""
        from ..dal import security_cache

        for name in ["day_frames", "week_frames", "month_frames"]:
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
        return datetime.datetime(
            int(s[:4]),
            int(s[4:6]),
            int(s[6:8]),
            int(s[8:10]),
            int(s[10:12]),
            tzinfo=cls._tz,
        )

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
    def date2int(cls, d: Union[datetime.datetime, datetime.date, Arrow]) -> int:
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
    def get_ticks(cls, frame_type: FrameType) -> Union[List, np.array]:
        if frame_type in cls.minute_level_frames:
            return cls.ticks[frame_type]

        if frame_type == FrameType.DAY:
            return cls.day_frames
        elif frame_type == FrameType.WEEK:
            return cls.week_frames
        elif frame_type == FrameType.MONTH:
            return cls.month_frames
        else:
            raise ValueError(f"{frame_type} not supported!")

    @classmethod
    def shift(
        cls,
        moment: Union[Arrow, datetime.date, datetime.datetime],
        n: int,
        frame_type: FrameType,
    ) -> Union[datetime.date, datetime.datetime]:
        """
        将指定的moment移动N个位置。当N为负数时，意味着向前移动；当N为正数时，意味着向后移动。
        如果n为零，意味着移动到最接近的一个已结束的frame。

        如果moment没有对齐到frame_type对应的时间，将首先进行对齐。
        Args:
            moment:
            n:
            frame_type:

        Returns:

        """
        if frame_type == FrameType.DAY:
            return cls.day_shift(moment, n)

        elif frame_type == FrameType.WEEK:
            return cls.week_shift(moment, n)
        elif frame_type == FrameType.MONTH:
            return cls.month_shift(moment, n)
        elif frame_type in [
            FrameType.MIN1,
            FrameType.MIN5,
            FrameType.MIN15,
            FrameType.MIN30,
            FrameType.MIN60,
        ]:
            tm = moment.hour * 60 + moment.minute

            new_tick_pos = cls.ticks[frame_type].index(tm) + n
            days = new_tick_pos // len(cls.ticks[frame_type])
            min_part = new_tick_pos % len(cls.ticks[frame_type])

            date_part = cls.day_shift(moment.date(), days)
            minutes = cls.ticks[frame_type][min_part]
            h, m = minutes // 60, minutes % 60
            return datetime.datetime(
                date_part.year, date_part.month, date_part.day, h, m, tzinfo=cls._tz
            )
        else:
            raise ValueError(f"{frame_type} is not supported.")

    @classmethod
    def count_day_frames(
        cls, start: Union[datetime.date, Arrow], end: Union[datetime.date, Arrow]
    ) -> int:
        """
        calc trade days between start and end in close-to-close way. if start == end,
        this will returns 1. Both start/end will be aligned to open trade day before
        calculation.

        args:
            start:
            end:
        """
        start = cls.date2int(start)
        end = cls.date2int(end)
        return int(accl.count_between(cls.day_frames, start, end))

    @classmethod
    def count_week_frames(cls, start: datetime.date, end: datetime.date) -> int:
        """
        calc trade weeks between start and end in close-to-close way. Both start and
        end will be aligned to open trade day before calculation. After that, if start
         == end, this will returns 1

        args:
            start:
            end:
        """
        start = cls.date2int(start)
        end = cls.date2int(end)
        return int(accl.count_between(cls.week_frames, start, end))

    @classmethod
    def count_month_frames(cls, start: datetime.date, end: datetime.date) -> int:
        """
        calc trade months between start and end date in close-to-close way. Both
        start and end will be aligned to open trade day before calculation. After that,
        if start == end, this will returns 1.

        Args:
            start:
            end:

        Returns:

        """
        start = cls.date2int(start)
        end = cls.date2int(end)

        return int(accl.count_between(cls.month_frames, start, end))

    @classmethod
    def count_frames(
        cls,
        start: Union[datetime.date, datetime.datetime, Arrow],
        end: Union[datetime.date, datetime.datetime, Arrow],
        frame_type,
    ) -> int:
        if frame_type == FrameType.DAY:
            return cls.count_day_frames(start, end)
        elif frame_type == FrameType.WEEK:
            return cls.count_week_frames(start, end)
        elif frame_type == FrameType.MONTH:
            return cls.count_month_frames(start, end)
        elif frame_type in [
            FrameType.MIN1,
            FrameType.MIN5,
            FrameType.MIN15,
            FrameType.MIN30,
            FrameType.MIN60,
        ]:
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
    def is_trade_day(cls, dt: Union[datetime.date, datetime.datetime, Arrow]) -> bool:
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

    def floor(self, moment: Frame, frame_type: FrameType) -> Frame:
        """
        根据frame_type,将moment对齐到最接近的上一个frame。用以将类似于10:37这样的时间处理到
        10：30（如果对应的frame_type是FrameType.MIN30)

        Examples:
            see unittest

        Args:
            moment:
            frame_type:

        Returns:

        """
        if frame_type in tf.minute_level_frames:
            tm, day_offset = accl.minute_frames_floor(
                self.ticks[frame_type], moment.hour * 60 + moment.minute
            )
            h, m = tm // 60, tm % 60
            if tf.day_shift(moment, 0) < moment.date() or day_offset == -1:
                h = 15
                m = 0
                new_day = tf.day_shift(moment, day_offset)
            else:
                new_day = moment.date()
            return datetime.datetime(
                new_day.year, new_day.month, new_day.day, h, m, tzinfo=moment.tzinfo
            )

        if type(moment) == datetime.date:
            moment = datetime.datetime(moment.year, moment.month, moment.day, 15)

        if moment.hour * 60 + moment.minute < 900:
            moment = self.day_shift(moment, -1)

        day = tf.date2int(moment)
        if frame_type == FrameType.DAY:
            arr = tf.day_frames
        elif frame_type == FrameType.WEEK:
            arr = tf.week_frames
        elif frame_type == FrameType.MONTH:
            arr = tf.month_frames
        else:
            raise ValueError(f"frame type {frame_type} not supported.")

        floored = accl.floor(arr, day)
        return tf.int2date(floored)

    @classmethod
    def last_frame(
        cls, day: Union[str, Arrow, datetime.date], frame_type: FrameType
    ) -> Union[datetime.date, datetime.datetime]:
        """
        获取指定日期的结束frame。注意这个frame可能位于将来。
        Args:
            day:
            frame_type:

        Returns:

        """
        if isinstance(day, str):
            day = cls.date2int(arrow.get(day).date())
        elif isinstance(day, Arrow) or isinstance(day, datetime.datetime):
            day = cls.date2int(day.date())
        elif isinstance(day, datetime.date):
            day = cls.date2int(day)
        else:
            raise TypeError(f"{type(day)} is not supported.")

        if frame_type == FrameType.DAY:
            raise ValueError("calling last_frame on FrameType.DAY is meaningless.")
        elif frame_type == FrameType.WEEK:
            ceil_day = cls.week_frames[cls.week_frames >= day][0]
            return cls.int2date(ceil_day)
        elif frame_type == FrameType.MONTH:
            ceil_day = cls.month_frames[cls.month_frames >= day][0]
            return cls.int2date(ceil_day)
        elif frame_type in cls.minute_level_frames:
            last_close_day = cls.day_frames[cls.day_frames <= day][-1]
            day = cls.int2date(last_close_day)
            return datetime.datetime(
                day.year, day.month, day.day, hour=15, minute=0, tzinfo=cls._tz
            )
        else:
            raise ValueError(f"{frame_type} not supported")

    @classmethod
    def frame_len(cls, frame_type: FrameType):
        """返回以分钟为单位的frame长度。

        对日线以上级别没有意义，但会返回240
        Args:
            frame_type:

        Returns:

        """

        if frame_type == FrameType.MIN1:
            return 1
        elif frame_type == FrameType.MIN5:
            return 5
        elif frame_type == FrameType.MIN15:
            return 15
        elif frame_type == FrameType.MIN30:
            return 30
        elif frame_type == FrameType.MIN60:
            return 60
        else:
            return 240

    @classmethod
    def first_frame(
        cls, day: Union[str, Arrow, datetime.date], frame_type: FrameType
    ) -> Union[datetime.date, datetime.datetime]:
        """
        获取指定日期的起始的frame。
        Args:
            day (Any):
            frame_type (FrameType):

        Returns:

        """
        if isinstance(day, str):
            day = arrow.get(day).date()
        elif isinstance(day, Arrow) or isinstance(day, datetime.datetime):
            day = day.date()
        elif day is not None:
            raise TypeError(f"{type(day)} is not supported.")

        if frame_type == FrameType.DAY:
            return cls.day_shift(day, 0)

        day = cls.date2int(day)
        if frame_type == FrameType.WEEK:
            floor_day = cls.week_frames[cls.week_frames <= day][-1]
            return cls.int2date(floor_day)
        elif frame_type == FrameType.MONTH:
            floor_day = cls.month_frames[cls.month_frames <= day][-1]
            return cls.int2date(floor_day)
        elif frame_type == FrameType.MIN1:
            floor_day = cls.day_frames[cls.day_frames <= day][-1]
            day = cls.int2date(floor_day)
            naive = datetime.datetime(day.year, day.month, day.day, hour=9, minute=31)
            return cls._tz.localize(naive)
        elif frame_type == FrameType.MIN5:
            floor_day = cls.day_frames[cls.day_frames <= day][-1]
            day = cls.int2date(floor_day)
            naive = datetime.datetime(day.year, day.month, day.day, hour=9, minute=35)
            return cls._tz.localize(naive)
        elif frame_type == FrameType.MIN15:
            floor_day = cls.day_frames[cls.day_frames <= day][-1]
            day = cls.int2date(floor_day)
            naive = datetime.datetime(day.year, day.month, day.day, hour=9, minute=45)
            return cls._tz.localize(naive)
        elif frame_type == FrameType.MIN30:
            floor_day = cls.day_frames[cls.day_frames <= day][-1]
            day = cls.int2date(floor_day)
            naive = datetime.datetime(day.year, day.month, day.day, hour=10)
            return cls._tz.localize(naive)
        elif frame_type == FrameType.MIN60:
            floor_day = cls.day_frames[cls.day_frames <= day][-1]
            day = cls.int2date(floor_day)
            naive = datetime.datetime(day.year, day.month, day.day, hour=10, minute=30)
            return cls._tz.localize(naive)
        else:
            raise ValueError(f"{frame_type} not supported")

    @classmethod
    def get_frames(cls, start: Arrow, end: Arrow, frame_type: FrameType) -> List[int]:
        """
        取[start, end]间所有类型为frame_type的frames。
        Args:
            start:
            end:
            frame_type:

        Returns:

        """
        n = cls.count_frames(start, end, frame_type)
        return cls.get_frames_by_count(end, n, frame_type)

    @classmethod
    def get_frames_by_count(
        cls, end: Arrow, n: int, frame_type: FrameType
    ) -> List[int]:
        """
        取以end为结束点,周期为frame_type的n个frame。
        Args:
            end:
            n:
            frame_type:

        Returns:

        """

        if frame_type == FrameType.DAY:
            end = tf.date2int(end)
            pos = np.searchsorted(tf.day_frames, end, side="right")
            return tf.day_frames[max(0, pos - n) : pos]
        elif frame_type == FrameType.WEEK:
            end = tf.date2int(end)
            pos = np.searchsorted(tf.week_frames, end, side="right")
            return tf.week_frames[max(0, pos - n) : pos]
        elif frame_type == FrameType.MONTH:
            end = tf.date2int(end)
            pos = np.searchsorted(tf.month_frames, end, side="right")
            return tf.month_frames[max(0, pos - n) : pos]
        elif frame_type in {
            FrameType.MIN1,
            FrameType.MIN5,
            FrameType.MIN15,
            FrameType.MIN30,
            FrameType.MIN60,
        }:
            n_days = n // len(tf.ticks[frame_type]) + 2
            ticks = tf.ticks[frame_type] * n_days

            days = cls.get_frames_by_count(end, n_days, FrameType.DAY)
            days = np.repeat(days, len(tf.ticks[frame_type]))

            ticks = [
                day * 10000 + int(tm / 60) * 100 + tm % 60
                for day, tm in zip(days, ticks)
            ]

            # list index is much faster than accl.index_sorted
            pos = ticks.index(tf.time2int(end)) + 1

            return ticks[max(0, pos - n) : pos]
        else:
            raise ValueError(f"{frame_type} not support yet")

    def ceiling(self, moment: Frame, frame_type: FrameType):
        """`moment`所在周期(类型由`frame_type`指定）的终止时间

        Example:
            >>> tf.ceiling(datetime.date(2005, 1, 7), FrameType.DAY)
            datetime.date(2005, 1, 7)

            >>> tf.ceiling(datetime.date(2005, 1, 4), FrameType.WEEK)
            datetime.date(2005, 1, 7)

            >>> tf.ceiling(datetime.date(2005,1,7), FrameType.WEEK)
            datetime.date(2005, 1, 7)

            >>> tf.ceiling(datetime.date(2005,1 ,1), FrameType.MONTH)
            datetime.date(2005, 1, 31)

            >>> tf.ceiling(datetime.datetime(2005, 1, 5, 14, 59), FrameType.MIN1)
            datetime.datetime(2005, 1, 5, 14, 59)

        Args:
            moment (datetime.datetime): [description]
            frame_type (FrameType): [description]

        Returns:
            [type]: [description]
        """
        if frame_type in tf.day_level_frames and type(moment) == datetime.datetime:
            moment = moment.date()

        floor = self.floor(moment, frame_type)
        if floor == moment:
            return moment
        elif floor > moment:
            return floor
        else:
            return self.shift(floor, 1, frame_type)

    def combine_time(
        self,
        date: datetime.date,
        hour: int,
        minute: int = 0,
        second: int = 0,
        microsecond: int = 0,
    ) -> datetime.datetime:
        """将日期与时间结合

        如果args只有一位，则必须为datetime.datetime类型
        Args:
            date (datetime.date): [description]
        """
        return datetime.datetime(
            date.year,
            date.month,
            date.day,
            hour,
            minute,
            second,
            microsecond,
            tzinfo=self._tz,
        )

    def replace_date(
        self, dtm: datetime.datetime, dt: datetime.date
    ) -> datetime.datetime:
        """将`dtm`变量的日期更换为`dt`指定的日期

        Args:
            sel ([type]): [description]
            dtm (datetime.datetime): [description]
            dt (datetime.date): [description]

        Returns:
            datetime.datetime: [description]
        """
        return datetime.datetime(
            dt.year, dt.month, dt.day, dtm.hour, dtm.minute, dtm.second, dtm.microsecond
        )


tf = TimeFrame()
__all__ = ["tf"]
