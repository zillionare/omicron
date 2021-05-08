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
from dateutil.tz.tz import datetime_exists

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
        """更新日历

        系统内部调用。Omega从数据源获取最新日历后，存入缓存，并通知监听者更新日历。
        """
        from omicron import cache

        for name in ["day_frames", "week_frames", "month_frames"]:
            frames = await cache.load_calendar(name)
            if frames and len(frames):
                setattr(cls, name, np.array(frames))

    @classmethod
    def int2time(cls, tm: int) -> datetime.datetime:
        """将整数表示的时间转换为`datetime`类型表示

        examples:
            >>> tf.int2time(202005011500)
            datetime.datetime(2020, 5, 1, 15, 0, tzinfo=tzfile('/usr/share/zoneinfo/Asia/Shanghai'))

        Args:
            tm: time in YYYYMMDDHHmm format

        Returns:

        """
        s = str(tm)
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
    def time2int(cls, tm: Union[datetime.datetime, Arrow]) -> int:
        """将时间类型转换为整数类型

        tm可以是Arrow类型，也可以是datetime.datetime或者任何其它类型，只要它有year,month...等
        属性
        Examples:
            >>> tf.time2int(datetime.datetime(2020, 5, 1, 15))
            202005011500

        Args:
            tm:

        Returns:

        """
        return int(f"{tm.year:04}{tm.month:02}{tm.day:02}{tm.hour:02}{tm.minute:02}")

    @classmethod
    def date2int(cls, d: Union[datetime.datetime, datetime.date, Arrow]) -> int:
        """将日期转换为整数表示

        在zillionare中，如果要对时间和日期进行持久化操作，我们一般将其转换为int类型

        Examples:
            >>> tf.date2int(datetime.date(2020,5,1))
            20200501

        Args:
            d: date

        Returns:

        """
        return int(f"{d.year:04}{d.month:02}{d.day:02}")

    @classmethod
    def int2date(cls, d: Union[int, str]) -> datetime.date:
        """将数字表示的日期转换成为日期格式

        Examples:
            >>> tf.int2date(20200501)
            datetime.date(2020, 5, 1)

        Args:
            d: YYYYMMDD表示的日期

        Returns:

        """
        s = str(d)
        # it's 8 times faster than arrow.get
        return datetime.date(int(s[:4]), int(s[4:6]), int(s[6:]))

    @classmethod
    def day_shift(cls, start: datetime.date, offset: int) -> datetime.date:
        """对指定日期进行前后移位操作

        如果 n == 0，则返回d对应的交易日（如果是非交易日，则返回刚结束的一个交易日）
        如果 n > 0，则返回d对应的交易日后第 n 个交易日
        如果 n < 0，则返回d对应的交易日前第 n 个交易日

        Examples:
            >>> tf.day_shift(datetime.date(2019,12,13), 0)
            datetime.date(2019, 12, 13)

            >>> tf.day_shift(datetime.date(2019, 12, 15), 0)
            datetime.date(2019, 12, 13)

            >>> tf.day_shift(datetime.date(2019, 12, 15), 1)
            datetime.date(2019, 12, 16)

            >>> tf.day_shift(datetime.date(2019, 12, 13), 1)
            datetime.date(2019, 12, 16)

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
        """对指定日期按周线帧进行前后移位操作

        参考 [omicron.core.timeframe.TimeFrame.day_shift][]
        Examples:
            >>> moment = arrow.get('2020-1-21').date()
            >>> tf.week_shift(moment, 1)
            datetime.date(2020, 1, 23)

            >>> tf.week_shift(moment, 0)
            datetime.date(2020, 1, 17)

            >>> tf.week_shift(moment, -1)
            datetime.date(2020, 1, 10)
        """
        start = cls.date2int(start)
        return cls.int2date(accl.shift(cls.week_frames, start, offset))

    @classmethod
    def month_shift(cls, start: datetime.date, offset: int) -> datetime.date:
        """求`start`所在的月移位后的frame

        本函数首先将`start`对齐，然后进行移位。
        Examples:
            >>> tf.month_shift(arrow.get('2015-2-26').date(), 0)
            datetime.date(2015, 1, 30)

            >>> tf.month_shift(arrow.get('2015-2-27').date(), 0)
            datetime.date(2015, 2, 27)

            >>> tf.month_shift(arrow.get('2015-3-1').date(), 0)
            datetime.date(2015, 2, 27)

            >>> tf.month_shift(arrow.get('2015-3-1').date(), 1)
            datetime.date(2015, 3, 31)

        """
        start = cls.date2int(start)
        return cls.int2date(accl.shift(cls.month_frames, start, offset))

    @classmethod
    def get_ticks(cls, frame_type: FrameType) -> Union[List, np.array]:
        """取月线、周线、日线及各分钟线对应的frame

        对分钟线，返回值仅包含时间，不包含日期（均为整数表示）

        Examples:
            >>> tf.get_ticks(FrameType.MONTH)[:3]
            array([20050131, 20050228, 20050331])

        Args:
            frame_type : [description]

        Raises:
            ValueError: [description]

        Returns:
            [description]
        """
        if frame_type in cls.minute_level_frames:
            return cls.ticks[frame_type]

        if frame_type == FrameType.DAY:
            return cls.day_frames
        elif frame_type == FrameType.WEEK:
            return cls.week_frames
        elif frame_type == FrameType.MONTH:
            return cls.month_frames
        else:  # pragma: no cover
            raise ValueError(f"{frame_type} not supported!")

    @classmethod
    def shift(
        cls,
        moment: Union[Arrow, datetime.date, datetime.datetime],
        n: int,
        frame_type: FrameType,
    ) -> Union[datetime.date, datetime.datetime]:
        """将指定的moment移动N个`frame_type`位置。

        当N为负数时，意味着向前移动；当N为正数时，意味着向后移动。如果n为零，意味着移动到最接近
        的一个已结束的frame。

        如果moment没有对齐到frame_type对应的时间，将首先进行对齐。

        See also:

        - [day_shift][omicron.core.timeframe.TimeFrame.day_shift]
        - [week_shift][omicron.core.timeframe.TimeFrame.week_shift]
        - [month_shift][omicron.core.timeframe.TimeFrame.month_shift]

        Examples:
            >>> tf.shift(datetime.date(2020, 1, 3), 1, FrameType.DAY)
            datetime.date(2020, 1, 6)

            >>> tf.shift(datetime.datetime(2020, 1, 6, 11), 1, FrameType.MIN30)
            datetime.datetime(2020, 1, 6, 11, 30)


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
                date_part.year,
                date_part.month,
                date_part.day,
                h,
                m,
                tzinfo=moment.tzinfo,
            )
        else:  # pragma: no cover
            raise ValueError(f"{frame_type} is not supported.")

    @classmethod
    def count_day_frames(
        cls, start: Union[datetime.date, Arrow], end: Union[datetime.date, Arrow]
    ) -> int:
        """calc trade days between start and end in close-to-close way.

        if start == end, this will returns 1. Both start/end will be aligned to open
        trade day before calculation.

        Examples:
            >>> start = datetime.date(2019, 12, 21)
            >>> end = datetime.date(2019, 12, 21)
            >>> tf.count_day_frames(start, end)
            1

            >>> # non-trade days are removed
            >>> start = datetime.date(2020, 1, 23)
            >>> end = datetime.date(2020, 2, 4)
            >>> tf.count_day_frames(start, end)
            3

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

        for examples, please refer to [count_day_frames][omicron.core.timeframe.TimeFrame.count_day_frames]
        args:
            start:
            end:
        """
        start = cls.date2int(start)
        end = cls.date2int(end)
        return int(accl.count_between(cls.week_frames, start, end))

    @classmethod
    def count_month_frames(cls, start: datetime.date, end: datetime.date) -> int:
        """calc trade months between start and end date in close-to-close way
        Both start and end will be aligned to open trade day before calculation. After
        that, if start == end, this will returns 1.

        For examples, please refer to [count_day_frames][omicron.core.timeframe.TimeFrame.count_day_frames]

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
        """计算start与end之间有多少个周期为frame_type的frames

        See also:

        - [count_day_frames][omicron.core.timeframe.TimeFrame.count_day_frames]
        - [count_week_frames][omicron.core.timeframe.TimeFrame.count_week_frames]
        - [count_month_frames][omicron.core.timeframe.TimeFrame.count_month_frames]

        Args:
            start : [description]
            end : [description]
            frame_type : [description]

        Raises:
            ValueError: 如果frame_type不支持(季线、年线），则会抛出此异常。

        Returns:
            从start到end的帧数
        """
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
        else:  # pragma: no cover
            raise ValueError(f"{frame_type} is not supported yet")

    @classmethod
    def is_trade_day(cls, dt: Union[datetime.date, datetime.datetime, Arrow]) -> bool:
        """判断`dt`是否为交易日

        Examples:
            >>> tf.is_trade_day(arrow.get('2020-1-1'))
            False

        Args:
            dt :

        Returns:
            [description]
        """
        return cls.date2int(dt) in cls.day_frames

    @classmethod
    def is_open_time(cls, tm: Union[datetime.datetime, Arrow] = None) -> bool:
        """判断`tm`指定的时间是否处在交易时间段。

        交易时间段是指集合竞价时间段之外的开盘时间

        Examples:
            >>> tf.is_open_time(arrow.get('2020-1-1 14:59', tzinfo='Asia/Shanghai'))
            False
            >>> tf.is_open_time(arrow.get('2020-1-3 14:59', tzinfo='Asia/Shanghai'))
            True

        Args:
            tm : [description]. Defaults to None.

        Returns:
            [description]
        """
        tm = tm or arrow.now(cls._tz)

        if not cls.is_trade_day(tm):
            return False

        tick = tm.hour * 60 + tm.minute
        return tick in cls.ticks[FrameType.MIN1]

    @classmethod
    def is_opening_call_auction_time(
        cls, tm: Union[Arrow, datetime.datetime] = None
    ) -> bool:
        """判断`tm`指定的时间是否为开盘集合竞价时间

        Args:
            tm : [description]. Defaults to None.

        Returns:
            [description]
        """
        if tm is None:
            tm = cls.now()

        if not cls.is_trade_day(tm):
            return False

        minutes = tm.hour * 60 + tm.minute
        return 9 * 60 + 15 < minutes <= 9 * 60 + 25

    @classmethod
    def is_closing_call_auction_time(
        cls, tm: Union[datetime.datetime, Arrow] = None
    ) -> bool:
        """判断`tm`指定的时间是否为收盘集合竞价时间

        Fixme:
            此处实现有误，收盘集合竞价时间应该还包含上午收盘时间

        Args:
            tm : [description]. Defaults to None.

        Returns:
            [description]
        """
        tm = tm or cls.now()

        if not cls.is_trade_day(tm):
            return False

        minutes = tm.hour * 60 + tm.minute
        return 15 * 60 - 3 <= minutes < 15 * 60

    def floor(self, moment: Frame, frame_type: FrameType) -> Frame:
        """求`moment`在指定的`frame_type`中的下界

        比如，如果`moment`为10:37，则当`frame_type`为30分钟时，对应的上界为10:00

        Examples:
            >>> # 如果moment为日期，则当成已收盘处理
            >>> tf.floor(datetime.date(2005, 1, 7), FrameType.DAY)
            datetime.date(2005, 1, 7)

            >>> # moment指定的时间还未收盘，floor到上一个交易日
            >>> tf.floor(datetime.datetime(2005, 1, 7, 14, 59), FrameType.DAY)
            datetime.date(2005, 1, 6)

            >>> tf.floor(datetime.date(2005, 1, 13), FrameType.WEEK)
            datetime.date(2005, 1, 7)

            >>> tf.floor(datetime.date(2005,2, 27), FrameType.MONTH)
            datetime.date(2005, 1, 31)

            >>> tf.floor(datetime.datetime(2005,1,5,14,59), FrameType.MIN30)
            datetime.datetime(2005, 1, 5, 14, 30)

            >>> tf.floor(datetime.datetime(2005, 1, 5, 14, 59), FrameType.MIN1)
            datetime.datetime(2005, 1, 5, 14, 59)

            >>> tf.floor(arrow.get('2005-1-5 14:59', tzinfo='Asia/Shanghai').datetime, FrameType.MIN1)
            datetime.datetime(2005, 1, 5, 14, 59, tzinfo=tzfile('/usr/share/zoneinfo/Asia/Shanghai'))

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
            moment = datetime.datetime(
                moment.year, moment.month, moment.day, 15, tzinfo=self._tz
            )

        # 如果是交易日，但还未收盘
        if (
            tf.date2int(moment) in self.day_frames
            and moment.hour * 60 + moment.minute < 900
        ):
            moment = self.day_shift(moment, -1)

        day = tf.date2int(moment)
        if frame_type == FrameType.DAY:
            arr = tf.day_frames
        elif frame_type == FrameType.WEEK:
            arr = tf.week_frames
        elif frame_type == FrameType.MONTH:
            arr = tf.month_frames
        else:  # pragma: no cover
            raise ValueError(f"frame type {frame_type} not supported.")

        floored = accl.floor(arr, day)
        return tf.int2date(floored)

    @classmethod
    def last_min_frame(
        cls, day: Union[str, Arrow, datetime.date], frame_type: FrameType
    ) -> Union[datetime.date, datetime.datetime]:
        """获取`day`日周期为`frame_type`的结束frame。

        Example:
            >>> tf.last_min_frame(arrow.get('2020-1-5').date(), FrameType.MIN30)
            datetime.datetime(2020, 1, 3, 15, 0, tzinfo=tzfile('/usr/share/zoneinfo/Asia/Shanghai'))

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

        if frame_type in cls.minute_level_frames:
            last_close_day = cls.day_frames[cls.day_frames <= day][-1]
            day = cls.int2date(last_close_day)
            return datetime.datetime(
                day.year, day.month, day.day, hour=15, minute=0, tzinfo=cls._tz
            )
        else:  # pragma: no cover
            raise ValueError(f"{frame_type} not supported")

    @classmethod
    def frame_len(cls, frame_type: FrameType):
        """返回以分钟为单位的frame长度。

        对日线以上级别没有意义，但会返回240

        Examples:
            >>> tf.frame_len(FrameType.MIN5)
            5

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
    def first_min_frame(
        cls, day: Union[str, Arrow, datetime.date], frame_type: FrameType
    ) -> Union[datetime.date, datetime.datetime]:
        """获取指定日期类型为`frame_type`的`frame`。

        Examples:
            >>> tf.first_min_frame('2019-12-31', FrameType.MIN1)
            datetime.datetime(2019, 12, 31, 9, 31, tzinfo=tzfile('/usr/share/zoneinfo/Asia/Shanghai'))

        Args:
            day:
            frame_type:

        Returns:

        """

        day = cls.date2int(arrow.get(day).date())

        if frame_type == FrameType.MIN1:
            floor_day = cls.day_frames[cls.day_frames <= day][-1]
            day = cls.int2date(floor_day)
            return datetime.datetime(
                day.year, day.month, day.day, hour=9, minute=31, tzinfo=cls._tz
            )
        elif frame_type == FrameType.MIN5:
            floor_day = cls.day_frames[cls.day_frames <= day][-1]
            day = cls.int2date(floor_day)
            return datetime.datetime(
                day.year, day.month, day.day, hour=9, minute=35, tzinfo=cls._tz
            )
        elif frame_type == FrameType.MIN15:
            floor_day = cls.day_frames[cls.day_frames <= day][-1]
            day = cls.int2date(floor_day)
            return datetime.datetime(
                day.year, day.month, day.day, hour=9, minute=45, tzinfo=cls._tz
            )
        elif frame_type == FrameType.MIN30:
            floor_day = cls.day_frames[cls.day_frames <= day][-1]
            day = cls.int2date(floor_day)
            return datetime.datetime(
                day.year, day.month, day.day, hour=10, tzinfo=cls._tz
            )
        elif frame_type == FrameType.MIN60:
            floor_day = cls.day_frames[cls.day_frames <= day][-1]
            day = cls.int2date(floor_day)
            return datetime.datetime(
                day.year, day.month, day.day, hour=10, minute=30, tzinfo=cls._tz
            )
        else:  # pragma: no cover
            raise ValueError(f"{frame_type} not supported")

    @classmethod
    def get_frames(cls, start: Arrow, end: Arrow, frame_type: FrameType) -> List[int]:
        """取[start, end]间所有类型为frame_type的frames

        调用本函数前，请先通过`floor`或者`ceiling`将时间帧对齐到`frame_type`的边界值

        Example:
            >>> start = arrow.get('2020-1-13 10:00', tzinfo='Asia/Shanghai')
            >>> end = arrow.get('2020-1-13 13:30', tzinfo='Asia/Shanghai')
            >>> tf.get_frames(start, end, FrameType.MIN30)
            [202001131000, 202001131030, 202001131100, 202001131130, 202001131330]

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
        """取以end为结束点,周期为frame_type的n个frame

        调用前请将`end`对齐到`frame_type`的边界

        Examples:
            >>> end = arrow.get('2020-1-6 14:30', tzinfo='Asia/Shanghai')
            >>> tf.get_frames_by_count(end, 2, FrameType.MIN30)
            [202001061400, 202001061430]

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
        else:  # pragma: no cover
            raise ValueError(f"{frame_type} not support yet")

    def ceiling(self, moment: Frame, frame_type: FrameType) -> Frame:
        """求`moment`所在类型为`frame_type`周期的上界

        比如`moment`为14:59分，如果`frame_type`为30分钟，则它的上界应该为15:00

        Example:
            >>> tf.ceiling(datetime.date(2005, 1, 7), FrameType.DAY)
            datetime.date(2005, 1, 7)

            >>> tf.ceiling(datetime.date(2005, 1, 4), FrameType.WEEK)
            datetime.date(2005, 1, 7)

            >>> tf.ceiling(datetime.date(2005,1,7), FrameType.WEEK)
            datetime.date(2005, 1, 7)

            >>> tf.ceiling(datetime.date(2005,1 ,1), FrameType.MONTH)
            datetime.date(2005, 1, 31)

            >>> tf.ceiling(datetime.datetime(2005,1,5,14,59), FrameType.MIN30)
            datetime.datetime(2005, 1, 5, 15, 0)

            >>> tf.ceiling(datetime.datetime(2005, 1, 5, 14, 59), FrameType.MIN1)
            datetime.datetime(2005, 1, 5, 14, 59)

            >>> tf.ceiling(arrow.get('2005-1-5 14:59', tzinfo='Asia/Shanghai').datetime, FrameType.MIN1)
            datetime.datetime(2005, 1, 5, 14, 59, tzinfo=tzfile('/usr/share/zoneinfo/Asia/Shanghai'))

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
        tzinfo="Asia/Shanghai",
    ) -> datetime.datetime:
        """用`date`指定的日期与`hour`, `minute`, `second`等参数一起合成新的时间

        Examples:
            >>> tf.combine_time(datetime.date(2020, 1, 1), 14, 30)
            datetime.datetime(2020, 1, 1, 14, 30, tzinfo=tzfile('/usr/share/zoneinfo/Asia/Shanghai'))

        Args:
            date : [description]
            hour : [description]
            minute : [description]. Defaults to 0.
            second : [description]. Defaults to 0.
            microsecond : [description]. Defaults to 0.

        Returns:
            [description]
        """
        return datetime.datetime(
            date.year,
            date.month,
            date.day,
            hour,
            minute,
            second,
            microsecond,
            tzinfo=tz.gettz(tzinfo),
        )

    def replace_date(
        self, dtm: datetime.datetime, dt: datetime.date
    ) -> datetime.datetime:
        """将`dtm`变量的日期更换为`dt`指定的日期

        Example:
            >>> tf.replace_date(arrow.get('2020-1-1 13:49').datetime, datetime.date(2019, 1,1))
            datetime.datetime(2019, 1, 1, 13, 49)

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
