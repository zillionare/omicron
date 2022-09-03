#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
在apscheduler.triggers的基础上提供了FrameTrigger和IntervalTrigger，使得它们只在交易日（或者
基于交易日+延时）时激发。
"""

import datetime
import logging
import re
from typing import Optional, Union

import pytz
import tzlocal
from apscheduler.triggers.base import BaseTrigger
from coretypes import FrameType

from omicron.models.timeframe import TimeFrame

logger = logging.getLogger(__name__)


class FrameTrigger(BaseTrigger):
    """
    A cron like trigger fires on each valid Frame
    """

    def __init__(self, frame_type: Union[str, FrameType], jitter: str = None):
        """构造函数

        jitter的格式用正则式表达为`r"([-]?)(\\d+)([mshd])"`，其中第一组为符号，'-'表示提前；
        第二组为数字，第三组为单位，可以为`m`(分钟), `s`(秒), `h`（小时）,`d`(天)。

        下面的示例构造了一个只在交易日，每30分钟触发一次，每次提前15秒触的trigger。即它的触发时
        间是每个交易日的09:29:45, 09:59:45, ...

        Examples:
            >>> FrameTrigger(FrameType.MIN30, '-15s') # doctest: +ELLIPSIS
            <omicron.core.triggers.FrameTrigger object at 0x...>

        Args:
            frame_type:
            jitter: 单位秒。其中offset必须在一个FrameType的长度以内
        """
        self.frame_type = FrameType(frame_type)
        if jitter is None:
            _jitter = 0
        else:
            matched = re.match(r"([-]?)(\d+)([mshd])", jitter)
            if matched is None:  # pragma: no cover
                raise ValueError(
                    "malformed. jitter should be [-](number)(unit), "
                    "for example, -30m, or 30s"
                )
            sign, num, unit = matched.groups()
            num = int(num)
            if unit.lower() == "m":
                _jitter = 60 * num
            elif unit.lower() == "s":
                _jitter = num
            elif unit.lower() == "h":
                _jitter = 3600 * num
            elif unit.lower() == "d":
                _jitter = 3600 * 24 * num
            else:  # pragma: no cover
                raise ValueError("bad time unit. only s,h,m,d is acceptable")

            if sign == "-":
                _jitter = -_jitter

        self.jitter = datetime.timedelta(seconds=_jitter)
        if (
            frame_type == FrameType.MIN1
            and abs(_jitter) >= 60
            or frame_type == FrameType.MIN5
            and abs(_jitter) >= 300
            or frame_type == FrameType.MIN15
            and abs(_jitter) >= 900
            or frame_type == FrameType.MIN30
            and abs(_jitter) >= 1800
            or frame_type == FrameType.MIN60
            and abs(_jitter) >= 3600
            or frame_type == FrameType.DAY
            and abs(_jitter) >= 24 * 3600
            # it's still not allowed if offset > week, month, etc. Would anybody
            # really specify an offset longer than that?
        ):
            raise ValueError("offset must be less than frame length")

    def __str__(self):
        return f"{self.__class__.__name__}:{self.frame_type.value}:{self.jitter}"

    def get_next_fire_time(
        self,
        previous_fire_time: Union[datetime.date, datetime.datetime],
        now: Union[datetime.date, datetime.datetime],
    ):
        """"""
        ft = self.frame_type

        # `now` is timezone aware, while ceiling isn't
        now = now.replace(tzinfo=None)
        next_tick = now
        next_frame = TimeFrame.ceiling(now, ft)
        while next_tick <= now:
            if ft in TimeFrame.day_level_frames:
                next_tick = TimeFrame.combine_time(next_frame, 15) + self.jitter
            else:
                next_tick = next_frame + self.jitter

            if next_tick > now:
                tz = tzlocal.get_localzone()
                return next_tick.astimezone(tz)
            else:
                next_frame = TimeFrame.shift(next_frame, 1, ft)


class TradeTimeIntervalTrigger(BaseTrigger):
    """只在交易时间触发的固定间隔的trigger"""

    def __init__(self, interval: str):
        """构造函数

        interval的格式用正则表达式表示为 `r"(\\d+)([mshd])"` 。其中第一组为数字，第二组为单位。有效的
        `interval`如 1 ，表示每1小时触发一次，则该触发器将在交易日的10:30, 11:30, 14:00和
        15：00各触发一次

        Args:
            interval : [description]

        Raises:
            ValueError: [description]
        """
        matched = re.match(r"(\d+)([mshd])", interval)
        if matched is None:
            raise ValueError(f"malform interval {interval}")

        interval, unit = matched.groups()
        interval = int(interval)
        unit = unit.lower()
        if unit == "s":
            self.interval = datetime.timedelta(seconds=interval)
        elif unit == "m":
            self.interval = datetime.timedelta(minutes=interval)
        elif unit == "h":
            self.interval = datetime.timedelta(hours=interval)
        elif unit == "d":
            self.interval = datetime.timedelta(days=interval)
        else:
            self.interval = datetime.timedelta(seconds=interval)

    def __str__(self):
        return f"{self.__class__.__name__}:{self.interval.seconds}"

    def get_next_fire_time(
        self,
        previous_fire_time: Optional[datetime.datetime],
        now: Optional[datetime.datetime],
    ):
        """"""
        if previous_fire_time is not None:
            fire_time = previous_fire_time + self.interval
        else:
            fire_time = now

        if TimeFrame.date2int(fire_time.date()) not in TimeFrame.day_frames:
            ft = TimeFrame.day_shift(now, 1)
            fire_time = datetime.datetime(
                ft.year, ft.month, ft.day, 9, 30, tzinfo=fire_time.tzinfo
            )
            return fire_time

        minutes = fire_time.hour * 60 + fire_time.minute

        if minutes < 570:
            fire_time = fire_time.replace(hour=9, minute=30, second=0, microsecond=0)
        elif 690 < minutes < 780:
            fire_time = fire_time.replace(hour=13, minute=0, second=0, microsecond=0)
        elif minutes > 900:
            ft = TimeFrame.day_shift(fire_time, 1)
            fire_time = datetime.datetime(
                ft.year, ft.month, ft.day, 9, 30, tzinfo=fire_time.tzinfo
            )

        return fire_time
