#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors:  

"""
import datetime
import logging
import re
from typing import Union

from apscheduler.triggers.base import BaseTrigger

from omicron.core.timeframe import tf
from omicron.core.types import FrameType

logger = logging.getLogger(__name__)


class FrameTrigger(BaseTrigger):
    """
    A cron like trigger fires on each valid Frame
    """

    def __init__(self, frame_type: Union[str, FrameType], jitter: str = None):
        """
        Args:
            frame_type:
            jitter: in seconds unit, offset must within one frame
        """
        self.frame_type = FrameType(frame_type)
        if jitter is None:
            _jitter = 0
        else:
            matched = re.match(r"([-]?)(\d+)([mshd])", jitter)
            if matched is None:
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
            else:
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
        ft = self.frame_type
        if ft in tf.day_level_frames:
            if previous_fire_time is None:
                frame = tf.floor(now, ft)
            else:
                frame = tf.shift(previous_fire_time, 1, ft)

            frame = datetime.datetime(
                frame.year, frame.month, frame.day, 15, tzinfo=now.tzinfo
            )
            frame += self.jitter

            if frame < now:  # 调整到下一个frame, 否则apscheduler会陷入死循环
                nf = tf.shift(frame, 1, ft)
                frame = frame.replace(year=nf.year, month=nf.month, day=nf.day)

            return frame
        else:
            if previous_fire_time is None:
                frame = tf.shift(tf.floor(now, ft), 1, ft)
            else:
                frame = tf.shift(tf.floor(previous_fire_time, ft), 1, ft)

            frame += self.jitter
            if frame < now:  # 调整到下一个frame, 否则apscheduler会陷入死循环
                frame = tf.shift(frame, 1, ft)

            return frame


class TradeTimeIntervalTrigger(BaseTrigger):
    def __init__(self, interval: str):
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

    def get_next_fire_time(self, previous_fire_time, now):
        if previous_fire_time is not None:
            fire_time = previous_fire_time + self.interval
        else:
            fire_time = now

        if tf.date2int(fire_time.date()) not in tf.day_frames:
            ft = tf.day_shift(now, 1)
            fire_time = datetime.datetime(ft.year, ft.month, ft.day, 9, 30)
            return fire_time

        minutes = fire_time.hour * 60 + fire_time.minute

        if minutes < 570:
            fire_time = fire_time.replace(hour=9, minute=30, second=0, microsecond=0)
        elif 690 < minutes < 780:
            fire_time = fire_time.replace(hour=13, minute=0, second=0, microsecond=0)
        elif minutes > 900:
            ft = tf.day_shift(fire_time, 1)
            fire_time = datetime.datetime(ft.year, ft.month, ft.day, 9, 30)

        return fire_time
