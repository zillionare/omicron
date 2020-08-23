#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors: 

"""
import datetime
import logging
from typing import Union

from apscheduler.triggers.base import BaseTrigger

from omicron.core.timeframe import tf
from omicron.core.types import FrameType

logger = logging.getLogger(__name__)


class FrameTrigger(BaseTrigger):
    """
    A trigger only fires at tradetime
    """

    def __init__(self, frame_type: Union[str, FrameType], jitter: int = 0):
        """

        Args:
            frame_type:
            jitter: in seconds unit, offset must within one frame
        """
        ft = FrameType(frame_type) if isinstance(frame_type, str) else frame_type
        self.frame_type = ft
        self.jitter = datetime.timedelta(seconds=jitter)
        if (frame_type == FrameType.MIN1 and abs(jitter) >= 60 or
                frame_type == FrameType.MIN5 and abs(jitter) >= 300 or
                frame_type == FrameType.MIN15 and abs(jitter) >= 900 or
                frame_type == FrameType.MIN30 and abs(jitter) >= 1800 or
                frame_type == FrameType.MIN60 and abs(jitter) >= 3600 or
                frame_type == FrameType.DAY and abs(jitter) >= 24 * 3600
                # it's still not allowed if offset > week, month, etc. Would anybody
                # really specify an offset longer than that?
        ):
            raise ValueError("offset must be less than frame length")

    def get_next_fire_time(self,
                           previous_fire_time: Union[datetime.date, datetime.datetime],
                           now: Union[datetime.date, datetime.datetime]):
        ft = self.frame_type
        if ft in tf.day_level_frames:
            if previous_fire_time is None:
                frame = tf.floor(now, ft)
            else:
                frame = tf.shift(previous_fire_time, 1, ft)

            frame = datetime.datetime(frame.year, frame.month, frame.day, 15,
                                      tzinfo=now.tzinfo)
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
