#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors: 

"""
import datetime
import logging
from enum import IntEnum
from typing import Union

import arrow
from apscheduler.triggers.base import BaseTrigger
from apscheduler.triggers.interval import IntervalTrigger

from omicron.core.timeframe import tf
from omicron.core.types import FrameType

logger = logging.getLogger(__name__)


class FrameTrigger(BaseTrigger):
    """
    A trigger only fires at tradetime
    """
    def __init__(self, frame_type:FrameType, offset:int=0):
        """

        Args:
            frame_type:
            offset: in seconds unit, offset must within one frame
        """
        self.frame_type = frame_type
        self.offset = offset
        if (frame_type == FrameType.MIN1 and abs(offset) >= 60 or
            frame_type == FrameType.MIN5 and abs(offset) >= 300 or
            frame_type == FrameType.MIN15 and abs(offset) >= 900 or
            frame_type == FrameType.MIN30 and abs(offset) >= 1800 or
            frame_type == FrameType.MIN60 and abs(offset) >= 3600 or
            frame_type == FrameType.DAY and abs(offset) >= 24 * 3600
            # it's still not allowed if offset > week, month, etc. Would anybody
            # really specify an offset longer than that?
        ):
            raise ValueError("offset must be less than frame length")

    def get_next_fire_time(self,
                           previous_fire_time: Union[datetime.date, datetime.datetime],
                           now: Union[datetime.date, datetime.datetime]):
        frame_type = self.frame_type
        if previous_fire_time is None:
            frame = tf.shift(tf.floor(now, frame_type), 1, frame_type)
        else:
            frame = tf.shift(tf.floor(previous_fire_time, frame_type), 1, frame_type)

        if frame < now:
            frame = tf.shift(frame, 1, frame_type)

        frame += datetime.timedelta(seconds=self.offset)

        return frame

