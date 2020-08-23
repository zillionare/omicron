#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors: 

"""
import datetime
import logging

from apscheduler.triggers.base import BaseTrigger

from omicron.core.timeframe import tf

logger = logging.getLogger(__name__)


class TradeTimeIntervalTrigger(BaseTrigger):
    def __init__(self, interval: int):
        self.interval = datetime.timedelta(interval)

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
        else:
            ft = tf.day_shift(fire_time, 1)
            fire_time = datetime.datetime(ft.year, ft.month, ft.day, 9, 30)

        return fire_time
