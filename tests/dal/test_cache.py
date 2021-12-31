import datetime
import unittest

import arrow
import cfg4py
import numpy as np

import omicron
from omicron import cache
from omicron.core.timeframe import tf
from omicron.core.types import FrameType, bars_dtype
from tests import clear_cache, init_test_env, start_omega
import logging
from unittest import mock


cfg = cfg4py.get_instance()
logger = logging.getLogger(__name__)

class CacheTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        """Set up test fixtures, if any."""
        init_test_env()

        await clear_cache(cfg.redis.dsn)
        self.omega = await start_omega()
        await omicron.init()
        await cache.clear_bars_range("000001.XSHE", FrameType.MIN30)

    async def asyncTearDown(self) -> None:
        await omicron.shutdown()
        if self.omega:
            self.omega.kill()

    async def test_bars_x(self):
        bars = np.array(
            [
                (
                    datetime.datetime(2020, 5, 11, 10),
                    0.90,
                    0.91,
                    0.92,
                    0.89,
                    10,
                    1000,
                    100,
                ),
                (
                    datetime.datetime(2020, 5, 11, 10, 30),
                    0.80,
                    0.81,
                    0.82,
                    0.79,
                    10,
                    1000,
                    101,
                ),
            ],
            dtype=bars_dtype,
        )

        await cache.save_bars("000001.XSHE", bars, FrameType.MIN30)
        head, tail = await cache.get_bars_range("000001.XSHE", FrameType.MIN30)
        self.assertEqual(arrow.get("2020-05-11 10:00", tzinfo=cfg.tz), head)
        self.assertEqual(arrow.get("2020-05-11 10:30", tzinfo=cfg.tz), tail)

        actual = await cache.get_bars("000001.XSHE", tail, 2, FrameType.MIN30)
        self.assertEqual(head, actual["frame"][0])
        self.assertEqual(tail, actual["frame"][1])
        self.assertAlmostEqual(0.90, actual["open"][0], places=2)

    async def test_calendar_crud(self):
        await cache.save_calendar("day_frames", tf.day_frames.tolist())
        actual = await cache.load_calendar("day_frames")

        self.assertListEqual(tf.day_frames.tolist(), actual)

    async def test_set_bars_range(self):
        start = arrow.get("2019-01-01").date()
        end = arrow.get("2021-01-01").date()

        await cache.set_bars_range('000001.XSHE', FrameType.DAY, start, end)

        start_, end_ = await cache.get_bars_range("000001.XSHE", FrameType.DAY)

        self.assertEqual(start, start_)
        self.assertEqual(end, end_)

    async def test_save_bars(self):
        """save bars out or range"""
        bars = np.array(
            [
                (
                    arrow.get('2020-05-11 10:00', tzinfo=cfg.tz).datetime,
                    0.90,
                    0.91,
                    0.92,
                    0.89,
                    10,
                    1000,
                    100,
                ),
                (
                    arrow.get('2020-05-11 10:30', tzinfo=cfg.tz).datetime,
                    0.80,
                    0.81,
                    0.82,
                    0.79,
                    10,
                    1000,
                    101,
                ),
            ],
            dtype=bars_dtype,
        )

        code = "000001.XSHE"

        start = arrow.get("2020-05-11 11:30", tzinfo="Asia/Shanghai")
        end = arrow.get("2020-05-11 15:00", tzinfo="Asia/Shanghai")
        await cache.set_bars_range(code, FrameType.MIN30, start, end)

        log_messages = []
        logger.setLevel(logging.INFO)
        def log_text_capture(msg):
            log_messages.append(msg)

        with mock.patch.object(logger, 'warning', side_effect = log_text_capture):
            await cache.save_bars(code, bars, FrameType.MIN30)

        print(log_messages)



