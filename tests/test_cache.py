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

cfg = cfg4py.get_instance()


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

    async def test_get_bars_raw_data(self):
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
        data = await cache.get_bars_raw_data(
            "000001.XSHE", datetime.datetime(2020, 5, 11, 10, 30), 2, FrameType.MIN30
        )
        exp = b"0.90 0.91 0.92 0.89 10.0 1000.00 100.000.80 0.81 0.82 0.79 10.0 1000.00 101.00"
        self.assertEqual(exp, data)

    async def test_calendar_crud(self):
        await cache.save_calendar("day_frames", tf.day_frames.tolist())
        actual = await cache.load_calendar("day_frames")

        self.assertListEqual(tf.day_frames.tolist(), actual)
