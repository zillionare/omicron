import logging
import unittest

import arrow

from omicron.core.triggers import FrameTrigger
from omicron.core.types import FrameType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _datetime(tm: str):
    return arrow.get(tm, tzinfo="Asia/Shanghai").datetime


class TriggersTest(unittest.IsolatedAsyncioTestCase):
    async def test_frame_trigger(self):
        X = [
            # FrameType      jitter  prev   now
            (FrameType.MIN1, "-30s", None, "2020-11-20 14:59:13"),
            (FrameType.MIN1, None, None, "2020-11-20 14:59:13"),
            (FrameType.MIN1, "59s", None, "2020-11-20 14:59:13"),
            (FrameType.MIN1, "-30s", "2020-11-20 13:00:00", "2020-11-20 14:49:01"),
            # now is non-trading day
            (FrameType.MIN1, "-20s", None, "2020-11-21 09:32:03"),
            (FrameType.MIN1, "40s", None, "2020-11-20 16:03:10"),
            # days
            (FrameType.DAY, "-30m", None, "2020-11-20 14:29"),
            (FrameType.DAY, "-30m", None, "2020-11-20 14:31"),
            (FrameType.WEEK, "-30m", None, "2020-11-20 14:29"),
            (FrameType.WEEK, "-30m", None, "2020-11-20 14:31"),
            (FrameType.DAY, "5m", None, "2020-11-21 14:40"),
            (FrameType.DAY, "5m", None, "2020-11-20 14:59"),
            # 下一个交易日处理。如果刚好scheduler在这一刻进行调度，会产生漏一拍的错误
            (FrameType.DAY, "5m", None, "2020-11-20 15:00"),
        ]

        Y = [
            "2020-11-20 14:59:30",
            "2020-11-20 15:00:00",
            "2020-11-20 15:00:59",
            "2020-11-20 14:49:30",
            "2020-11-23 09:30:40",
            "2020-11-23 09:31:40",
            # days
            "2020-11-20 14:30:00",
            "2020-11-23 14:30:00",
            "2020-11-20 14:30:00",
            "2020-11-27 14:30:00",
            "2020-11-23 15:05:00",
            "2020-11-20 15:05:00",
            "2020-11-23 15:05:00",
        ]

        for i in range(11, len(X)):
            logger.info("%s: %s", i, X[i])
            trigger = FrameTrigger(X[i][0], X[i][1])
            next_tick = trigger.get_next_fire_time(None, _datetime(X[i][3]))
            self.assertEqual(_datetime(Y[i]), next_tick)


if __name__ == "__main__":
    unittest.main()
