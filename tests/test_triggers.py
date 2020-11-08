import asyncio
import logging
import unittest

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from omicron.core.triggers import FrameTrigger, TradeTimeIntervalTrigger
from omicron.core.types import FrameType

logging.basicConfig(level=logging.INFO)


class TriggersTest(unittest.IsolatedAsyncioTestCase):
    async def test_something(self):
        sched_logger = logging.getLogger("apscheduler")
        sched_logger.setLevel(logging.DEBUG)

        async def say_hi():
            print("hi")

        sched = AsyncIOScheduler()
        sched.start()

        trigger = FrameTrigger(FrameType.MIN1, "-30s")
        sched.add_job(say_hi, trigger=trigger, name=f"min1")

        trigger = FrameTrigger(FrameType.DAY, "1h")
        sched.add_job(say_hi, trigger=trigger, name="day")

        trigger = FrameTrigger(FrameType.WEEK, "-1d")
        sched.add_job(say_hi, trigger=trigger, name="week")

        trigger = TradeTimeIntervalTrigger("3s")
        sched.add_job(say_hi, trigger=trigger)

        await asyncio.sleep(1)


if __name__ == "__main__":
    unittest.main()
