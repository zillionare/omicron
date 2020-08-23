import asyncio
import logging
import unittest

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from omicron.core.frametrigger import FrameTrigger
from omicron.core.tradetimeintervaltrigger import TradeTimeIntervalTrigger
from omicron.core.lang import async_run
from omicron.core.types import FrameType

logging.basicConfig(level=logging.INFO)


class MyTestCase(unittest.TestCase):
    @async_run
    async def test_something(self):
        trigger = FrameTrigger(FrameType.MIN1,-1)
        sched = AsyncIOScheduler()

        async def say_hi():
            print("hi")

        #sched.add_job(say_hi, trigger=trigger)
        sched.start()
        trigger = TradeTimeIntervalTrigger(3)
        sched.add_job(say_hi, trigger=trigger)

        await asyncio.sleep(5)


if __name__ == '__main__':
    unittest.main()
