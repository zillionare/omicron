import unittest

import arrow
from omega.remote.fetchquotes import FetchQuotes

from omicron.core.lang import async_run
import logging

from omicron.core.types import FrameType

logger = logging.getLogger(__name__)

class MyTestCase(unittest.TestCase):
    @async_run
    async def test_remote_fetch(self):
        logger.info("ensure zillionare-omega server is running!")

        sec = '000001.XSHE'
        end = arrow.get('2020-04-04')
        fq = FetchQuotes(sec, end, 10, FrameType.DAY)
        bars = await fq.invoke()
        self.assertEqual(bars[0]['frame'], arrow.get('2020-03-23').date())
        self.assertEqual(bars[-1]['frame'], arrow.get('2020-04-03').date())


if __name__ == '__main__':
    unittest.main()
