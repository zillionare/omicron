import datetime
import logging
import unittest
from unittest import mock

import aiohttp

from omicron.client.quotes_fetcher import (
    get_bars,
    get_bars_batch,
    get_security_list,
    get_server_version,
)
from omicron.core.types import FrameType
from tests import init_test_env, start_omega

logger = logging.getLogger(__name__)

cfg = init_test_env()


class OmegaClientTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        """Set up test fixtures, if any."""
        init_test_env()
        self.omega = await start_omega()

    async def asyncTearDown(self):
        """Tear down test fixtures, if any."""
        if self.omega:
            self.omega.kill()

    async def test_get_security_list(self):
        secs = await get_security_list()
        expected = ["000001.XSHE", "平安银行", "PAYH", "1991-04-03", "2200-01-01", "stock"]

        self.assertListEqual(expected, secs[0].tolist())

        # test if server is down
        with mock.patch(
            "aiohttp.ClientSession.get", side_effect=aiohttp.ClientConnectionError()
        ):
            self.assertIsNone(await get_security_list())

    async def test_get_bars(self):
        bars = await get_bars(
            "000001.XSHE", datetime.date(2021, 2, 5), 1, FrameType.DAY
        )
        close = 24.93
        date = datetime.date(2021, 2, 5)
        self.assertEqual(date, bars[0]["frame"])
        self.assertAlmostEqual(close, bars[0]["close"], places=2)

        sec = "000001.XSHG"
        ft = FrameType.WEEK
        bars = await get_bars(sec, datetime.date(2021, 2, 9), 1, ft)
        print(bars)
        bars = await get_bars(sec, datetime.date(2021, 2, 9), 1, ft, False)
        print(bars)

    async def test_get_bars_batch(self):
        secs = ["000001.XSHE", "000001.XSHG"]
        end = datetime.date(2021, 2, 8)
        n = 3
        ft = FrameType.DAY
        iu = False

        bars = await get_bars_batch(secs, end, n, ft, iu)
        payh = bars.get("000001.XSHE")
        self.assertEqual(datetime.date(2021, 2, 5), payh["frame"][-1])
        self.assertAlmostEqual(23.48, payh["open"][0], places=2)

        iu = True
        bars = await get_bars_batch(secs, end, n, ft, iu)
        sh = bars.get("000001.XSHG")
        self.assertEqual(datetime.date(2021, 2, 8), sh["frame"][-1])

    async def test_get_server_version(self):
        ver = await get_server_version()
        self.assertTrue(len(ver) > 0)
