import logging
import unittest
from unittest.mock import patch

import arrow

import omicron
from omicron.models.funds import Funds
from tests import init_test_env

logger = logging.getLogger(__name__)


class FundsTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.cfg = await init_test_env()

        self.cfg.postgres.enabled = True
        await omicron.init()

    async def asyncTearDown(self) -> None:
        await omicron.close()

    async def test_crud(self):
        date = arrow.get("2021-12-21")

        logger.info("setp 1: fetch from remote")
        recs = await Funds.get("512690", date)
        self.assertEqual(recs[0]["code"], "512690")
