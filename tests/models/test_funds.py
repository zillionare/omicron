import logging
import unittest
from unittest.mock import patch
import arrow
import omicron
from tests import init_test_env, start_omega
from omicron.models.funds import Funds

logger = logging.getLogger(__name__)


class FundsTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.cfg = init_test_env()
        self.omega = await start_omega()

        self.cfg.postgres.enabled = True
        await omicron.init()

    async def asyncTearDown(self) -> None:
        await omicron.shutdown()
        if self.omega:
            self.omega.kill()

    async def test_crud(self):
        await Funds.truncate()

        date = arrow.get("2021-12-21")

        logger.info("setp 1: fetch from remote")
        recs = await Funds.get("512690", date)
        self.assertEqual(recs[0]["code"], "512690")
