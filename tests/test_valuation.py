import logging
import unittest
from unittest import mock
from unittest.mock import patch

import arrow
import cfg4py

import omicron
from omicron.models.valuation import Valuation
from tests import init_test_env

logger = logging.getLogger(__name__)
cfg4py.enable_logging()

cfg = init_test_env()


class ValuationTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        await omicron.init()

    async def asyncTearDown(self) -> None:
        await omicron.shutdown()

    async def test_crud(self):
        await Valuation.truncate()

        date = arrow.get("2020-10-26").date()
        # step 1, fetch from remote, and create/update into database
        logger.info("step 1: fetch from remote")
        with patch("omicron.has_db", return_value=False):
            recs = await Valuation.get("000001.XSHE", date)
            self.assertEqual(recs[0]["code"], "000001.XSHE")

        # step 2, now data should exist in database
        logger.info("step 2: get data from database")
        recs = await Valuation.get("000001.XSHE", date)
        self.assertEqual(recs[0]["code"], "000001.XSHE")

        # step 3, to fetch more than one securities
        date = arrow.get("2020-11-04").date()
        logger.info("step 3: fetch more than one securities")
        with patch("omicron.has_db", return_value=False):
            recs = await Valuation.get(["000001.XSHE", "600000.XSHG"], date)
            self.assertEqual(2, len(recs))
            self.assertEqual("000001.XSHE", recs[0]["code"])

        # step 4: fetch more than one day, from remote, make sure the order is asc
        logger.info("step 4: one sec many day")
        with patch("omicron.has_db", return_value=False):
            recs = await Valuation.get(["000001.XSHE"], date, n=3)
            self.assertEqual(3, len(recs))
            self.assertEqual(date, recs[2]["date"])
            self.assertEqual(arrow.get("2020-11-2").date(), recs[0]["date"])

        # step 5: fetch more than one day, from database, make sure the order is asc
        logger.info("step 5: more than one day")
        recs = await Valuation.get(["000001.XSHE"], date, n=3)
        self.assertEqual(3, len(recs))
        self.assertEqual(date, recs[2]["date"])
        self.assertEqual(arrow.get("2020-11-2").date(), recs[0]["date"])

        # step 6: more than one day, more than one sec
        logger.info("step 6: more than one day, more than one sec")
        recs = await Valuation.get(["000001.XSHE", "600000.XSHG"], date, n=3)
        self.assertEqual(6, len(recs))

    async def test_get_circulating_cap(self):
        sec = "000001.XSHE"
        date = arrow.get("2020-11-4").date()
        recs = await Valuation.get_circulating_cap(sec, date, 3)
        self.assertAlmostEqual(1940575.25, recs[0]["circulating_cap"], places=0)
        self.assertEqual(date, recs[-1]["date"])
