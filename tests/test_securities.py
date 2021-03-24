#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import unittest

import omicron
from omicron import cache
from omicron.models.securities import Securities
from omicron.models.security import Security
from tests import init_test_env, start_omega

logger = logging.getLogger(__name__)

cfg = init_test_env()


class SecuritiesTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        """Set up test fixtures, if any."""
        init_test_env()
        self.omega = await start_omega()
        await omicron.init()

    async def asyncTearDown(self):
        """Tear down test fixtures, if any."""
        await omicron.shutdown()
        if self.omega:
            self.omega.kill()

    async def test_000_load(self):
        s = Securities()

        # invalidate cache, then load from remote
        await cache.security.delete("securities")
        await s.load()
        logger.info(s)
        self.assertEqual(s[0]["code"], "000001.XSHE")

        # read from cache
        s.reset()
        await s.load()
        self.assertEqual(s[0]["code"], "000001.XSHE")
        self.assertEqual(s["000001.XSHE"]["display_name"], "平安银行")

    async def test_001_choose(self):
        s = Securities()
        result = s.choose(["stock", "index"])
        self.assertEqual("000001.XSHE", result[0])

        result = s.choose(["stock"], exclude_300=True)
        self.assertTrue(all([not x.startswith("300") for x in result]))

        result = s.choose(["stock"], exclude_st=True)
        for code in result:
            sec = Security(code)
            self.assertTrue(sec.display_name.upper().find("ST") == -1)

        result = s.choose(["stock"], exclude_688=True)
        self.assertTrue(all([not x.startswith("688") for x in result]))

    async def test_choose_cyb(self):
        cyb = Securities().choose_cyb()
        self.assertTrue(len(cyb) > 0)
        self.assertTrue(all([x.startswith("300") for x in cyb]))

    async def test_fuzzy_match(self):
        for query in ["600001", "PFYH", "浦发"]:
            result = Securities().fuzzy_match(query)
            self.assertTrue(len(result) != 0, f"{query}")
