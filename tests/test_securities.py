#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors:

"""
import logging
import unittest

from pyemit import emit

import omicron
from omicron.core.lang import async_run
from omicron.dal import cache
from omicron.models.securities import Securities
from tests import init_test_env

logger = logging.getLogger(__name__)

cfg = init_test_env()

class TestSecurity(unittest.TestCase):
    """Tests for `omicron` package."""

    @async_run
    async def setUp(self) -> None:
        """Set up test fixtures, if any."""
        await omicron.init(cfg)
        await emit.start(emit.Engine.REDIS, dsn=cfg.redis.dsn)

    def tearDown(self):
        """Tear down test fixtures, if any."""

    @async_run
    async def test_000_load(self):
        s = Securities()

        # invalidate cache, then load from remote
        await cache.security.delete('securities')
        await s.load()
        logger.info(s)
        self.assertEqual(s[0]['code'], '000001.XSHE')

        # read from cache
        s.reset()
        await s.load()
        self.assertEqual(s[0]['code'], '000001.XSHE')
        self.assertEqual(s['000001.XSHE']['display_name'], '平安银行')

    @async_run
    async def test_001_choose(self):
        s = Securities()
        result = s.choose(['stock', 'index'])
