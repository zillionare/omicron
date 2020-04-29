#!/usr/bin/env python

"""Tests for `omicron` package."""
import os
import unittest

import arrow
import cfg4py
from click.testing import CliRunner
from omicron.core.types import FrameType

from omicron import cli
from omicron.core.lang import async_run
from omega.remote.fetchquotes import FetchQuotes
from pyemit import emit
import logging

from omicron.dal import cache

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class TestOmicron(unittest.TestCase):
    """Tests for `omicron` package."""

    @async_run
    async def setUp(self):
        """Set up test fixtures, if any."""
        os.environ[cfg4py.envar] = 'TEST'
        emit._started = False
        home = os.path.dirname(__file__)
        config_path = os.path.join(home, '../omicron/config')

        cfg = cfg4py.init(config_path)
        await cache.init()
        await emit.start(emit.Engine.REDIS, dsn=cfg.redis.dsn, exchange='zillionare-omega')

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_command_line_interface(self):
        """Test the CLI."""
        runner = CliRunner()
        result = runner.invoke(cli.main)
        assert result.exit_code == 0
        assert 'omicron.cli.main' in result.output
        help_result = runner.invoke(cli.main, ['--help'])
        assert help_result.exit_code == 0
        assert '--help  Show this message and exit.' in help_result.output

    @async_run
    async def test_remote_fetch(self):
        logger.info("ensure zillionare-omega server is running!")

        sec = '000001.XSHE'
        end = arrow.get('2020-04-04')
        fq = FetchQuotes(sec, end, 10, FrameType.DAY)
        bars = await fq.invoke()
        self.assertEqual(bars[0]['frame'], arrow.get('2020-03-23').date())
        self.assertEqual(bars[-1]['frame'], arrow.get('2020-04-03').date())
