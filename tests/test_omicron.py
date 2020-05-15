#!/usr/bin/env python

"""Tests for `omicron` package."""
import logging
import unittest

import arrow
from click.testing import CliRunner
from omega.remote.fetchquotes import FetchQuotes
from pyemit import emit

from omicron import cli
from omicron.core.lang import async_run
from omicron.core.types import FrameType
from omicron.dal import cache
from tests import init_test_env

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

cfg = init_test_env()

class TestOmicron(unittest.TestCase):
    """Tests for `omicron` package."""

    @async_run
    async def setUp(self):
        """Set up test fixtures, if any."""
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
