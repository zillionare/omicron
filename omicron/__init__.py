"""Top-level package for omicron."""

__author__ = """Aaron Yang"""
__email__ = 'code@jieyu.ai'
__version__ = '0.1.2'

import logging

logger = logging.getLogger(__name__)


async def init():
    from omicron.dal import cache
    from omicron.models.securities import Securities
    await cache.init()
    sec = Securities()
    await sec.load()
