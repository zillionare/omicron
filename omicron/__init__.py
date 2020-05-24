"""Top-level package for omicron."""

__author__ = """Aaron Yang"""
__email__ = 'code@jieyu.ai'
__version__ = '0.1.1'

import logging

logger = logging.getLogger(__name__)


async def init(cfg):
    from omicron.dal import cache
    from omicron.models.securities import Securities
    logger.info("init omicron with %s", cfg)
    await cache.init(cfg)
    sec = Securities()
    await sec.load()
