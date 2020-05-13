"""Top-level package for omicron."""

__author__ = """Aaron Yang"""
__email__ = 'code@jieyu.ai'
__version__ = '0.1.0'

from omicron.dal import cache
from omicron.dal import security_cache
from omicron.models.securities import Securities


async def init(cfg):
    await cache.init(cfg)
    sec = Securities()
    await sec.load()
