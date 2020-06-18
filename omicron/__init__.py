"""Top-level package for omicron."""

__author__ = """Aaron Yang"""
__email__ = 'code@jieyu.ai'
__version__ = '0.1.2'

import logging

logger = logging.getLogger(__name__)

# 如果omicron是与omega在同一进程中，则会调用AbstractQuotesFetcher来获取行情数据，否则将使用
# 远程接口来获取数据
_local_fetcher = None


async def init(fetcher=None):
    """

    Args:
        fetcher: instance of AbstractQuotesFetcher。如果不为None,则Omicron会使用这个fetcher
        来获取行情数据，否则使用远程接口。

    Returns:

    """
    global _local_fetcher

    from omicron.dal import cache
    from omicron.models.securities import Securities

    _local_fetcher = fetcher
    await cache.init()
    sec = Securities()
    await sec.load()

def get_local_fetcher():
    global _local_fetcher

    return _local_fetcher