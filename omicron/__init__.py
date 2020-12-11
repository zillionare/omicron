"""Top-level package for omicron."""

import logging

import cfg4py
import gino
import pkg_resources

from .dal.cache import cache
from .dal.postgres import db
from .dal.postgres import init as init_db

__version__ = pkg_resources.get_distribution("zillionare-omicron").version
logger = logging.getLogger(__name__)

# 如果omicron是与omega在同一进程中，则会调用AbstractQuotesFetcher来获取行情数据，否则将使用
# 远程接口来获取数据
_local_fetcher = None


async def init(fetcher=None):
    """初始化omicron

    Args:
        fetcher: instance of AbstractQuotesFetcher。如果不为None,则Omicron会使用这个fetcher
        来获取行情数据，否则使用远程接口。

    Returns:

    """
    # to avoid circular import
    from .models.securities import Securities

    global _local_fetcher, cache
    _local_fetcher = fetcher

    await cache.init()
    secs = Securities()
    await secs.load()

    cfg = cfg4py.get_instance()
    if cfg.postgres.enabled:
        await init_db(cfg.postgres.dsn)


async def shutdown():
    try:
        await db.pop_bind().close()
    except gino.exceptions.UninitializedError:
        pass


def get_local_fetcher():
    global _local_fetcher

    return _local_fetcher


def has_db():
    return cfg4py.get_instance().postgres.enabled


__all__ = ["cache", "db"]
