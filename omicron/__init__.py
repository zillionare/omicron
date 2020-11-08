"""Top-level package for omicron."""

__author__ = """Aaron Yang"""
__email__ = "code@jieyu.ai"
__version__ = "__version__ = '0.2.0'"

import logging

import cfg4py

from .dal.cache import cache
from .dal.postgres import db
from .dal.postgres import init as init_db

logger = logging.getLogger(__name__)

# 如果omicron是与omega在同一进程中，则会调用AbstractQuotesFetcher来获取行情数据，否则将使用
# 远程接口来获取数据
_local_fetcher = None

# 是否配置了数据库（Redis必须配置）
_has_db = False


async def init(fetcher=None):
    """
    Args:
        fetcher: instance of AbstractQuotesFetcher。如果不为None,则Omicron会使用这个fetcher
        来获取行情数据，否则使用远程接口。

    Returns:

    """
    # to avoid circular import
    from .models.securities import Securities

    global _local_fetcher, cache, _has_db
    _local_fetcher = fetcher

    await cache.init()
    sec = Securities()
    await sec.load()

    cfg = cfg4py.get_instance()
    try:
        dsn = cfg.postgres.dsn
    except AttributeError:
        pass
    else:  # user configured valid dsn, then init database connection
        _has_db = True
        await init_db(dsn)


async def shutdown():
    await db.pop_bind().close()


def get_local_fetcher():
    global _local_fetcher

    return _local_fetcher


def has_db():
    return _has_db


__all__ = ["cache", "db"]
