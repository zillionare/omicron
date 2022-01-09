"""Omicron提供数据持久化、时间（日历、triggers)、行情数据model、基础运算和基础量化因子"""

import logging

import cfg4py
import gino
import pkg_resources

from omicron.dal.cache import cache
from omicron.dal.postgres import db
from omicron.dal.postgres import init as init_db
from omicron.models.calendar import Calendar as cal

__version__ = pkg_resources.get_distribution("zillionare-omicron").version
logger = logging.getLogger(__name__)


async def init(fetcher=None):
    """初始化omicron

    Args:
        fetcher (AbstractQuotesFetcher): 如果不为None,则Omicron会使用这个fetcher
        来获取行情数据，否则使用远程接口。适用于在`omega`中调用`omicron.init`的情况

    Returns:

    """
    global _local_fetcher, cache
    _local_fetcher = fetcher

    await cache.init()
    await cal.init()

    from omicron.models.stock import Stock

    await Stock.init()

    cfg = cfg4py.get_instance()
    if cfg.postgres.enabled:
        await init_db(cfg.postgres.dsn)


async def close():
    """关闭与数据库、缓存的连接"""
    try:
        await db.pop_bind().close()
    except gino.exceptions.UninitializedError:
        pass

    try:
        await cache.close()
    except Exception as e:  # noqa
        pass


def has_db():
    return cfg4py.get_instance().postgres.enabled


__all__ = ["cache", "db"]
