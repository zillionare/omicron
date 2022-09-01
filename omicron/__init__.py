"""Omicron提供数据持久化、时间（日历、triggers)、行情数据model、基础运算和基础量化因子"""

import logging

import pkg_resources

from omicron.dal.cache import cache
from omicron.models.timeframe import TimeFrame as tf

__version__ = pkg_resources.get_distribution("zillionare-omicron").version
logger = logging.getLogger(__name__)


async def init():
    """初始化Omicron

    初始化influxDB, 缓存等连接， 并加载日历和证券列表

    上述初始化的连接，应该在程序退出时，通过调用`close()`关闭
    """
    global cache

    await cache.init()
    await tf.init()

    from omicron.models.security import Security

    await Security.init()


async def close():
    """关闭与缓存的连接"""

    try:
        await cache.close()
    except Exception as e:  # noqa
        pass


__all__ = ["tf", "cache", "db"]
