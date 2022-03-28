"""Omicron提供数据持久化、时间（日历、triggers)、行情数据model、基础运算和基础量化因子"""

import importlib
import logging
import os
import pkgutil
from typing import List

import cfg4py
import gino
import pkg_resources

from omicron.dal.cache import cache
from omicron.dal.postgres import db
from omicron.dal.postgres import init as init_db
from omicron.models.timeframe import TimeFrame as tf

__version__ = pkg_resources.get_distribution("zillionare-omicron").version
logger = logging.getLogger(__name__)


async def init():
    """初始化Omicron

    初始化数据库(postgres), influxDB, 缓存等连接， 并加载日历和证券列表

    上述初始化的连接，应该在程序退出时，通过调用`close()`关闭
    """
    global cache

    await cache.init()
    await tf.init()
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


def export(subpackages: List[str]):
    for sub in subpackages:
        path = [os.path.join(os.path.dirname(__file__), sub)]

        for mod_info in pkgutil.walk_packages(path, f"omicron.{sub}."):
            mod = importlib.import_module(mod_info.name)

            try:
                names = mod.__dict__["__all__"]
            except KeyError:
                names = [k for k in mod.__dict__ if not k.startswith("_")]

            globals().update({k: getattr(mod, k) for k in names})


__all__ = ["tf", "cache", "db"]

export(("talib", "extensions"))
