#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This is a awesome
        python script!"""
import enum
import logging

import cfg4py
import aioredis
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from ..config.cfg4py_auto_gen import Config

logger = logging.getLogger(__file__)


class RedisDB(enum.Enum):
    # stores info for solo system usage, like jobs
    SYS = 0
    SECURITY = 1
    # store daily, weekly, quarterly, yearly bars, if any
    DAY = 10
    MIN1 = 11
    MIN5 = 12
    MIN15 = 13
    MIN30 = 14
    MIN60 = 15


class RedisCache:
    databases = {}

    async def sanity_check(self):
        pass

    async def init(self):
        cfg: Config = cfg4py.get_instance()
        for name in RedisDB.__members__:
            item = getattr(RedisDB, name)
            cache = await aioredis.create_redis_pool(cfg.redis.dsn, encoding='utf-8', maxsize=2, db=item.value)
            await self.sanity_check()
            await cache.set("__meta__.database", item.name)
            self.databases[item.name] = cache

    def get_db(self, database: Union[RedisDB, str]) -> aioredis.Redis:
        if isinstance(database, RedisDB):
            database = database.name
        return self.databases[database]
