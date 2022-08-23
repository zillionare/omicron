#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from asyncio import Lock

import aioredis
import cfg4py
from aioredis.commands import Redis

logger = logging.getLogger(__file__)


_cache_lock = Lock()


class RedisCache:
    databases = ["_sys_", "_security_", "_temp_", "_feature_"]

    _security_: Redis
    _sys_: Redis
    _temp_: Redis
    _feature_: Redis

    _initialized = False

    @property
    def security(self) -> Redis:
        if self._initialized is False:
            return None
        else:
            return self._security_

    @property
    def sys(self) -> Redis:
        if self._initialized is False:
            return None
        else:
            return self._sys_

    @property
    def temp(self) -> Redis:
        if self._initialized is False:
            return None
        else:
            return self._temp_

    @property
    def feature(self) -> Redis:
        if self._initialized is False:
            return None
        else:
            return self._feature_

    def __init__(self):
        self._initialized = False

    async def close(self):
        global _cache_lock

        async with _cache_lock:
            if self._initialized is False:
                return True

            logger.info("closing redis cache...")
            for redis in [self.sys, self.security, self.temp, self.feature]:
                redis.close()
                await redis.wait_closed()

            self._initialized = False
            logger.info("redis caches are all closed")

    async def init(self):
        global _cache_lock

        async with _cache_lock:
            if self._initialized:
                return True

            logger.info("init redis cache...")
            cfg = cfg4py.get_instance()
            for i, name in enumerate(self.databases):
                db = await aioredis.create_redis_pool(
                    cfg.redis.dsn, encoding="utf-8", maxsize=10, db=i
                )
                await db.set("__meta__.database", name)
                setattr(self, name, db)

            self._initialized = True
            logger.info("redis cache is inited")

        return True


cache = RedisCache()
__all__ = ["cache"]
