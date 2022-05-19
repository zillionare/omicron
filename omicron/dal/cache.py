#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from threading import Lock

import aioredis
import cfg4py
from aioredis.commands import Redis

logger = logging.getLogger(__file__)


_cache_lock = Lock()


class RedisCache:
    databases = ["_sys_", "_security_", "_temp_"]

    _security_: Redis
    _sys_: Redis
    _temp_: Redis

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

    def __init__(self):
        self._initialized = False

    async def close(self):
        global _cache_lock

        try:
            _cache_lock.acquire()
            if self._initialized is False:
                return True

            for redis in [self.sys, self.security, self.temp]:
                redis.close()
                await redis.wait_closed()

            self._initialized = False
        finally:
            _cache_lock.release()

    async def init(self):
        global _cache_lock

        try:
            _cache_lock.acquire()
            if self._initialized:
                return True

            cfg = cfg4py.get_instance()
            for i, name in enumerate(self.databases):
                db = await aioredis.create_redis_pool(
                    cfg.redis.dsn, encoding="utf-8", maxsize=10, db=i
                )
                await db.set("__meta__.database", name)
                setattr(self, name, db)

            self._initialized = True
        finally:
            _cache_lock.release()

        return True


cache = RedisCache()
__all__ = ["cache"]
