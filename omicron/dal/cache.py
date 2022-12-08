#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from asyncio import Lock

import aioredis
import cfg4py
from aioredis.client import Redis

logger = logging.getLogger(__file__)


_cache_lock = Lock()


class RedisCache:
    databases = ["_sys_", "_security_", "_temp_", "_feature_"]

    _security_: Redis
    _sys_: Redis
    _temp_: Redis
    _feature_: Redis
    _app_: Redis

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

    @property
    def app(self) -> Redis:
        if self._initialized is False:
            return None
        else:
            return self._app_

    def __init__(self):
        self._initialized = False

    async def close(self):
        global _cache_lock

        async with _cache_lock:
            if self._initialized is False:
                return True

            logger.info("closing redis cache...")
            await self._sys_.close()
            self._sys_ = None
            await self._security_.close()
            self._security_ = None
            await self._temp_.close()
            self._temp_ = None
            await self._feature_.close()
            self._feature_ = None
            await self._app_.close()
            self._app_ = None

            self._initialized = False
            logger.info("redis caches are all closed")

    async def init(self, app: int = 5):
        global _cache_lock

        async with _cache_lock:
            if self._initialized:
                return True

            logger.info("init redis cache...")
            cfg = cfg4py.get_instance()
            for i, name in enumerate(self.databases):
                auto_decode = True
                if name == "_temp_":
                    auto_decode = False
                db = aioredis.from_url(
                    cfg.redis.dsn,
                    encoding="utf-8",
                    decode_responses=auto_decode,
                    max_connections=10,
                    db=i,
                )
                await db.set("__meta__.database", name)
                setattr(self, name, db)

            # init app pool
            if app < 5 or app > 15:
                app = 5
            db = aioredis.from_url(
                cfg.redis.dsn,
                encoding="utf-8",
                decode_responses=True,
                max_connections=10,
                db=app,
            )
            await db.set("__meta__.database", "__app__")
            setattr(self, "_app_", db)

            self._initialized = True
            logger.info("redis cache is inited")

        return True


cache = RedisCache()
__all__ = ["cache"]
