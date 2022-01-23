#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import logging
from typing import Iterable, List, Optional, Tuple, Union

import aioredis
import cfg4py
import numpy as np
from aioredis.commands import Redis
from arrow.arrow import Arrow

from omicron.models.timeframe import TimeFrame

logger = logging.getLogger(__file__)


class RedisCache:
    databases = ["_sys_", "_security_", "_temp_"]

    _security_: Redis
    _sys_: Redis
    _temp_: Redis

    @property
    def security(self) -> Redis:
        return self._security_

    @property
    def sys(self) -> Redis:
        return self._sys_

    @property
    def temp(self) -> Redis:
        return self._temp_

    async def close(self):
        for redis in [self.sys, self.security, self.temp]:
            redis.close()
            await redis.wait_closed()

    async def init(self):
        cfg = cfg4py.get_instance()
        for i, name in enumerate(self.databases):
            db = await aioredis.create_redis_pool(
                cfg.redis.dsn, encoding="utf-8", maxsize=10, db=i
            )
            await db.set("__meta__.database", name)
            setattr(self, name, db)


cache = RedisCache()
__all__ = ["cache"]
