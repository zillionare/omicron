#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This is a awesome
        python script!"""
import logging
from typing import TYPE_CHECKING

import aioredis
import cfg4py
from aioredis.commands import Redis

if TYPE_CHECKING:
    from ..config.cfg4py_auto_gen import Config

logger = logging.getLogger(__file__)


class RedisCache:
    databases = ['_sys_', '_security_', '_temp_']

    def __init__(self):
        self._security_ = None
        self._sys_ = None
        self._temp_ = None

    @property
    def security(self)->Redis:
        return self._security_

    @property
    def sys(self)->Redis:
        return self._sys_

    @property
    def temp(self)->Redis:
        return self._temp_

    async def sanity_check(self, db):
        pass

    async def init(self, cfg=None):
        cfg: Config = cfg or cfg4py.get_instance()
        for i, name in enumerate(self.databases):
            db = await aioredis.create_redis_pool(cfg.redis.dsn, encoding='utf-8', maxsize=2, db=i)
            await self.sanity_check(db)
            await db.set("__meta__.database", name)
            setattr(self, name, db)

    async def get_securities(self):
        return await self.security.lrange('securities', 0, -1, encoding='utf-8')


