#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors:

"""
import logging
from functools import lru_cache

import numpy as np
from ..core.lang import singleton
from ..dal import cache, RedisDB
from omega.remote.fetchsecuritylist import FetchSecurityList

logger = logging.getLogger(__name__)


@singleton
class Securities(object):
    INDEX_XSHE = "399001.XSHE"
    INDEX_XSHG = "000001.XSHG"
    INDEX_CYB = "399006.XSHE"

    _secs = np.array([])
    dtypes = [
        ('code', 'O'),
        ('display_name', 'O'),
        ('name', 'O'),
        ('ipo', 'O'),
        ('end', 'O'),
        ('type', 'O')
    ]

    def __str__(self):
        return f"{len(self._secs)} securities"

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, stride = key.indices(len(self._bars))
            return self._secs[start: stop]
        elif isinstance(key, int):
            return self._secs[key]
        elif isinstance(key, str):
            # assume the key is the security code
            try:
                return self._secs[self._secs['code'] == key][0]
            except IndexError:
                raise ValueError(f'{key} not exists in our database, is it valid?')
        else:
            raise TypeError('Invalid argument type: {}'.format(type(key)))

    def reset(self):
        self._secs = np.array([])

    async def load(self):
        db = cache.get_db(RedisDB.SECURITY)
        secs = await db.lrange('securities', 0, -1, encoding='utf-8')
        if len(secs) != 0:
            self._secs = np.array([tuple(x.split(',')) for x in secs], dtype=self.dtypes)
        else:
            secs = await FetchSecurityList().execute()
            self._secs = np.array([tuple(x) for x in secs], dtype=self.dtypes)
            if len(secs) == 0:
                raise ValueError("Failed to load security list")

    @lru_cache
    def choose(self, block: str) -> list:
        """
        根据指定的类型（板块）来选择证券列表
        Args:
            block:

        Returns:

        """
        pass
