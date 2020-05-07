#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors:

"""
import datetime
import logging
from functools import lru_cache

import arrow
import numpy as np
from ..core.lang import singleton
from ..dal import cache
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
        secs = await cache.get_securities()
        if len(secs) != 0:
            self._secs = np.array([tuple(x.split(',')) for x in secs], dtype=self.dtypes)
        else:
            secs = await FetchSecurityList().invoke()
            self._secs = np.array([tuple(x) for x in secs], dtype=self.dtypes)
            if len(secs) == 0:
                raise ValueError("Failed to load security list")

        # apply_alon_axis doesn't work on structured array. The following will cost 0.03 secs on 11370 recs
        self._secs['ipo'] = [datetime.date(*[int(y) for y in x.split('-')]) for x in self._secs['ipo']]
        self._secs['end'] = [datetime.date(*[int(y) for y in x.split('-')]) for x in self._secs['end']]


    @lru_cache
    def choose(self, _type='stock', exclude_exit=True, block: str = '') -> list:
        """
        根据指定的类型（板块）来选择证券列表
        Args:
            _type:
            block:
            exclude_exit:

        Returns:

        """
        result = self._secs[self._secs['type'] == _type]
        if exclude_exit:
            result = result[result['end'] > arrow.now().date()]
        return result
