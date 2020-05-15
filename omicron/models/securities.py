#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors:

"""
import datetime
import logging
from typing import List

import arrow
import numpy as np
from omega.remote.fetchsecuritylist import FetchSecurityList

from ..core.lang import singleton
from ..dal import cache

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
            self._secs = np.array([tuple(x.split(',')) for x in secs],
                                  dtype=self.dtypes)
            # docme: apply_along_axis doesn't work on structured array. The following
            # will cost 0.03 secs on 11370 recs
            self._secs['ipo'] = [datetime.date(*map(int, x.split('-'))) for x in
                                 self._secs['ipo']]
            self._secs['end'] = [datetime.date(*map(int, x.split('-'))) for x in
                                 self._secs['end']]
            logger.info("%s securities loaded from database", len(self._secs))
        else:
            logger.info("no securities in database, fetching from server...")
            secs = await FetchSecurityList().invoke()
            self._secs = np.array([tuple(x) for x in secs], dtype=self.dtypes)
            if len(self._secs) == 0:
                raise ValueError("Failed to load security list")

            logger.info("%s securities saved in database", len(self._secs))

    def choose(self, _types: List[str], exclude_exit=True, block: str = '') -> list:
        """
        根据指定的类型（板块）来选择证券列表
        Args:
            _types:
            block:
            exclude_exit:

        Returns:

        """
        cond = np.array([False] * len(self._secs))
        for _type in _types:
            cond |= (self._secs['type'] == _type)

        result = self._secs[cond]
        if exclude_exit:
            result = result[result['end'] > arrow.now().date()]
        return result['code']
