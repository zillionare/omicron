#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import logging
import re
from typing import List

import arrow
import cfg4py
import numpy as np

from omicron import cache

from ..client.quotes_fetcher import get_security_list
from ..core.lang import singleton

logger = logging.getLogger(__name__)
cfg = cfg4py.get_instance()


@singleton
class Securities(object):
    INDEX_XSHE = "399001.XSHE"
    INDEX_XSHG = "000001.XSHG"
    INDEX_CYB = "399006.XSHE"

    _secs = np.array([])
    dtypes = [
        ("code", "O"),
        ("display_name", "O"),
        ("name", "O"),
        ("ipo", "O"),
        ("end", "O"),
        ("type", "O"),
    ]

    def __str__(self):
        return f"{len(self._secs)} securities"

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, stride = key.indices(len(self._bars))
            return self._secs[start:stop]
        elif isinstance(key, int):
            return self._secs[key]
        elif isinstance(key, str):
            # assume the key is the security code
            try:
                return self._secs[self._secs["code"] == key][0]
            except IndexError:
                raise ValueError(f"{key} not exists in our database, is it valid?")
        else:
            raise TypeError("Invalid argument type: {}".format(type(key)))

    def reset(self):
        self._secs = np.array([])

    async def load(self):
        secs = await cache.get_securities()
        if len(secs) != 0:
            self._secs = np.array(
                [tuple(x.split(",")) for x in secs], dtype=self.dtypes
            )
            logger.info("%s securities loaded from cache", len(self._secs))
        else:
            logger.info("no securities in cache, fetching from server...")
            secs = await get_security_list()
            if secs is None or len(secs) == 0:
                raise ValueError("Failed to load security list")
            logger.info("%s records fetched from server.", len(secs))

            self._secs = np.array([tuple(x) for x in secs], dtype=self.dtypes)

        # docme: apply_along_axis doesn't work on structured array. The following
        # will cost 0.03 secs on 11370 recs
        if len(self._secs) == 0:
            raise ValueError("No security records")

        self._secs["ipo"] = [
            datetime.date(*map(int, x.split("-"))) for x in self._secs["ipo"]
        ]
        self._secs["end"] = [
            datetime.date(*map(int, x.split("-"))) for x in self._secs["end"]
        ]

    def choose(
        self,
        _types: List[str],
        exclude_exit=True,
        exclude_st=True,
        exclude_300=False,
        exclude_688=True,
    ) -> list:
        """
        根据指定的类型（板块）来选择证券列表
        Args:
            _types:
            exlcude:
        Returns:

        """
        cond = np.array([False] * len(self._secs))
        if not _types:
            return []

        for _type in _types:
            cond |= self._secs["type"] == _type

        result = self._secs[cond]
        if exclude_exit:
            result = result[result["end"] > arrow.now().date()]
        if exclude_300:
            result = [rec for rec in result if not rec["code"].startswith("300")]
        if exclude_688:
            result = [rec for rec in result if not rec["code"].startswith("688")]
        if exclude_st:
            result = [rec for rec in result if rec["display_name"].find("ST") == -1]
        result = np.array(result, dtype=self.dtypes)
        return result["code"].tolist()

    def choose_cyb(self):
        return [rec["code"] for rec in self._secs if rec["code"].startswith("300")]

    def fuzzy_match(self, query: str):
        query = query.upper()
        if re.match(r"\d+", query):
            return {
                sec["code"]: sec.tolist()
                for sec in self._secs
                if sec["code"].startswith(query)
            }
        elif re.match(r"[A-Z]+", query):
            return {
                sec["code"]: sec.tolist()
                for sec in self._secs
                if sec["name"].startswith(query)
            }
        else:
            return {
                sec["code"]: sec.tolist()
                for sec in self._secs
                if sec["display_name"].find(query) != -1
            }
