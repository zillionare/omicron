#!/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio
import datetime
import logging
import re
from collections import ChainMap
from types import FrameType
from typing import AsyncIterator, List

import arrow
import cfg4py
import numpy as np

from omicron import cache
from omicron.client.quotes_fetcher import get_bars_batch, get_security_list
from omicron.core.lang import singleton
from omicron.core.timeframe import tf
from omicron.core.types import Frame

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
        """选择证券标的

        本函数用于选择部分证券标的。先根据指定的类型(`stock`, `index`等）来加载证券标的，再根
        据其它参数进行排除。

        Args:
            _types : 支持的类型为`index`, `stock`, `fund`等。
            exclude_exit : 是否排除掉已退市的品种. Defaults to True.
            exclude_st : 是否排除掉作ST处理的品种. Defaults to True.
            exclude_300 : 是否排除掉创业板品种. Defaults to False.
            exclude_688 : 是否排除掉科创板品种. Defaults to True.

        Returns:
            筛选出的证券代码列表
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

    async def load_bars_batch(
        self, codes: List[str], end: Frame, n: int, frame_type: FrameType
    ) -> AsyncIterator:
        """为一批证券品种加载行情数据

        examples:
        ```
        codes = ["000001.XSHE", "000001.XSHG"]

        end = arrow.get("2020-08-27").datetime
        async for code, bars in Security.load_bars_batch(codes, end, 5, FrameType.DAY):
            print(code, bars[-2:])
            self.assertEqual(5, len(bars))
            self.assertEqual(bars[-1]["frame"], end.date())
            if code == "000001.XSHG":
                self.assertAlmostEqual(3350.11, bars[-1]["close"], places=2)
        ```

        Args:
            codes : 证券列表
            end : 结束帧
            n : 周期数
            frame_type : 帧类型

        Returns:
            [description]

        Yields:
            [description]
        """
        import warnings

        warnings.warn(
            "Security.load_bars_batch will be deprecated in version 2, use Securities.load_bars_batch instead",
            category=PendingDeprecationWarning,
        )

        assert type(end) in (datetime.date, datetime.datetime)
        closed_frame = tf.floor(end, frame_type)

        if end == closed_frame:
            start = tf.shift(closed_frame, -n + 1, frame_type)

            cached = [
                asyncio.create_task(
                    self._get_bars(code, start, closed_frame, frame_type)
                )
                for code in codes
            ]
            for fut in asyncio.as_completed(cached):
                rec = await fut
                yield rec
        else:
            start = tf.shift(closed_frame, -n + 2, frame_type)

            cached = [
                asyncio.create_task(
                    self._get_bars(code, start, closed_frame, frame_type)
                )
                for code in codes
            ]
            recs1 = await asyncio.gather(*cached)
            recs2 = await self._load_bars_batch(codes, end, 1, frame_type)

            for code, bars in recs1:
                _bars = recs2.get(code)
                if _bars is None or len(_bars) != 1:
                    logger.warning("wrong/emtpy records for %s", code)
                    continue

                yield code, np.append(bars, _bars)

    async def _load_bars_batch(
        self, codes: List[str], end: Frame, n: int, frame_type: FrameType
    ):
        batch = 1000 // n
        tasks = []
        for i in range(0, batch + 1):
            if i * batch > len(codes):
                break

            task = asyncio.create_task(
                get_bars_batch(codes[i * batch : (i + 1) * batch], end, n, frame_type)
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        return dict(ChainMap(*results))

    async def _get_bars(self, code, start, stop, frame_type):
        from omicron.models.security import Security

        sec = Security(code)
        bars = await sec.load_bars(start, stop, frame_type)
        return code, bars
