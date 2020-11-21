#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors:

"""
import datetime
import logging
import pickle
from typing import List, Union

import aiohttp
import cfg4py
import numpy as np

from omicron import get_local_fetcher
from omicron.core.types import Frame, FrameType

logger = logging.getLogger(__name__)
cfg = cfg4py.get_instance()


async def _quotes_server_get(item: str, params: dict = None):
    url = f"{cfg.omega.urls.quotes_server}/quotes/{item}"
    try:
        async with aiohttp.ClientSession() as client:
            async with client.get(url, json=params) as resp:
                if resp.status == 200:
                    content = await resp.content.read(-1)
                    return pickle.loads(content)
    except aiohttp.ClientConnectionError as e:
        logger.exception(e)
        logger.warning("failed to fetch %s with args: %s", item, params)

    return None


async def get_security_list():
    fetcher = get_local_fetcher()
    if fetcher:
        return await fetcher.get_security_list()
    else:
        return await _quotes_server_get("security_list")


async def get_bars(
    code: str,
    end: Frame,
    n_bars: int,
    frame_type: FrameType,
    include_unclosed: bool = True,
):
    fetcher = get_local_fetcher()
    if fetcher:
        return await fetcher.get_bars(code, end, n_bars, frame_type, include_unclosed)
    else:
        params = {
            "sec": code,
            "end": str(end),
            "n_bars": n_bars,
            "frame_type": frame_type.value,
            "include_unclosed": include_unclosed,
        }

        return await _quotes_server_get("bars", params)


async def get_bars_batch(
    secs: List[str],
    end: Frame,
    n_bars: int,
    frame_type: FrameType,
    include_unclosed: bool = True,
):
    fetcher = get_local_fetcher()
    if fetcher:
        return await fetcher.get_bars_batch(
            secs, end, n_bars, frame_type, include_unclosed
        )
    else:
        params = {
            "secs": secs,
            "end": str(end),
            "n_bars": n_bars,
            "frame_type": frame_type.value,
            "include_unclosed": include_unclosed,
        }

        return await _quotes_server_get("bars_batch", params)


async def get_valuation(
    sec: str, date: datetime.date, fields: Union[str, List[str]] = None, n: int = 1
) -> np.array:
    """从上游服务器获取截止`date`日的`n`条市值数据

    Args:
        sec (str): [description]
        fields (Union[str, List[str]]): [description]
        date (datetime.date): [description]
        n (int): [description]

    Returns:
        np.array: [description]
    """
    fetcher = get_local_fetcher()
    if isinstance(fields, str):
        fields = [fields]

    if isinstance(sec, str):
        sec = [sec]
    if fetcher:
        return await fetcher.get_valuation(sec, date, fields, n)
    else:
        params = {"secs": sec, "fields": fields, "date": str(date), "n": n}

        return await _quotes_server_get("valuation", params)


async def get_server_version():
    url = f"{cfg.omega.urls.quotes_server}/sys/version"
    async with aiohttp.ClientSession() as client:
        async with client.get(url) as resp:
            if resp.status == 200:
                return await resp.text()
