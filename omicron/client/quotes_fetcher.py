#!/usr/bin/env python
# -*- coding: utf-8 -*-

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


async def get_security_list() -> np.ndarray:
    """从Omega获取证券列表

    返回数据格式为numpy数组，列表中每一项又由证券代码、名称、拼音简写、上市日、终止日和类型组成，
    示例如下：
        ```
        [['000001.XSHE' '平安银行' 'PAYH' '1991-04-03' '2200-01-01' 'stock']
        ['000001.XSHG' '上证指数' 'SZZS' '1991-07-15' '2200-01-01' 'index']
        ['000002.XSHE' '万科A' 'WKA' '1991-01-29' '2200-01-01' 'stock']
        ['000002.XSHG' 'A股指数' 'AGZS' '1992-02-21' '2200-01-01' 'index']
        ['000003.XSHG' 'B股指数' 'BGZS' '1992-02-21' '2200-01-01' 'index']]

        ```
    Returns:
        上游服务器返回的证券列表
    """
    fetcher = get_local_fetcher()
    if fetcher:  # pragma: no cover
        return await fetcher.get_security_list()
    else:
        return await _quotes_server_get("security_list")


async def get_bars(
    code: str,
    end: Frame,
    n_bars: int,
    frame_type: FrameType,
    include_unclosed: bool = True,
) -> np.array:
    """从omega服务器获取单个证券的K线数据

    返回的数据为numpy结构化数组，示例如下：

    ```python
    array([
        (datetime.date(2021, 2, 5),
        24.6,
        25.31,
        24.27,
        24.93,
        1.01557559e+08,
        2.51780416e+09,
        120.76944)],
      dtype=[('frame', 'O'), ('open', '<f4'), ('high', '<f4'), ('low', '<f4'),
      ('close', '<f4'), ('volume', '<f8'), ('amount', '<f8'), ('factor', '<f4')])
    ```

    ```
    Args:
        code : 证券代码，如000001.XSHE
        end : 行情截止日期
        n_bars : 将获取的行情数据记录数
        frame_type : 行情数据的周期类型
        include_unclosed : 是否包含当前未结束的那个周期数据. Defaults to True.

    Returns:
        截止到end（或者前一个交易周期结束时间）、不超过n条的行情数据
    """
    fetcher = get_local_fetcher()
    if fetcher:  # pragma: no cover
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
) -> dict:
    """从Omega服务器获取一批证券的K线数据

    返回结果为一个集合，key为证券代码，value为对应的行情数据，为numpy的结构化数组格式，示例如下：
    ```json
    {
    '000001.XSHE': array([
        (datetime.date(2021, 2, 4), 24.18, 25.24, 24.04, 24.6 , 1.25524750e+08, 3.08455375e+09, 120.76944),
        (datetime.date(2021, 2, 5), 24.6 , 25.31, 24.27, 24.93, 1.01557559e+08, 2.51780416e+09, 120.76944)],
        dtype=[('frame', 'O'), ('open', '<f4'), ('high', '<f4'), ('low', '<f4'), ('close', '<f4'), ('volume', '<f8'), ('amount', '<f8'), ('factor', '<f4')]),
    '000001.XSHG': array([
        (datetime.date(2021, 2, 4), 3503.78, 3524.72, 3465.77, 3501.86, 2.98834854e+10, 4.18742553e+11, 1.),
        (datetime.date(2021, 2, 5), 3509.49, 3536.54, 3492.96, 3496.33, 2.90146174e+10, 3.97391920e+11, 1.)],
        dtype=[('frame', 'O'), ('open', '<f4'), ('high', '<f4'), ('low', '<f4'), ('close', '<f4'), ('volume', '<f8'), ('amount', '<f8'), ('factor', '<f4')])
    }
    ```
    如果end为datetime.date类型，则当include_unclosed为真时，返回截止到当前日期的数据（即使当
    前周期未结束）；否则返回前一个交易周期的数据（即使该周期已结束）。
    Args:
        secs : 证券列表
        end : 行情截止日期
        n_bars : 获取的行情数据记录数
        frame_type : 行情数据的周期类型
        include_unclosed : 是否包含当前未结束的那个周期数据。 Defaults to True.

    Returns:
        以证券代码为key，行情数据为value的集合
    """
    fetcher = get_local_fetcher()
    if fetcher:  # pragma: no cover
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
    if fetcher:  # pragma: no cover
        return await fetcher.get_valuation(sec, date, fields, n)
    else:
        params = {"secs": sec, "fields": fields, "date": str(date), "n": n}

        return await _quotes_server_get("valuation", params)


async def get_server_version() -> str:
    """获取Omega的版本号

    Returns:
        版本号，如1.0.0.a0
    """
    url = f"{cfg.omega.urls.quotes_server}/sys/version"
    async with aiohttp.ClientSession() as client:
        async with client.get(url) as resp:
            if resp.status == 200:
                return await resp.text()
