"""各类排名类因子
"""

import datetime
from typing import List, Optional, Union

import numpy as np
from numpy.lib import recfunctions as rfn

import omicron
from omicron.core.timeframe import tf
from omicron.core.types import Frame, FrameType
from omicron.dal import cache, db
from omicron.extension.numpy import dict_to_numpy_array, numpy_array_to_dict
from omicron.features.maths import rank
from omicron.models.security import Security
from omicron.models.valuation import Valuation


async def get_turnover_rank(end: datetime.date, win: int):
    """
    TODO: add in-memory cache here
    """
    return await cache.temp.hgetall(f"to:{tf.date2int(end)}:{win}")


async def cache_turnover_rank(turnover: dict, end: datetime.date, win: int):
    """[summary]

    Args:
        turnover: [description]
        end: [description]
        win: [description]
    """
    key = f"to:{tf.date2int(end)}:{win}"

    turnover = dict_to_numpy_array(turnover, [("code", "O"), ("turnover", "<f4")])
    ranked = rank(turnover, by="turnover")
    ranked = numpy_array_to_dict(ranked, "code", "rank")
    await cache.temp.hmset_dict(key, ranked)
    await cache.temp.pexpire(key, 3 * 24 * 3600 * 1000)


async def rank_turnover(code: str, win: int, end: datetime.date):
    """获取``code``表示的证券在到``end``为止的``win``个交易日里换手率的排名

    当本函数被第一次调用时，寻找缓存中是否有排序结果。如果没有，会进行一次全排序，并将结果存入
    缓存。因此第一次调用会比较花时间。

    # todo: 实现在指定范围内的排名。当前返回的系全排名
    # todo: 实现其它级别（比如分钟线）的换手率排名

    Args:
        code: [description]
        win: [description]
        end: [description]

    Returns:
        [description]
    """
    rank = await get_turnover_rank(end, win)
    if rank is None:
        turnover = Valuation.get_last_n_by_frame_with_agg(
            None, "turnover", "avg", end, win
        )
        rank = await cache_turnover_rank(turnover, end, win)

    return rank.get(code)
