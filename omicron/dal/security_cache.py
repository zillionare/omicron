#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors:

"""
import datetime
import logging
from typing import Union, Iterable, Optional, Tuple

import numpy as np
from arrow import Arrow

from omicron.core.timeframe import tf
from omicron.core.types import FrameType, Frame, bars_dtypes
from omicron.dal import cache

logger = logging.getLogger(__name__)


async def get_bars_range(code: str, frame_type: FrameType) -> Tuple[Frame, Frame]:
    pl = cache.security.pipeline()
    pl.hget(f"{code}:{frame_type.value}", 'head')
    pl.hget(f"{code}:{frame_type.value}", 'tail')
    head, tail = await pl.execute()
    converter = tf.int2time if frame_type in [FrameType.MIN1, FrameType.MIN5,
                                              FrameType.MIN15,
                                              FrameType.MIN30,
                                              FrameType.MIN60] else tf.int2date
    return converter(head) if head else None, converter(tail) if tail else None


async def clear_bars_range(code: str, frame_type: FrameType):
    pl = cache.security.pipeline()
    pl.delete(f"{code}:{frame_type.value}", 'head')
    pl.delete(f"{code}:{frame_type.value}", 'tail')
    return await pl.execute()


async def set_bars_range(code: str, frame_type: FrameType, start: Arrow = None,
                         end: Arrow = None):
    converter = tf.time2int if frame_type in [FrameType.MIN1, FrameType.MIN5,
                                              FrameType.MIN15,
                                              FrameType.MIN30,
                                              FrameType.MIN60] else tf.date2int
    if start:
        await cache.security.hset(f"{code}:{frame_type.value}", 'head',
                                  converter(start))
    if end:
        await cache.security.hset(f"{code}:{frame_type.value}", 'tail', converter(end))


async def save_bars(sec: str, bars: np.ndarray, frame_type: FrameType,
                    sync_mode: int = 1):
    """
    为每条k线记录生成一个ID，将时间：id存入该sec对应的ordered set
    code:frame_type -> {
            20191204: [date, o, l, h, c, v]::json
            20191205: [date, o, l, h, c, v]::json
            head: date or datetime
            tail: date or datetime
            }
    :param sec: the full qualified code of a security or index
    :param bars: the data to save
    :param frame_type: use this to decide which store to use
    :param sync_mode: 1 for update, 2 for overwrite
    :return:
    """
    if bars is None or len(bars) == 0:
        return

    head, tail = await get_bars_range(sec, frame_type)

    if not (head and tail) or sync_mode == 2:
        await _save_bars(sec, bars, frame_type)
        return

    # convert_to_time = tf.int2time if frame_type in [FrameType.MIN1, FrameType.MIN5,
    # FrameType.MIN15,
    #                                                 FrameType.MIN30,
    #                                                 FrameType.MIN60] else tf.int2date
    # dt_head, dt_tail = convert_to_time(head), convert_to_time(tail)

    if tf.shift(bars['frame'][-1], 1, frame_type) < head or tf.shift(bars['frame'][0],
                                                                     -1,
                                                                     frame_type) > tail:
        # don't save to database, otherwise the data is not continuous
        logger.warning("discrete bars found, code: %s, db(%s, %s), bars(%s,%s)",
                       sec, head, tail, bars['frame'][0], bars['frame'][-1])
        return

    # both head and tail exist, only save bars out of database's range
    bars_to_save = bars[(bars['frame'] < head) | (bars['frame'] > tail)]
    if len(bars_to_save) == 0:
        return

    await _save_bars(sec, bars_to_save, frame_type, min(head, bars['frame'][0]),
                     max(tail, bars['frame'][-1]))


async def _save_bars(code: str, bars: np.ndarray, frame_type: FrameType,
                     head: Frame = None,
                     tail: Frame = None):
    if frame_type not in [FrameType.MIN1, FrameType.MIN5, FrameType.MIN15,
                          FrameType.MIN30, FrameType.MIN60]:
        head = tf.date2int(head or bars['frame'][0])
        tail = tf.date2int(tail or bars['frame'][-1])
        frame_convert_func = tf.date2int
    else:
        head = tf.time2int(head or bars['frame'][0])
        tail = tf.time2int(tail or bars['frame'][-1])
        frame_convert_func = tf.time2int

    pipeline = cache.security.pipeline()
    # the cache is empty or error during syncing, save all bars
    key = f"{code}:{frame_type.value}"
    # docme: it takes 0.05 secs to save 1000 bars, compares to 0.19 secs if we use
    # the comment out codes:
    # for row in bars:
    #     frame, o, h, l, c, v, a, fq = row
    #     frame = frame_convert_func(frame)
    #     value = f"{o:.2f} {h:.2f} {l:.2f} {c:.2f} {v} {a:.2f} {fq:.2f}"
    #     pipeline.hset(key, frame, value)
    hmset = {
        frame_convert_func(
                frame): f"{o:.2f} {h:.2f} {l:.2f} {c:.2f} {v} {a:.2f} {fq:.2f}" for
        frame, o, h, l, c, v, a, fq in bars
    }

    pipeline.hmset_dict(key, hmset)
    pipeline.hset(key, 'head', head)
    pipeline.hset(key, 'tail', tail)
    await pipeline.execute()


async def get_bars(code: str, end: Union[datetime.date, datetime.datetime, Arrow],
                   n: int,
                   frame_type: FrameType) -> np.ndarray:
    if n == 0: return np.ndarray([], dtype=bars_dtypes)

    frames = tf.get_frames_by_count(end, n, frame_type)
    tr = cache.security.pipeline()
    key = f"{code}:{frame_type.value}"
    [tr.hget(key, int(frame)) for frame in frames]
    recs = await tr.execute()

    converter = tf.int2time if frame_type in [FrameType.MIN1, FrameType.MIN5,
                                              FrameType.MIN15, FrameType.MIN30,
                                              FrameType.MIN60] else tf.int2date
    data = np.empty(len(frames), dtype=bars_dtypes)
    for i, frame in enumerate(frames):
        rec = recs[i]
        if rec is None:
            data[i] = (converter(frame), None, None, None, None, None, None, None)
        else:
            o, h, l, c, v, a, f = rec.split(" ")
            data[i] = (
                converter(frame), float(o), float(h), float(l), float(c), float(v),
                float(a), float(f))

    # todo: possible performance increase
    return np.array(data, dtype=bars_dtypes)


async def get_bars_raw_data(code: str, end: Union[datetime.date, datetime.datetime,
                                                  Arrow], n: int,
                            frame_type: FrameType) -> bytes:
    """
    如果没有数据，返回空字节串b''
    """
    if n == 0: return b''
    frames = tf.get_frames_by_count(end, n, frame_type)

    pl = cache.security.pipeline()
    key = f"{code}:{frame_type.value}"
    [pl.hget(key, int(frame), encoding=None) for frame in frames]
    recs = await pl.execute()

    return b''.join(filter(None, recs))


async def save_calendar(_type: str, days: Iterable[int]):
    key = f'calendar:{_type}'
    pl = cache.security.pipeline()
    pl.delete(key)
    pl.rpush(key, *days)
    await pl.execute()


async def load_calendar(_type):
    key = f'calendar:{_type}'
    await cache.security.lrange(key, 0, -1)
