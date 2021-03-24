#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import logging
from typing import Iterable, List, Optional, Tuple, Union

import aioredis
import cfg4py
import numpy as np
from aioredis.commands import Redis
from arrow.arrow import Arrow

from omicron.core.timeframe import tf
from omicron.core.types import Frame, FrameType, bars_dtype

logger = logging.getLogger(__file__)


class RedisCache:
    databases = ["_sys_", "_security_", "_temp_"]

    _security_: Redis
    _sys_: Redis
    _temp_: Redis

    @property
    def security(self) -> Redis:
        return self._security_

    @property
    def sys(self) -> Redis:
        return self._sys_

    @property
    def temp(self) -> Redis:
        return self._temp_

    async def sanity_check(self, db):
        pass

    async def close(self):
        for redis in [self.sys, self.security, self.temp]:
            redis.close()
            await redis.wait_closed()

    async def init(self):
        cfg = cfg4py.get_instance()
        for i, name in enumerate(self.databases):
            db = await aioredis.create_redis_pool(
                cfg.redis.dsn, encoding="utf-8", maxsize=2, db=i
            )
            await self.sanity_check(db)
            await db.set("__meta__.database", name)
            setattr(self, name, db)

    async def get_securities(self):
        return await self.security.lrange("securities", 0, -1, encoding="utf-8")

    async def get_bars_range(
        self, code: str, frame_type: FrameType
    ) -> Tuple[Optional[Frame], ...]:
        pl = self.security.pipeline()
        pl.hget(f"{code}:{frame_type.value}", "head")
        pl.hget(f"{code}:{frame_type.value}", "tail")
        head, tail = await pl.execute()
        converter = (
            tf.int2time
            if frame_type
            in [
                FrameType.MIN1,
                FrameType.MIN5,
                FrameType.MIN15,
                FrameType.MIN30,
                FrameType.MIN60,
            ]
            else tf.int2date
        )
        return converter(head) if head else None, converter(tail) if tail else None

    async def clear_bars_range(self, code: str, frame_type: FrameType):
        pl = self.security.pipeline()
        pl.delete(f"{code}:{frame_type.value}", "head")
        pl.delete(f"{code}:{frame_type.value}", "tail")
        return await pl.execute()

    async def set_bars_range(
        self, code: str, frame_type: FrameType, start: Arrow = None, end: Arrow = None
    ):
        converter = (
            tf.time2int
            if frame_type
            in [
                FrameType.MIN1,
                FrameType.MIN5,
                FrameType.MIN15,
                FrameType.MIN30,
                FrameType.MIN60,
            ]
            else tf.date2int
        )
        if start:
            await self.security.hset(
                f"{code}:{frame_type.value}", "head", converter(start)
            )
        if end:
            await self.security.hset(
                f"{code}:{frame_type.value}", "tail", converter(end)
            )

    async def save_bars(
        self, sec: str, bars: np.ndarray, frame_type: FrameType, sync_mode: int = 1
    ):
        """将行情数据存入缓存

        在redis cache中的数据以如下方式存储

        ```text
        "000001.XSHE:30m" -> {
            # frame     open    low  high  close volume      amount        factor
            "20200805": "13.82 13.85 13.62 13.76 144020313.0 1980352978.34 120.77"
            "20200806": "13.82 13.96 13.65 13.90 135251068.0 1868047342.49 120.77"
            "head": "20200805"
            "tail": "20200806"
        }
        ```

        这里的amount即对应frame的成交额；factor为复权因子

        args:
            sec: the full qualified code of a security or index
            bars: the data to save
            frame_type: use this to decide which store to use
            sync_mode: 1 for update, 2 for overwrite
        """
        if bars is None or len(bars) == 0:
            return

        head, tail = await self.get_bars_range(sec, frame_type)

        if not (head and tail) or sync_mode == 2:
            await self._save_bars(sec, bars, frame_type)
            return

        if (
            tf.shift(bars["frame"][-1], 1, frame_type) < head
            or tf.shift(bars["frame"][0], -1, frame_type) > tail
        ):
            # don't save to database, otherwise the data is not continuous
            logger.warning(
                "discrete bars found, code: %s, db(%s, %s), bars(%s,%s)",
                sec,
                head,
                tail,
                bars["frame"][0],
                bars["frame"][-1],
            )
            return

        # both head and tail exist, only save bars out of database's range
        bars_to_save = bars[(bars["frame"] < head) | (bars["frame"] > tail)]
        if len(bars_to_save) == 0:
            return

        await self._save_bars(
            sec,
            bars_to_save,
            frame_type,
            min(head, bars["frame"][0]),
            max(tail, bars["frame"][-1]),
        )

    async def _save_bars(
        self,
        code: str,
        bars: np.ndarray,
        frame_type: FrameType,
        head: Frame = None,
        tail: Frame = None,
    ):
        if frame_type not in [
            FrameType.MIN1,
            FrameType.MIN5,
            FrameType.MIN15,
            FrameType.MIN30,
            FrameType.MIN60,
        ]:
            head = tf.date2int(head or bars["frame"][0])
            tail = tf.date2int(tail or bars["frame"][-1])
            frame_convert_func = tf.date2int
        else:
            head = tf.time2int(head or bars["frame"][0])
            tail = tf.time2int(tail or bars["frame"][-1])
            frame_convert_func = tf.time2int

        pipeline = self.security.pipeline()
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
                frame
            ): f"{o:.2f} {h:.2f} {l:.2f} {c:.2f} {v} {a:.2f} {fq:.2f}"
            for frame, o, h, l, c, v, a, fq in bars
        }

        pipeline.hmset_dict(key, hmset)
        pipeline.hset(key, "head", head)
        pipeline.hset(key, "tail", tail)
        await pipeline.execute()

    async def get_bars(
        self,
        code: str,
        end: Union[datetime.date, datetime.datetime, Arrow],
        n: int,
        frame_type: FrameType,
    ) -> np.ndarray:
        if n == 0:
            return np.ndarray([], dtype=bars_dtype)

        frames = tf.get_frames_by_count(end, n, frame_type)
        tr = self.security.pipeline()
        key = f"{code}:{frame_type.value}"
        [tr.hget(key, int(frame)) for frame in frames]
        recs = await tr.execute()

        converter = (
            tf.int2time
            if frame_type
            in [
                FrameType.MIN1,
                FrameType.MIN5,
                FrameType.MIN15,
                FrameType.MIN30,
                FrameType.MIN60,
            ]
            else tf.int2date
        )
        data = np.empty(len(frames), dtype=bars_dtype)
        for i, frame in enumerate(frames):
            rec = recs[i]
            if rec is None:
                data[i] = (converter(frame), None, None, None, None, None, None, None)
            else:
                o, h, l, c, v, a, f = rec.split(" ")
                data[i] = (
                    converter(frame),
                    float(o),
                    float(h),
                    float(l),
                    float(c),
                    float(v),
                    float(a),
                    float(f),
                )

        return data

    async def get_bars_raw_data(
        self,
        code: str,
        end: Union[datetime.date, datetime.datetime, Arrow],
        n: int,
        frame_type: FrameType,
    ) -> bytes:
        """
        如果没有数据，返回空字节串''
        """
        if n == 0:
            return b""
        frames = tf.get_frames_by_count(end, n, frame_type)

        pl = self.security.pipeline()
        key = f"{code}:{frame_type.value}"
        [pl.hget(key, int(frame), encoding=None) for frame in frames]
        recs = await pl.execute()

        return b"".join(filter(None, recs))

    async def save_calendar(self, _type: str, days: Iterable[int]):
        key = f"calendar:{_type}"
        pl = self.security.pipeline()
        pl.delete(key)
        pl.rpush(key, *days)
        await pl.execute()

    async def load_calendar(self, _type):
        key = f"calendar:{_type}"
        result = await self.security.lrange(key, 0, -1)
        return [int(x) for x in result]


cache = RedisCache()
__all__ = ["cache"]
