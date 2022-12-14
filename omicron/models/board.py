import datetime
import json
import logging
from enum import Enum
from typing import List, Optional

import cfg4py
import httpx
from coretypes import (
    BarsArray,
    Frame,
    FrameType,
    bars_cols,
    bars_dtype,
    bars_dtype_with_code,
)

from omicron import tf
from omicron.dal.influx.flux import Flux
from omicron.dal.influx.serialize import DataframeDeserializer
from omicron.models import get_influx_client

logger = logging.getLogger(__name__)
cfg = cfg4py.get_instance()


class BoardType(Enum):
    INDUSTRY = "industry"
    CONCEPT = "concept"


class Board:
    server_ip: str
    server_port: int
    measurement = "board_bars_1d"

    @classmethod
    def init(cls, ip: str, port: int = 3180):
        cls.server_ip = ip
        cls.server_port = port

    @classmethod
    async def _rpc_call(cls, url: str, param: str):
        _url = f"http://{cls.server_ip}:{cls.server_port}/api/board/{url}"

        async with httpx.AsyncClient() as client:
            r = await client.post(_url, json=param, timeout=10)
            if r.status_code != 200:
                logger.error(
                    f"failed to post RPC call, {_url}: {param}, response: {r.content.decode()}"
                )
                return {"rc": r.status_code}

            rsp = json.loads(r.content)
            return {"rc": 200, "data": rsp}

    @classmethod
    async def board_list(cls, _btype: BoardType = BoardType.CONCEPT):
        rsp = await cls._rpc_call("board_list", {"board_type": _btype.value})
        if rsp["rc"] != 200:
            return {"status": 500, "msg": "httpx RPC call failed"}

        return rsp["data"]

    @classmethod
    async def fuzzy_match_board_name(
        cls, pattern: str, _btype: BoardType = BoardType.CONCEPT
    ):
        # 模糊查询板块代码的名字
        if not pattern:
            return []

        rsp = await cls._rpc_call(
            "fuzzy_match_name", {"board_type": _btype.value, "pattern": pattern}
        )
        if rsp["rc"] != 200:
            return {"status": 500, "msg": "httpx RPC call failed"}

        return rsp["data"]

    @classmethod
    async def board_info_by_id(cls, board_id: str, full_mode: bool = False):
        """通过板块代码查询板块信息（名字，成员数目或清单）

        返回值：
            {'code': '301505', 'name': '医疗器械概念', 'stocks': 242}
            or
            {'code': '301505', 'name': '医疗器械概念', 'stocks': [['300916', '朗特智能'], ['300760', '迈瑞医疗']]}
        """

        if not board_id:
            return {}
        if board_id[0] == "3":
            _btype = BoardType.CONCEPT
        else:
            _btype = BoardType.INDUSTRY

        _mode = 0
        if full_mode:  # 转换bool类型
            _mode = 1

        rsp = await cls._rpc_call(
            "info",
            {"board_type": _btype.value, "board_id": board_id, "fullmode": _mode},
        )
        if rsp["rc"] != 200:
            return {"status": 500, "msg": "httpx RPC call failed"}

        return rsp["data"]

    @classmethod
    async def board_info_by_security(
        cls, security: str, _btype: BoardType = BoardType.CONCEPT
    ):
        """获取股票所在板块信息：名称，代码

        返回值：
            [{'code': '301505', 'name': '医疗器械概念'}]
        """

        if not security:
            return []

        rsp = await cls._rpc_call(
            "info_by_sec", {"board_type": _btype.value, "security": security}
        )
        if rsp["rc"] != 200:
            return {"status": 500, "msg": "httpx RPC call failed"}

        return rsp["data"]

    @classmethod
    async def board_filter_members(
        cls,
        included: List[str],
        excluded: List[str] = [],
        _btype: BoardType = BoardType.CONCEPT,
    ):
        """根据板块名筛选股票，参数为include, exclude

        返回值：
            [['300181', '佐力药业'], ['600056', '中国医药']]
        """
        if not included:
            return []
        if excluded is None:
            excluded = []

        rsp = await cls._rpc_call(
            "board_filter_members",
            {
                "board_type": _btype.value,
                "include_boards": included,
                "exclude_boards": excluded,
            },
        )
        if rsp["rc"] != 200:
            return {"status": 500, "msg": "httpx RPC call failed"}

        return rsp["data"]

    @classmethod
    async def new_concept_boards(cls, days: int = 10):
        raise NotImplementedError("not ready")

    @classmethod
    async def latest_concept_boards(n: int = 3):
        raise NotImplementedError("not ready")

    @classmethod
    async def new_concept_members(days: int = 10, prot: int = None):
        raise NotImplementedError("not ready")

    @classmethod
    async def board_filter(
        cls, industry=None, with_concepts: Optional[List[str]] = None, without=[]
    ):
        raise NotImplementedError("not ready")

    @classmethod
    async def save_bars(cls, bars):
        client = get_influx_client()

        logger.info(
            "persisting bars to influxdb: %s, %d secs", cls.measurement, len(bars)
        )
        await client.save(bars, cls.measurement, tag_keys=["code"], time_key="frame")
        return True

    @classmethod
    async def get_last_date_of_bars(cls, code: str):
        # 行业板块回溯1年的数据，概念板块只取当年的数据
        code = f"{code}.THS"

        client = get_influx_client()

        now = datetime.datetime.now()
        dt_end = tf.day_shift(now, 0)
        # 250 + 60: 可以得到60个MA250的点, 默认K线图120个节点
        dt_start = tf.day_shift(now, -310)

        flux = (
            Flux()
            .measurement(cls.measurement)
            .range(dt_start, dt_end)
            .bucket(client._bucket)
            .tags({"code": code})
        )

        data = await client.query(flux)
        if len(data) == 2:  # \r\n
            return dt_start
        ds = DataframeDeserializer(
            sort_values="_time", usecols=["_time"], time_col="_time", engine="c"
        )
        bars = ds(data)
        secs = bars.to_records(index=False).astype("datetime64[s]")

        _dt = secs[-1].item()
        return _dt.date()

    @classmethod
    async def get_bars_in_range(
        cls, code: str, start: Frame, end: Frame = None
    ) -> BarsArray:
        """从持久化数据库中获取介于[`start`, `end`]间的行情记录

        Args:
            code: 板块代码（概念、行业）
            start: 起始时间
            end: 结束时间，如果未指明，则取当前时间

        Returns:
            返回dtype为`coretypes.bars_dtype`的一维numpy数组。
        """
        end = end or datetime.datetime.now()
        code = f"{code}.THS"

        keep_cols = ["_time"] + list(bars_cols[1:])

        flux = (
            Flux()
            .bucket(cfg.influxdb.bucket_name)
            .range(start, end)
            .measurement(cls.measurement)
            .fields(keep_cols)
            .tags({"code": code})
        )

        serializer = DataframeDeserializer(
            encoding="utf-8",
            names=[
                "_",
                "table",
                "result",
                "frame",
                "code",
                "amount",
                "close",
                "factor",
                "high",
                "low",
                "open",
                "volume",
            ],
            engine="c",
            skiprows=0,
            header=0,
            usecols=bars_cols,
            parse_dates=["frame"],
        )

        client = get_influx_client()
        result = await client.query(flux, serializer)
        return result.to_records(index=False).astype(bars_dtype)
