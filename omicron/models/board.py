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
    async def board_list(cls, _btype: BoardType = BoardType.CONCEPT) -> List[List]:
        """获取板块列表

        Args:
            _btype: 板块类别，可选值`BoardType.CONCEPT`和`BoardType.INDUSTRY`.

        Returns:
            板块列表。每一个子元素仍为一个列表，由板块代码(str), 板块名称(str)和成员数组成。示例：
            ```
            [
                ['881101', '种植业与林业', 24],
                ['881102', '养殖业', 27],
                ['881103', '农产品加工', 41],
                ['881104', '农业服务', 16],
            ]
            ```
        """
        rsp = await cls._rpc_call("board_list", {"board_type": _btype.value})
        if rsp["rc"] != 200:
            return {"status": 500, "msg": "httpx RPC call failed"}

        return rsp["data"]

    @classmethod
    async def fuzzy_match_board_name(
        cls, pattern: str, _btype: BoardType = BoardType.CONCEPT
    ) -> dict:
        """模糊查询板块代码的名字

        Examples:
        ```python
        await Board.fuzzy_match_board_name("汽车", BoardType.INDUSTRY)

        # returns:
        [
            '881125 汽车整车',
            '881126 汽车零部件',
            '881127 非汽车交运',
            '881128 汽车服务',
            '884107 汽车服务Ⅲ',
            '884194 汽车零部件Ⅲ'
        ]
        ```
        Args:
            pattern: 待查询模式串
            _btype: 查询类型

        Returns:
            包含以下key的dict: code(板块代码), name（板块名）, stocks(股票数)
        """
        if not pattern:
            return []

        rsp = await cls._rpc_call(
            "fuzzy_match_name", {"board_type": _btype.value, "pattern": pattern}
        )
        if rsp["rc"] != 200:
            return {"status": 500, "msg": "httpx RPC call failed"}

        return rsp["data"]

    @classmethod
    async def board_info_by_id(cls, board_id: str, full_mode: bool = False) -> dict:
        """通过板块代码查询板块信息（名字，成员数目或清单）

        Examples:
        ```python
        board_code = '881128' # 汽车服务 可自行修改
        board_info = await Board.board_info_by_id(board_code)
        print(board_info) # 字典形式

        # returns
        {'code': '881128', 'name': '汽车服务', 'stocks': 14}
        ```

        Returns:
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
    ) -> List[dict]:
        """获取股票所在板块信息：名称，代码

        Examples:
        ```python
        stock_code = '002236'  # 大华股份，股票代码不带字母后缀
        stock_in_board = await Board.board_info_by_security(stock_code, _btype=BoardType.CONCEPT)
        print(stock_in_board)

        # returns:
        [
            {'code': '301715', 'name': '证金持股', 'stocks': 208},
            {'code': '308870', 'name': '数字经济', 'stocks': 195},
            {'code': '308642', 'name': '数据中心', 'stocks': 188},
            ...,
            {'code': '300008', 'name': '新能源汽车', 'stocks': 603}
        ]
        ```

        Returns:
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
    ) -> List:
        """根据板块名筛选股票，参数为include, exclude

        Fixme:
            this function doesn't work
            Raise status 500

        Returns:
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

        Examples:
        ```python
        start = datetime.date(2022, 9, 1)  # 起始时间， 可修改
        end = datetime.date(2023, 3, 1)  # 截止时间， 可修改
        board_code = '881128' # 汽车服务， 可修改
        bars = await Board.get_bars_in_range(board_code, start, end)
        bars[-3:] # 打印后3条数据

        # prints:
        rec.array([
            ('2023-02-27T00:00:00', 1117.748, 1124.364, 1108.741, 1109.525, 1.77208600e+08, 1.13933095e+09, 1.),
            ('2023-02-28T00:00:00', 1112.246, 1119.568, 1109.827, 1113.43 , 1.32828124e+08, 6.65160380e+08, 1.),
            ('2023-03-01T00:00:00', 1122.233, 1123.493, 1116.62 , 1123.274, 7.21718910e+07, 3.71172850e+08, 1.)
           ],
          dtype=[('frame', '<M8[s]'), ('open', '<f4'), ('high', '<f4'), ('low', '<f4'), ('close', '<f4'), ('volume', '<f8'), ('amount', '<f8'), ('factor', '<f4')])
        ```
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
