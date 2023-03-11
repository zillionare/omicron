import datetime
import unittest
from unittest import mock

import arrow
import cfg4py
import pandas as pd
from coretypes import Frame, FrameType, bars_dtype, bars_dtype_with_code
from freezegun import freeze_time

import omicron
from omicron.dal.influx.influxclient import InfluxClient
from omicron.models.board import Board, BoardType
from omicron.models.stock import Stock
from tests import init_test_env, set_security_data_to_db

cfg = cfg4py.get_instance()


class HttpxRsp:
    status_code: int
    content: str


class BoardTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        await init_test_env()
        await omicron.init()

        url, token, bucket, org = (
            cfg.influxdb.url,
            cfg.influxdb.token,
            cfg.influxdb.bucket_name,
            cfg.influxdb.org,
        )
        self.client = InfluxClient(url, token, bucket, org)

        # 关于数据准备，请参阅: data/README.md
        await Stock.reset_cache()
        await self.client.delete_bucket()
        await self.client.create_bucket()
        await set_security_data_to_db(self.client)

        return await super().asyncSetUp()

    async def asyncTearDown(self) -> None:
        await omicron.close()
        return await super().asyncTearDown()

    async def test_rpc_call(self):
        Board.init("192.168.100.101")

        with mock.patch("httpx.AsyncClient.post") as f:
            rsp = HttpxRsp()
            rsp.status_code = 400
            rsp.content = b"xxxx"
            f.return_value = rsp
            rc = await Board._rpc_call("xxx", {"a": 1})
            self.assertEqual(rc, {"rc": 400})

            rsp.status_code = 200
            rsp.content = '{"b":1}'
            f.return_value = rsp
            rc = await Board._rpc_call("xxx", {"a": 1})
            self.assertEqual(rc, {"rc": 200, "data": {"b": 1}})

    @mock.patch("omicron.models.board.Board._rpc_call")
    async def test_board_list(self, _call):
        _call.return_value = {"rc": 200, "data": "xxx"}
        rc = await Board.board_list()
        self.assertEqual(rc, "xxx")

        _call.return_value = {"rc": 500, "data": "xxx"}
        rc = await Board.board_list()
        error_rc = {"status": 500, "msg": "httpx RPC call failed"}
        self.assertEqual(rc, error_rc)

    @mock.patch("omicron.models.board.Board._rpc_call")
    async def test_fuzzy_match_names(self, _call):
        _call.return_value = {"rc": 500, "data": "xxx"}
        rc = await Board.fuzzy_match_board_name("pattern", _btype=BoardType.INDUSTRY)
        error_rc = {"status": 500, "msg": "httpx RPC call failed"}
        self.assertEqual(rc, error_rc)

        _call.return_value = {"rc": 200, "data": "123"}
        rc = await Board.fuzzy_match_board_name("pattern", _btype=BoardType.INDUSTRY)
        self.assertEqual(rc, "123")

    @mock.patch("omicron.models.board.Board._rpc_call")
    async def test_board_info_by_id(self, _call):
        _call.return_value = {"rc": 500, "data": "xxx"}
        rc = await Board.board_info_by_id("11111")
        error_rc = {"status": 500, "msg": "httpx RPC call failed"}
        self.assertEqual(rc, error_rc)

        _call.return_value = {"rc": 200, "data": "123"}
        rc = await Board.board_info_by_id("300000", full_mode=1)
        self.assertEqual(rc, "123")
        rc = await Board.board_info_by_id("800000", full_mode=0)
        self.assertEqual(rc, "123")

    @mock.patch("omicron.models.board.Board._rpc_call")
    async def test_board_info_by_security(self, _call):
        _call.return_value = {"rc": 500, "data": "xxx"}
        rc = await Board.board_info_by_security("000001.XSHE")
        error_rc = {"status": 500, "msg": "httpx RPC call failed"}
        self.assertEqual(rc, error_rc)

        _call.return_value = {"rc": 200, "data": "123"}
        rc = await Board.board_info_by_security("000001.XSHE")
        self.assertEqual(rc, "123")
        rc = await Board.board_info_by_security(None)
        self.assertEqual(rc, [])

    @mock.patch("omicron.models.board.Board._rpc_call")
    async def test_board_filter_members(self, _call):
        _call.return_value = {"rc": 500, "data": "xxx"}
        rc = await Board.board_filter_members(["300000"], excluded=[])
        error_rc = {"status": 500, "msg": "httpx RPC call failed"}
        self.assertEqual(rc, error_rc)

        rc = await Board.board_filter_members([])
        self.assertEqual(rc, [])

        _call.return_value = {"rc": 200, "data": "123"}
        rc = await Board.board_filter_members(["300000"], excluded=None)
        self.assertEqual(rc, "123")
        rc = await Board.board_filter_members(["300000"], [])
        self.assertEqual(rc, "123")

    async def test_save_bars(self):
        df = pd.DataFrame(
            [
                [datetime.date(2022, 11, 28), 130, 153, 80, 103, 1000000, 2000000],
                [datetime.date(2022, 11, 29), 130, 153, 80, 103, 1000000, 2000000],
                [datetime.date(2022, 11, 30), 130, 153, 80, 103, 1000000, 2000000],
                [datetime.date(2022, 12, 1), 121, 123, 80, 103, 1000000, 2000000],
                [datetime.date(2022, 12, 2), 122, 128, 80, 103, 1000000, 2000000],
                [datetime.date(2022, 12, 5), 130, 153, 80, 103, 1000000, 2000000],
            ],
            columns=["日期", "开盘价", "最高价", "最低价", "收盘价", "成交量", "成交额"],
        )
        new_df = df.rename(
            columns={
                "日期": "frame",
                "开盘价": "open",
                "最高价": "high",
                "最低价": "low",
                "收盘价": "close",
                "成交量": "volume",
                "成交额": "amount",
            }
        )
        new_df.insert(0, "code", f"881101.THS")
        new_df["factor"] = 1
        bars = (
            new_df[
                [
                    "code",
                    "frame",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "amount",
                    "factor",
                ]
            ]
            .to_records(index=False)
            .astype(bars_dtype_with_code)
        )

        rc = await Board.save_bars(bars)
        self.assertTrue(rc)

        with freeze_time("2022-12-14 15:00:00"):
            rc = await Board.get_last_date_of_bars("881102")
            self.assertEqual(rc, datetime.date(2021, 9, 3))

        rc = await Board.get_last_date_of_bars("881101")
        self.assertEqual(rc, datetime.date(2022, 12, 5))

        dt1 = datetime.date(2022, 12, 1)
        dt2 = datetime.date(2022, 12, 31)
        rc = await Board.get_bars_in_range("881101", dt1, dt2)
        self.assertEqual(len(rc), 3)
