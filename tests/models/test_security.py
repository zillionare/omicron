import datetime
import os
import pickle
import unittest
from unittest import mock

import arrow
import cfg4py
import numpy as np
import pandas as pd
from coretypes import FrameType, SecurityType, bars_dtype, bars_with_limit_dtype
from numpy.testing import assert_array_equal

import omicron
from omicron import tf
from omicron.dal.influx.influxclient import InfluxClient
from omicron.extensions.np import numpy_append_fields
from omicron.models.security import Security, convert_nptime_to_datetime
from omicron.models.stock import Stock
from tests import (
    assert_bars_equal,
    bars_from_csv,
    init_test_env,
    set_security_data_to_db,
    test_dir,
)

cfg = cfg4py.get_instance()


class SecurityTest(unittest.IsolatedAsyncioTestCase):
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

        return super().setUp()

    async def asyncTearDown(self) -> None:
        await omicron.close()
        return await super().asyncTearDown()

    async def test_select_eval(self):
        dt = arrow.now().date()
        query = await Security.select(dt)
        query.types([]).exclude_st().exclude_kcb()
        results = await query.eval()
        tmp = [x[0] for x in results]
        codes = set(tmp)
        exp = {"000001.XSHE", "300001.XSHE", "600000.XSHG", "000001.XSHG"}
        self.assertSetEqual(exp, codes)

        # 允许ST
        query = await Security.select(dt)
        query.types([]).exclude_kcb()
        results = await query.eval()
        tmp = [x[0] for x in results]
        codes = set(tmp)
        exp = {
            "000001.XSHE",
            "000005.XSHE",
            "300001.XSHE",
            "600000.XSHG",
            "000007.XSHE",
            "000001.XSHG",
        }
        self.assertSetEqual(exp, codes)

        # 允许科创板
        query = await Security.select(dt)
        query.types([]).exclude_st()
        results = await query.eval()
        tmp = [x[0] for x in results]
        codes = set(tmp)
        exp = {
            "000001.XSHE",
            "300001.XSHE",
            "600000.XSHG",
            "688001.XSHG",
            "000001.XSHG",
        }
        self.assertSetEqual(exp, codes)

        # stock only
        query = await Security.select(dt)
        query.types(["stock"]).exclude_st().exclude_kcb()
        results = await query.eval()
        tmp = [x[0] for x in results]
        codes = set(tmp)
        exp = {"000001.XSHE", "300001.XSHE", "600000.XSHG"}
        self.assertSetEqual(exp, codes)

        # index only
        query = await Security.select(dt)
        query.types(["index"]).exclude_st().exclude_kcb()
        results = await query.eval()
        tmp = [x[0] for x in results]
        codes = set(tmp)
        exp = {"000001.XSHG"}
        self.assertSetEqual(exp, codes)

        # 排除创业板
        query = await Security.select(dt)
        query.types([]).exclude_cyb().exclude_st().exclude_kcb()
        results = await query.eval()
        tmp = [x[0] for x in results]
        codes = set(tmp)
        exp = {
            "000001.XSHE",
            "000001.XSHG",
            "600000.XSHG",
        }
        self.assertSetEqual(exp, codes)

    async def test_choose_cyb(self):
        dt = datetime.date(2022, 5, 20)
        query = await Security.select(dt)
        query.types([]).exclude_st().only_cyb()
        results = await query.eval()
        actual = [x[0] for x in results]
        self.assertListEqual(["300001.XSHE"], actual)

        # to check if we could omit `types` method
        query.exclude_st().only_cyb()
        results = await query.eval()
        actual = [x[0] for x in results]
        self.assertListEqual(["300001.XSHE"], actual)

    async def test_choose_kcb(self):
        dt = datetime.date(2022, 5, 20)

        query = await Security.select(dt)
        query.types([]).exclude_st().only_kcb()
        results = await query.eval()
        tmp = [x[0] for x in results]
        self.assertListEqual(["688001.XSHG"], tmp)

    async def test_query_info(self):
        dt = datetime.date(2022, 5, 20)
        rc = await Security.info("688001.XSHG", dt)
        self.assertEqual(rc["display_name"], "华兴源创")

    def test_fuzzy_match_ex(self):
        exp = set(["600000.XSHG"])
        self.assertSetEqual(exp, set(Stock.fuzzy_match_ex("600").keys()))

        exp = set(["000001.XSHE", "600000.XSHG"])
        self.assertSetEqual(exp, set(Stock.fuzzy_match_ex("P").keys()))

        exp = set(["000001.XSHE"])
        self.assertSetEqual(exp, set(Stock.fuzzy_match_ex("平").keys()))

    async def test_update_securities(self):
        stocks = [
            ("000001.XSHE", "平安银行", "PAYH", "1991-04-03", "2200-01-01", "stock"),
            ("000001.XSHG", "上证指数", "SZZS", "1991-07-15", "2200-01-01", "index"),
            ("000406.XSHE", "石油大明", "SYDM", "1996-06-28", "2006-04-20", "stock"),
            ("000005.XSHE", "ST星源", "STXY", "1990-12-10", "2200-01-01", "stock"),
            ("300001.XSHE", "特锐德", "TRD", "2009-10-30", "2200-01-01", "stock"),
            ("600000.XSHG", "浦发银行", "PFYH", "1999-11-10", "2200-01-01", "stock"),
            ("688001.XSHG", "华兴源创", "HXYC", "2019-07-22", "2200-01-01", "stock"),
            ("000007.XSHE", "*ST全新", "*STQX", "1992-04-13", "2200-01-01", "stock"),
        ]

        dt = arrow.now().naive
        await Security.update_secs_cache(dt, stocks)

        # make sure no duplicate
        await Security.update_secs_cache(dt, stocks)

        stocks = await Security.load_securities()
        self.assertEqual(len(stocks), 8)
        t1 = convert_nptime_to_datetime(stocks[0]["ipo"]).date()
        self.assertEqual(t1, datetime.date(1991, 4, 3))
