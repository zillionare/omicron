import datetime
import os
import random
import time
import unittest

import cfg4py
import numpy as np
import pandas as pd
from coretypes import FrameType
from unittest_parametrize import ParametrizedTestCase, parametrize

from omicron.dal.haystore import Haystore
from tests.config import get_config_dir


class HaystoreTest(ParametrizedTestCase):
    def setUp(self):
        cfg4py.init(get_config_dir())
        self.haystore = Haystore()
        cmd = "truncate database if exists tests"
        self.haystore.client.command(cmd)

        # create tables
        scripts = os.path.join(get_config_dir(), "clickhouse.txt")
        with open(scripts, "r", encoding="utf-8") as f:
            content = f.read()

            for sql in content.split("\n\n"):
                if len(sql) < 5:
                    continue
                self.haystore.client.command(sql)

    @parametrize("n", [(1,), (10,)])
    def test_performance(self, n):
        n = n * 10000
        codes = [f"{i:06d}.XSHG" for i in range(1, 8000)]
        df = pd.DataFrame(
            [],
            columns=["frame", "symbol", "open", "high", "low", "close", "volume", "money"],
        )

        end = datetime.datetime(2023, 12, 31)
        df["frame"] = [end - datetime.timedelta(minutes=i) for i in range(0, n)]
        sampled = random.sample(codes, 5000) * int(n / 5000)
        df["symbol"] = sampled
        df["open"] = np.random.random(n)
        df["close"] = np.random.random(n)
        df["low"] = np.random.random(n)
        df["high"] = np.random.random(n)
        df["volume"] = np.random.random(n) * 100_0000
        df["money"] = np.random.random(n) * 1_0000_0000
        df["factor"] = np.random.random(n) * 100

        t0 = time.time()
        self.haystore.save_bars(FrameType.DAY, df)
        t1 = time.time()
        bars = self.haystore.get_bars(sampled[-1], -1, FrameType.DAY, end)
        t2 = time.time()
        print(f"query returns {len(bars)}")
        print(f"insert {n} records cost {t1-t0:.1f} seconds, read cost {t2-t1:.1f} seconds")


    def test_save_securities(self):
        # ["dt", "symbol", "alias", "ipo", "type"]
        tm = datetime.date(2024, 3, 11)
        shares = pd.DataFrame(
            [
                (tm, "000001.SZ", "平安银行", tm, "stock"),
                (tm, "600001.SH", "浦发银行", tm, "stock"),
            ],
            columns=["dt", "symbol", "alias", "ipo", "type"],
        )
        self.haystore.save_ashare_list(shares)
