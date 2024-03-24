import datetime
from typing import List, Tuple

import cfg4py
import clickhouse_connect
import pandas as pd
from clickhouse_connect.driver.client import Client
from coretypes import Frame, FrameType, SecurityInfoSchema, SecurityType
from pandera.typing import DataFrame


class Haystore(object):
    TBL_DAY_BARS = "bars_1d"
    TBL_MIN1_BARS = "bars_1m"
    TBL_SECURITIES = "securities"
    
    def init(self):
        cfg = cfg4py.get_instance()
        host = cfg.haystore.host
        port = cfg.haystore.port
        user = cfg.haystore.user
        password = cfg.haystore.password
        database = cfg.haystore.database
        self.client = clickhouse_connect.get_client(
            host=host, username=user, password=password, database=database, port=port
        )

    def close(self):
        """关闭clickhouse连接"""
        self.client.close()

    def save_bars(self, frame_type: FrameType, bars: pd.DataFrame):
        """保存行情数据。

        Args:
            frame_type: 行情数据的周期。只接受1分钟和日线
            bars: 行情数据，必须包括symbol, frame, OHLC, volume, money字段
        """
        assert frame_type in [FrameType.DAY, FrameType.MIN1]
        if frame_type == FrameType.DAY:
            table = Haystore.TBL_DAY_BARS
        else:
            table = Haystore.TBL_MIN1_BARS

        self.client.insert_df(table, bars)

    def get_bars(
        self, code: str, n: int, frame_type: FrameType, end: datetime.datetime
    ):
        """从clickhouse中获取持久化存储的行情数据

        Args:
            code: 股票代码，以.SZ/.SH结尾
            frame_type: 行情周期。必须为1分钟或者日线
            n: 记录数
            end: 记录截止日期

        """
        sql = "SELECT * from {table: Identifier} where frame < {frame: DateTime} and symbol = {symbol:String}"
        params = {"table": f"bars_{frame_type.value}", "frame": end, "symbol": code}
        return self.client.query_np(sql, parameters=params)

    def query_df(self, sql: str, **params) -> pd.DataFrame:
        """执行任意查询命令"""
        return self.client.query_df(sql, parameters=params)

    def update_factors(self, sec: str, factors: pd.Series):
        """更新复权因子。

        TODO:
            参考https://clickhouse.com/blog/handling-updates-and-deletes-in-clickhouse进行优化。

        Args:
            sec: 待更新复权因子的证券代码
            factors: 以日期为索引，复权因子为值的Series
        """

        for dt, factor in factors.items():
            sql = "alter table bars_day update factor = %(v1)s where symbol = %(v2)s and frame = %(v3)s"

            self.client.command(sql, {"v1": factor, "v2": sec, "v3": dt})

    def save_securities(self, securities: pd.DataFrame):
        """将每日证券列表存入haystore
        
        Args:
            securities应该包含dt, code, alias, initials, ipo, type字段
        """
        self.client.insert_df(self.TBL_SECURITIES, securities)

    def load_securities(self, dt: datetime.date, code: str|None = None)->DataFrame[SecurityInfoSchema]:
        """加载证券列表
        """
        assert dt is not None
        if code is None:
            sql = f"select * from {self.TBL_SECURITIES} where dt=%(v1)s"
            df = self.client.query_df(sql, (dt, ))
        else:
            sql = f"select * from {self.TBL_SECURITIES} where dt=%(v1)s and code=%(v2)s"
            df = self.client.query_df(sql, (dt, code))

        return df

haystore = Haystore()

__all__ = ["haystore"]
