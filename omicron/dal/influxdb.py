from typing import List

import cfg4py
import pandas as pd
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import WriteApi, WriteOptions, WriteType

from omicron.core.types import Frame


class PerssidentInfluxDb(object):
    @property
    def url(self) -> str:
        return self._url

    @property
    def token(self) -> str:
        return self._token

    @property
    def org(self) -> str:
        return self._org

    @property
    def client(self) -> InfluxDBClient:
        return InfluxDBClient(self.url, self.token, self.org)

    @property
    def write_api(self) -> WriteApi:
        return self.client.write_api(
            write_options=WriteOptions(
                batch_size=500,
                flush_interval=10_000,
                jitter_interval=2_000,
                retry_interval=5_000,
                max_retries=5,
                max_retry_delay=30_000,
                exponential_base=2,
                write_type=WriteType.synchronous,
            )
        )

    async def close(self):
        self.client.close()

    async def write(
        self,
        bucket: str,
        sequence: pd.DataFrame,
        data_frame_measurement_name: str = None,
        data_frame_tag_columns: List[str] = None,
    ):
        if not bucket or sequence.empty:
            return

        self.write_api.write(
            bucket,
            self.org,
            record=sequence,
            data_frame_measurement_name=data_frame_measurement_name,
            data_frame_tag_columns=data_frame_tag_columns,
        )

    async def init(self):
        cfg = cfg4py.get_instance()
        self._org = cfg.influxdb.org
        self._token = cfg.influxdb.token
        self._url = cfg.influxdb.url

    async def get_limit_in_date_range(
        self, bucket: str, code: str, begin: Frame, end: Frame
    ) -> pd.DataFrame:
        params = {
            "bucket": bucket,
            "begin": begin,
            "end": end,
            "code": code,
        }
        query = """
        from(bucket: bucket)
            |> range(start: -200d)
            |> filter(fn: (r) => r["_measurement"] == "stock")
            |> filter(fn: (r) => r["_field"] == "high_limit" or r["_field"] == "low_limit" or r["_field"] = "close")
            |> filter(fn: (r) => r["code"] == code)
            |> filter(fn: (r) => r["frame"] >= begin or r["frame"] <= end)
            |> filter(fn: (r) => r["frame_type"] == "6")
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            |> keep(columns: ["_time", "code", "frame", "frame_type","high_limit", "low_limit"])
        """
        data = self.client.query_api().query_data_frame(
            query, params=params, org=self.org
        )
        return data


cfg = cfg4py.get_instance()

influxdb = PerssidentInfluxDb()
