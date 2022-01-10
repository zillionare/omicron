from typing import List

import cfg4py
import pandas as pd
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import (
    WriteApi,
    WriteOptions,
    WriteType,
)


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


cfg = cfg4py.get_instance()

influxdb = PerssidentInfluxDb()
