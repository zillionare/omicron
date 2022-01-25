from copy import deepcopy
from typing import List, Union

import cfg4py
import arrow
import numpy as np
import pandas as pd
from coretypes import Frame, FrameType
from influxdb_client import InfluxDBClient, Point
from influxdb_client.domain.write_precision import WritePrecision
from influxdb_client.client.write_api import WriteApi, WriteOptions, WriteType


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
    def bucket_name(self) -> str:
        return self._bucket_name

    @property
    def client(self) -> InfluxDBClient:
        return InfluxDBClient(self.url, self.token, org=self.org)

    @property
    def write_api(self) -> WriteApi:
        return self.client.write_api(
            write_options=WriteOptions(
                batch_size=5000,
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
        sequence: np.array,
        fields: List[str] = None,
        data_frame_measurement_name: str = None,
    ):
        if not data_frame_measurement_name or not len(sequence):
            return
        fields = set(fields) - set(["code", "frame", "frame_type"])
        points = []
        for recs in sequence:
            data = {}
            item = dict(zip(sequence.dtype.names, recs))
            field_values = {}
            for field in fields:
                value = item.get(field)
                if value is not None:
                    field_values[field] = value
            data["fields"] = ",".join(
                map(
                    lambda x: "{field}={value}".format(field=x, value=field_values[x]),
                    field_values,
                )
            )
            data["frame"] = arrow.get(item["frame"]).timestamp
            data["code"] = item["code"]
            data["data_frame_measurement_name"] = data_frame_measurement_name
            line = "{data_frame_measurement_name},code={code} {fields} {frame}".format(
                **data
            )
            points.append(line)
        for i in range(0, len(points), 100000):
            self.write_api.write(
                self.bucket_name,
                self.org,
                points[i : i + 100000],
                write_precision=WritePrecision.S,
            )

    async def init(self):
        cfg = cfg4py.get_instance()
        self._org = cfg.influxdb.org
        self._token = cfg.influxdb.token
        self._url = cfg.influxdb.url
        self._bucket_name = cfg.influxdb.bucket_name

    async def get_stocks_in_date_range(
        self,
        code: Union[str, List[str]],
        fields: List[str],
        begin: Frame = None,
        end: Frame = None,
        limit: int = None,
        frame_type: FrameType = FrameType.DAY,
    ) -> pd.DataFrame:
        params = {
            "bucket": self.bucket_name,
        }
        end = (end.strftime("%Y-%m-%d %H:%M:%S"),)

        code = [code] if isinstance(code, str) else code
        fields_query = " or ".join(
            list(map(lambda x: f'''r["_field"] == "{x}"''', fields))
        )
        if code:
            codes_query = "|> filter(fn: (r) => %s)" % (
                " or ".join(list(map(lambda x: f'''r["code"] == "{x}"''', code)))
            )
        else:
            codes_query = ""
        columns = deepcopy(fields) or []
        columns.extend(["_value", "_field"])
        columns = '","'.join(columns)
        query = f"""
        from(bucket: bucket)
            |> range(start: -200d, stop: "{end}")
            |> filter(fn: (r) => r["_measurement"] == "stock_{frame_type.name.lower()}")
            |> filter(fn: (r) =>  {fields_query})
            {codes_query}
            |> keep(columns: ["{columns}"])
            {("|> limit(n: %d)" % limit) if limit else ""}
        """
        df = self.client.query_api().query_data_frame(
            query, params=params, org=self.org
        )
        df = pd.concat(df) if isinstance(df, list) else df
        if df.empty:
            df = pd.DataFrame(columns=fields)
        else:
            df = df.pivot(
                index=["code", "frame", "frame_type"],
                columns=["_field"],
                values=["_value"],
            )["_value"].reset_index()
        return df[fields]


cfg = cfg4py.get_instance()

influxdb = PerssidentInfluxDb()
