import asyncio
import datetime
import os
import time

import cfg4py
from coretypes import FrameType

from omicron.models.stock import Stock

os.environ[cfg4py.envar] = "DEV"
src_dir = os.path.dirname(__file__)
i = src_dir.index("tests")
src_dir = src_dir[:i]

config_path = os.path.join(src_dir, "omicron/config")

cfg = cfg4py.init(config_path)

# connect to production env, make sure using read-only account
org = "zillonare"
bucket = "zillonare"
token = "36DEKB4AMiiMjyT2enLKVonnmWSWNPhv7v3Dft5IQ6B2DLGGrUfmXEfLeyKIcPkHzT3N5jz1hKy4bpzOmrqGDg=="
url = "http://192.168.100.101:38086"
start = datetime.datetime(2022, 2, 15, 1, 31)
end = datetime.datetime(2022, 2, 15, 7)

cfg4py.update_config(
    {
        "influxdb": {
            "org": org,
            "bucket_name": bucket,
            "token": token,
            "url": url,
            "enable_compress": True,
        }
    }
)

print(cfg.influxdb.url)


async def main():
    t0 = time.time()
    print("running test since ", datetime.datetime.now())

    from unittest import mock

    with mock.patch(
        "omicron.models.stock.Stock._measurement_name", return_value="stock_min1"
    ):
        data = await Stock._batch_get_persisted_bars(
            [], FrameType.MIN1, begin=start, end=end
        )
        try:
            print(f"total {len(data)} bars")
            print(f'{len(data.get("000001.XSHE"))} in each')
        except Exception as e:
            print(e)
        print("query cost", round(time.time() - t0, 1), "seconds")


# asyncio.run(main())

"""
----------- test results:
53 seconds, when query one day's 1m bars (960k return records in total, 176MB transfer bytes), and influxdb contains more than 2GB records (2009 ~ 2022)
"""
