import asyncio
import datetime
import os
import time

import cfg4py
from coretypes import FrameType

import omicron
from omicron.models.security import Security
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
token = "hwxHycJfp_t6bCOYe2MhEDW4QBOO4FDtgeBWnPR6bGZJGEZ_41m_OHtTJFZKyD2HsbVqkZM8rJNkMvjyoXCG6Q=="
url = "http://192.168.100.101:58086"
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
            "max_query_size": 500 * 300,
        }
    }
)

print(cfg.influxdb.url)


async def batch_get_persisted_bars_n(codes=None):
    t0 = time.time()
    print("running test since ", datetime.datetime.now())

    from unittest import mock

    # with mock.patch(
    #     "omicron.models.stock.Stock._measurement_name", return_value="stock_min1"
    # ):
    data = await Stock._batch_get_persisted_bars_n(
        FrameType.MIN1, 240, end=end, codes=codes
    )
    try:
        print(f"total {len(data)} bars")
        print(f'{len(data[data.code=="000001.XSHE"])} in each')
    except Exception as e:
        print(e)
    print("query cost", round(time.time() - t0, 1), "seconds")


async def batch_get_persisted_bars_in_range(secs):
    print("running test since ", datetime.datetime.now())

    t0 = time.time()

    # from unittest import mock

    # with mock.patch(
    #     "omicron.models.stock.Stock._measurement_name", return_value="stock_min1"
    # ):
    # data = await Stock._batch_get_persisted_bars_in_range(
    #     secs, FrameType.DAY, begin=datetime.datetime(2021, 2, 14, 7), end=end
    # )
    data = await Stock._batch_get_persisted_bars_in_range(
        secs,
        FrameType.MIN1,
        begin=datetime.datetime(2022, 7, 28, 9, 31),
        end=datetime.datetime(2022, 7, 28, 15),
    )
    try:
        print(f"total {len(data)} bars")
        # for _, group in data.groupby("code"):
        #     assert all(group.frame.values[1:] > group.frame.values[:-1])
    except Exception as e:
        print(e)
    print("query cost", round(time.time() - t0, 1), "seconds")


async def main():
    cfg4py.update_config({"redis": {"dsn": "redis://192.168.100.101:56379"}})
    await omicron.init()
    secs = await Security.select(datetime.date(2022, 7, 8)).types(["stock"]).eval()
    tasks = []
    t0 = time.time()
    # barss = await batch_get_persisted_bars_in_range(secs)

    results = await Stock._batch_get_cached_bars_n(FrameType.MIN1, 2)
    print(len(results))
    # print(len(results.get("000001.XSHE")))
    print("total secs:", round(time.time() - t0, 1))


asyncio.run(main())

"""
----------- test results:
53 seconds, when query one day's 1m bars (960k return records in total, 176MB transfer bytes), and influxdb contains more than 2GB records (2009 ~ 2022)
"""
