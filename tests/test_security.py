import logging
import unittest

import arrow
import cfg4py
import numpy as np
from pyemit import emit

from omicron.core.types import SecurityType, FrameType
from omicron.core.lang import async_run
from omicron.core.timeframe import tf
from omicron.dal import cache, security_cache
from omicron.dal.security_cache import construct_frame_keys
from omicron.models.securities import Securities
from omicron.models.security import Security
from tests import init_test_env

logger = logging.getLogger(__name__)
cfg4py.enable_logging()

cfg = init_test_env()


class MyTestCase(unittest.TestCase):
    @async_run
    async def setUp(self) -> None:
        await cache.init()
        await emit.start(emit.Engine.REDIS, dsn=cfg.redis.dsn,
                         exchange='zillionare-omega')
        self.securities = Securities()
        await self.securities.load()

    def assert_bars_equal(self, expected, actual):
        self.assertEqual(expected[0][0], actual[0][0])
        self.assertEqual(expected[-1][0], actual[-1][0])
        np.testing.assert_array_almost_equal(expected[0][1:5], list(actual[0])[1:5], 2)
        np.testing.assert_array_almost_equal(expected[-1][1:5], list(actual[-1])[1:5],
                                             2)
        self.assertAlmostEqual(expected[0][6] / 10000, list(actual[0])[6] / 10000, 0)
        self.assertAlmostEqual(expected[-1][6] / 10000, list(actual[-1])[6] / 10000, 0)

    def test_000_properties(self):
        sec = Security('000001.XSHE')
        for key, value in zip('display_name name ipo_date end_date'.split(' '),
                              '平安银行 PAYH 1991-04-03 2200-01-01'.split(' ')):
            self.assertEqual(str(getattr(sec, key)), value)

        sec = Security('399001.XSHE')
        print(sec)

    def test_001_parse_security_type(self):
        codes = [
            '600001.XSHG',  # 浦发银行
            '000001.XSHG',  # 上证指数
            '880001.XSHG',  # 总市值
            '999999.XSHG',  # 上证指数
            '511010.XSHG',  # 国债ETF
            '100303.XSHG',  # 国债0303
            '110031.XSHG',  # 航信转债
            '120201.XSHG',  # 02三峡债
            '000001.XSHE',  # 平安银行
            '300001.XSHE',  # 特锐德
            '399001.XSHE',  # 深成指
            '150001.XSHE',  # 福锐进取
            '131800.XSHE',  # 深圳债券
            '200011.XSHE',  # B股
        ]

        expected = [
            SecurityType.STOCK,
            SecurityType.INDEX,
            SecurityType.INDEX,
            SecurityType.INDEX,
            SecurityType.ETF,
            SecurityType.BOND,
            SecurityType.BOND,
            SecurityType.BOND,
            SecurityType.STOCK,
            SecurityType.STOCK,
            SecurityType.INDEX,
            SecurityType.ETF,
            SecurityType.BOND,
            SecurityType.STOCK_B
        ]

        for i, code in enumerate(codes):
            self.assertEqual(Security.parse_security_type(code), expected[i])

    @async_run
    async def test_002_load_bars(self):
        sec = Security('000001.XSHE')
        start = arrow.get('2020-01-04')  # started at 2020-01-03 actually
        frame_type = FrameType.DAY

        expected = [
            [arrow.get('2020-01-03').date(), 16.94, 17.31, 16.92, 17.18, 1.11619481e8,
             1914495474.63, 118.73],
            [arrow.get('2020-01-16').date(), 16.52, 16.57, 16.2, 16.33, 1.02810467e8,
             1678888507.83, 118.73]
        ]

        logger.info("scenario: no cache")
        await security_cache.clear_bars_range(sec.code, frame_type)
        bars = await sec.load_bars(start, 10, frame_type)

        self.assert_bars_equal(expected, bars)

        logger.info("scenario: load from cache")
        bars = await sec.load_bars(start, 10, frame_type)

        self.assert_bars_equal(expected, bars)

        logger.info("scenario: partial data fetch: head")
        await security_cache.set_bars_range(sec.code, frame_type,
                                            start=arrow.get('2020-01-03').date())
        bars = await sec.load_bars(start, 10, frame_type)

        self.assert_bars_equal(expected, bars)

        logger.info("scenario: partial data fetch: tail")
        await security_cache.set_bars_range(sec.code, frame_type,
                                            end=arrow.get('2020-01-14').date())
        bars = await sec.load_bars(start, 10, frame_type)

        self.assert_bars_equal(expected, bars)

        logger.info("scenario: backward")
        expected = [
            [arrow.get('2019-12-20').date(), 16.55, 16.68, 16.44, 16.59, 6.44478380e7,
             1067869779.78, 118.73],
            [arrow.get('2020-01-03').date(), 16.94, 17.31, 16.92, 17.18, 1.11619481e8,
             1914495474.63, 118.73]
        ]
        bars = await sec.load_bars(start, -10, frame_type)
        self.assert_bars_equal(expected, bars)

        logger.info("scenario: 1min level backward")
        frame_type = FrameType.MIN1
        start = arrow.get('2020-05-06 15:00:00')
        await security_cache.clear_bars_range(sec.code, frame_type)
        bars = await sec.load_bars(start, -250, frame_type)
        expected = [
            [arrow.get('2020-04-30 14:51:00', tzinfo=cfg.tz), 13.99, 14., 13.98, 13.99,
             281000.,
             3931001., 118.725646],
            [arrow.get('2020-05-06 15:00:00', tzinfo=cfg.tz), 13.77, 13.77, 13.77,
             13.77,
             1383400.0,
             19049211.45000005, 118.725646]
        ]

        self.assert_bars_equal(expected, bars)

        logger.info("scenario: 30 min level")
        frame_type = FrameType.MIN15
        start = arrow.get('2020-05-06 15:00:00')
        await security_cache.clear_bars_range(sec.code, frame_type)
        bars = await sec.load_bars(start, -14, frame_type)
        expected = [
            [arrow.get('2020-05-06 10:15:00', tzinfo=cfg.tz), 13.67, 13.74, 13.66,
             13.72,
             8341905.,
             1.14258451e+08, 118.725646],
            [arrow.get('2020-05-06 15:00:00', tzinfo=cfg.tz), 13.72, 13.77, 13.72,
             13.77,
             7053085.,
             97026350.76999998, 118.725646]
        ]
        self.assert_bars_equal(expected, bars)

    @async_run
    async def test_003_slice(self):
        sec = Security('000001.XSHE')
        await sec.load_bars(arrow.get('2020-01-03'), 10, FrameType.DAY)
        bars = sec[0:]
        expected = [
            [arrow.get('2020-01-03').date(), 16.94, 17.31, 16.92, 17.18, 1.11619481e8,
             1914495474.63, 118.73],
            [arrow.get('2020-01-16').date(), 16.52, 16.57, 16.2, 16.33, 1.02810467e8,
             1678888507.83, 118.73]
        ]

        self.assert_bars_equal(expected, bars)
        expected = [
            [arrow.get('2020-01-03').date(), 16.94, 17.31, 16.92, 17.18, 1.11619481e8,
             1914495474.63, 118.73],
            [arrow.get('2020-01-06').date(), 17.01, 17.34, 16.91, 17.07, 86208350.0,
             1477930193.19, 118.73]
        ]
        self.assert_bars_equal(expected, sec[0:2])

    @async_run
    async def test_004_fq(self):
        sec = Security('002320.XSHE')
        start = arrow.get('2020-05-06')
        # bars with no fq
        bars1 = await sec.load_bars(start, -250, FrameType.DAY, fq=False)
        bars2 = await sec.load_bars(start, -250, FrameType.DAY)
        expected1 = [
            [arrow.get("2019-04-24").date(), 16.26, 16.38, 15.76, 16.00, 5981087.0,
             9.598480e+07, 3.846000],
            [arrow.get("2020-05-06").date(), 10.94, 11.22, 10.90, 11.15, 22517883.0,
             2.488511e+08, 8.849346]
        ]
        expected2 = [
            [arrow.get('2019-04-24').date(), 7.07, 7.12, 6.85, 6.95, 13762015.0,
             9.598480e+07, 3.846000],
            [arrow.get('2020-05-06').date(), 10.94, 11.22, 10.90, 11.15, 22517883.0,
             2.488511e+08, 8.849346]
        ]
        self.assert_bars_equal(expected2, bars2)
        self.assert_bars_equal(expected1, bars1)

    def test_005_construct_frame_keys(self):
        days = [20200117, 20200120, 20200121, 20200122, 20200123, 20200203,
                20200204, 20200205, 20200206, 20200207, 20200210, 20200211]

        for i in range(len(days)):
            logger.info("testing %s, %s", days[i], i)
            end, n = tf.int2date(days[i]), i + 1
            expected = days[:n]
            actual = construct_frame_keys(end, n, FrameType.DAY)
            self.assertListEqual(expected, list(actual))

        X = [
            (202002041030, 1, [202002041030]),
            (202002041030, 2, [202002041000, 202002041030]),
            (202002041030, 3, [202002031500, 202002041000, 202002041030]),
            (202002041030, 4, [202002031430, 202002031500, 202002041000, 202002041030]),
            (202002041030, 5, [202002031400, 202002031430, 202002031500, 202002041000,
                               202002041030]),
            (202002041030, 6, [202002031330, 202002031400, 202002031430, 202002031500,
                               202002041000, 202002041030]),
            (202002041030, 7, [202002031130, 202002031330, 202002031400, 202002031430,
                               202002031500, 202002041000, 202002041030]),
            (202002041030, 8, [202002031100, 202002031130, 202002031330, 202002031400,
                               202002031430, 202002031500, 202002041000, 202002041030]),
            (202002041030, 9, [202002031030, 202002031100, 202002031130, 202002031330,
                               202002031400, 202002031430, 202002031500, 202002041000,
                               202002041030]),
            (202002041030, 10, [202002031000, 202002031030, 202002031100, 202002031130,
                                202002031330, 202002031400, 202002031430, 202002031500,
                                202002041000, 202002041030]),
            (202002041030, 11, [202001231500, 202002031000, 202002031030, 202002031100,
                                202002031130, 202002031330, 202002031400, 202002031430,
                                202002031500, 202002041000, 202002041030]),
        ]
        for i, (end, n, expected) in enumerate(X):
            logger.info("testing %s", X[i])
            end = tf.int2time(end)
            actual = construct_frame_keys(end, n, FrameType.MIN30)
            self.assertListEqual(expected, actual)

    if __name__ == '__main__':
        unittest.main()
