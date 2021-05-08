import logging
import unittest

import arrow
import numpy as np

import omicron
from omicron import cache
from omicron.core.timeframe import tf
from omicron.core.types import FrameType, SecurityType
from omicron.models.securities import Securities
from omicron.models.security import Security
from tests import init_test_env, start_omega

logger = logging.getLogger(__name__)

cfg = init_test_env()


class SecurityTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        # check if omega is running
        self.omega = await start_omega()

        await omicron.init()

        self.securities = Securities()
        await self.securities.load()

    async def asyncTearDown(self) -> None:
        await omicron.shutdown()
        if self.omega is not None:
            self.omega.kill()

    def assert_bars_equal(self, expected, actual):
        self.assertEqual(expected[0][0], actual[0][0])
        self.assertEqual(expected[-1][0], actual[-1][0])
        np.testing.assert_array_almost_equal(expected[0][1:5], list(actual[0])[1:5], 2)
        np.testing.assert_array_almost_equal(
            expected[-1][1:5], list(actual[-1])[1:5], 2
        )
        self.assertAlmostEqual(expected[0][6] / 10000, list(actual[0])[6] / 10000, 0)
        self.assertAlmostEqual(expected[-1][6] / 10000, list(actual[-1])[6] / 10000, 0)

    def test_000_properties(self):
        sec = Security("000001.XSHE")
        for key, value in zip(
            "display_name ipo_date end_date".split(" "),
            "平安银行 1991-04-03 2200-01-01".split(" "),
        ):
            self.assertEqual(str(getattr(sec, key)), value)

        sec = Security("399001.XSHE")
        print(sec)

    def test_001_parse_security_type(self):
        codes = [
            "600001.XSHG",  # 浦发银行
            "000001.XSHG",  # 上证指数
            "880001.XSHG",  # 总市值
            "999999.XSHG",  # 上证指数
            "511010.XSHG",  # 国债ETF
            "100303.XSHG",  # 国债0303
            "110031.XSHG",  # 航信转债
            "120201.XSHG",  # 02三峡债
            "000001.XSHE",  # 平安银行
            "300001.XSHE",  # 特锐德
            "399001.XSHE",  # 深成指
            "150001.XSHE",  # 福锐进取
            "131800.XSHE",  # 深圳债券
            "200011.XSHE",  # B股
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
            SecurityType.STOCK_B,
        ]

        for i, code in enumerate(codes):
            self.assertEqual(Security.parse_security_type(code), expected[i])

    async def test_002_load_bars(self):
        sec = Security("000001.XSHE")
        start = arrow.get("2020-01-03").date()
        stop = arrow.get("2020-1-16").date()
        frame_type = FrameType.DAY

        expected = [
            [
                arrow.get("2020-01-03").date(),
                16.94,
                17.31,
                16.92,
                17.18,
                1.11619481e8,
                1914495474.63,
                118.73,
            ],
            [stop, 16.52, 16.57, 16.2, 16.33, 1.02810467e8, 1678888507.83, 118.73],
        ]

        logger.info("scenario: no cache")
        await cache.clear_bars_range(sec.code, frame_type)
        bars = await sec.load_bars(start, start, frame_type)
        self.assert_bars_equal([expected[0]], bars)

        bars = await sec.load_bars(start, stop, frame_type)
        self.assert_bars_equal(expected, bars)

        logger.info("scenario: load from cache")
        bars = await sec.load_bars(start, stop, frame_type)

        self.assert_bars_equal(expected, bars)

        logger.info("scenario: partial data fetch: head")
        await cache.set_bars_range(
            sec.code, frame_type, start=arrow.get("2020-01-07").date()
        )
        bars = await sec.load_bars(start, stop, frame_type)

        self.assert_bars_equal(expected, bars)

        logger.info("scenario: partial data fetch: tail")
        await cache.set_bars_range(
            sec.code, frame_type, end=arrow.get("2020-01-14").date()
        )
        bars = await sec.load_bars(start, stop, frame_type)

        self.assert_bars_equal(expected, bars)

        logger.info("scenario: 1min level backward")
        frame_type = FrameType.MIN1
        start = arrow.get("2020-05-06 15:00:00", tzinfo=cfg.tz).datetime
        await cache.clear_bars_range(sec.code, frame_type)
        stop = tf.shift(start, -249, frame_type)
        start, stop = stop, start
        bars = await sec.load_bars(start, stop, frame_type)
        # fmt:off
        expected = [
            [
                arrow.get('2020-04-30 14:51:00', tzinfo=cfg.tz).datetime, 13.99, 14.,
                13.98, 13.99, 281000., 3931001., 118.725646
            ],
            [
                arrow.get('2020-05-06 15:00:00', tzinfo=cfg.tz).datetime, 13.77,
                13.77, 13.77, 13.77, 1383400.0, 19049211.45000005, 118.725646
            ]
        ]
        # fmt:on
        self.assert_bars_equal(expected, bars)

        logger.info("scenario: 30 min level")
        frame_type = FrameType.MIN15
        start = arrow.get("2020-05-06 10:15:00", tzinfo=cfg.tz).datetime
        await cache.clear_bars_range(sec.code, frame_type)
        stop = arrow.get("2020-05-06 15:00:00", tzinfo=cfg.tz).datetime
        bars = await sec.load_bars(start, stop, frame_type)
        # fmt: off
        expected = [
            [
                arrow.get('2020-05-06 10:15:00', tzinfo=cfg.tz).datetime, 13.67,
                13.74, 13.66, 13.72, 8341905., 1.14258451e+08, 118.725646
            ],
            [
                arrow.get('2020-05-06 15:00:00', tzinfo=cfg.tz).datetime, 13.72,
                13.77, 13.72, 13.77, 7053085., 97026350.76999998, 118.725646
            ]
        ]
        # fmt: on
        self.assert_bars_equal(expected, bars)

    async def test_005_realtime_bars(self):
        """测试获取实时行情"""

        sec = Security("000001.XSHE")
        frame_type = FrameType.MIN15

        logger.info("scenario: get realtime bars")
        start = arrow.get("2020-05-06 10:15:00", tzinfo=cfg.tz).datetime
        stop = arrow.get("2020-05-06 10:25:00", tzinfo=cfg.tz).datetime
        await cache.clear_bars_range(sec.code, frame_type)

        bars = await sec.load_bars(start, stop, frame_type)
        self.assertEqual(start, bars[0]["frame"])
        self.assertEqual(stop, bars[-1]["frame"])

        # now we've cached bars at 2020-05-06 10:15:00
        bars = await sec.load_bars(start, stop, frame_type)
        self.assertEqual(start, bars[0]["frame"])
        self.assertEqual(stop, bars[-1]["frame"])

    async def test_003_slice(self):
        sec = Security("000001.XSHE")
        start = arrow.get("2020-01-03").date()
        stop = arrow.get("2020-01-16").date()
        await sec.load_bars(start, stop, FrameType.DAY)
        bars = sec[0:]
        expected = [
            [start, 16.94, 17.31, 16.92, 17.18, 1.11619481e8, 1914495474.63, 118.73],
            [stop, 16.52, 16.57, 16.2, 16.33, 1.02810467e8, 1678888507.83, 118.73],
        ]

        self.assert_bars_equal(expected, bars)
        expected = [
            [
                arrow.get("2020-01-03").date(),
                16.94,
                17.31,
                16.92,
                17.18,
                1.11619481e8,
                1914495474.63,
                118.73,
            ],
            [
                arrow.get("2020-01-06").date(),
                17.01,
                17.34,
                16.91,
                17.07,
                86208350.0,
                1477930193.19,
                118.73,
            ],
        ]
        self.assert_bars_equal(expected, sec[0:2])

    async def test_004_fq(self):
        """测试复权"""
        sec = Security("002320.XSHE")
        start = arrow.get("2020-05-06").date()
        stop = tf.shift(start, -249, FrameType.DAY)
        start, stop = stop, start
        # bars with no fq
        bars1 = await sec.load_bars(start, stop, FrameType.DAY, fq=False)
        bars2 = await sec.load_bars(start, stop, FrameType.DAY)

        self.assertEqual(250, len(bars1))
        expected1 = [
            [
                arrow.get("2019-04-24").date(),
                16.26,
                16.38,
                15.76,
                16.00,
                5981087.0,
                9.598480e07,
                3.846000,
            ],
            [
                arrow.get("2020-05-06").date(),
                10.94,
                11.22,
                10.90,
                11.15,
                22517883.0,
                2.488511e08,
                8.849346,
            ],
        ]
        expected2 = [
            [
                arrow.get("2019-04-24").date(),
                7.07,
                7.12,
                6.85,
                6.95,
                13762015.0,
                9.598480e07,
                3.846000,
            ],
            [
                arrow.get("2020-05-06").date(),
                10.94,
                11.22,
                10.90,
                11.15,
                22517883.0,
                2.488511e08,
                8.849346,
            ],
        ]
        self.assert_bars_equal(expected2, bars2)
        self.assert_bars_equal(expected1, bars1)

    async def test_price_change(self):
        sec = Security("000001.XSHG")
        frame_type = FrameType.DAY
        start = arrow.get("2020-07-29").date()
        end = arrow.get("2020-8-7").date()

        pc = await sec.price_change(start, end, frame_type, False)
        self.assertAlmostEqual(pc, 3354.04 / 3294.55 - 1, places=3)

    async def test_load_bars_batch(self):
        codes = ["000001.XSHE", "000001.XSHG"]
        # end = arrow.now(tz=cfg.tz).datetime
        # async for code, bars in Security.load_bars_batch(codes, end, 10,
        #                                                  FrameType.MIN30):
        #     print(bars[-2:])
        #     self.assertEqual(10, len(bars))
        #
        # codes = ['000001.XSHE', '000001.XSHG']
        end = arrow.get("2020-08-27").datetime
        async for code, bars in Security.load_bars_batch(codes, end, 5, FrameType.DAY):
            print(code, bars[-2:])
            self.assertEqual(5, len(bars))
            self.assertEqual(bars[-1]["frame"], end.date())
            if code == "000001.XSHG":
                self.assertAlmostEqual(3350.11, bars[-1]["close"], places=2)

    async def test_get_bars_with_turnover(self):
        code = "000001.XSHE"
        start = arrow.get("2020-01-03").date()
        stop = arrow.get("2020-1-16").date()
        frame_type = FrameType.DAY

        expected = [
            0.5752,
            0.4442,
            0.3755,
            0.4369,
            0.5316,
            0.3017,
            0.4494,
            0.6722,
            0.4429,
            0.5298,
        ]

        await cache.security.delete(f"{code}:{frame_type.value}")

        sec = Security(code)
        bars = await sec.load_bars(start, stop, frame_type, turnover=True)
        for i, bar in enumerate(bars):
            self.assertAlmostEqual(expected[i], bar["turnover"], places=3)

        start = arrow.get("2020-11-02 15:00:00", tzinfo=cfg.tz).datetime
        stop = arrow.get("2020-11-06 14:30:00", tzinfo=cfg.tz).datetime
        frame_type = FrameType.MIN30
        await cache.security.delete(f"{code}:{frame_type.value}")

        sec = Security(code)
        bars = await sec.load_bars(start, stop, frame_type, turnover=True)
        expected = [0.02299885, 0.02921041]
        self.assertAlmostEqual(expected[0], bars["turnover"][-2], places=3)
        self.assertAlmostEqual(expected[1], bars["turnover"][-1], places=3)


if __name__ == "__main__":
    unittest.main()
