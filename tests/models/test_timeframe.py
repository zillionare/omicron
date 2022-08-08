import datetime
import logging
import unittest
from unittest import mock

import arrow
import numpy as np
from coretypes import FrameType

import omicron
from omicron.models.timeframe import TimeFrame as tf
from tests import init_test_env

logger = logging.getLogger(__name__)


class TimeFrameTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        await init_test_env()
        await omicron.init()
        return await super().asyncSetUp()

    async def asyncTearDown(self) -> None:
        await omicron.close()
        return await super().asyncTearDown()

    def test_resample_frames(self):
        trade_days = np.array(
            [
                datetime.date(2021, 8, 31),
                datetime.date(2021, 9, 1),
                datetime.date(2021, 9, 2),
                datetime.date(2021, 9, 3),
                datetime.date(2021, 9, 6),
                datetime.date(2021, 9, 7),
                datetime.date(2021, 9, 8),
                datetime.date(2021, 9, 9),
                datetime.date(2021, 9, 10),
                datetime.date(2021, 9, 13),
                datetime.date(2021, 9, 14),
                datetime.date(2021, 9, 15),
                datetime.date(2021, 9, 16),
                datetime.date(2021, 9, 17),
                datetime.date(2021, 9, 22),
                datetime.date(2021, 9, 23),
                datetime.date(2021, 9, 24),
                datetime.date(2021, 9, 27),
                datetime.date(2021, 9, 28),
                datetime.date(2021, 9, 29),
                datetime.date(2021, 9, 30),
                datetime.date(2021, 10, 8),
                datetime.date(2021, 10, 11),
                datetime.date(2021, 10, 12),
                datetime.date(2021, 10, 13),
                datetime.date(2021, 10, 14),
                datetime.date(2021, 10, 15),
                datetime.date(2021, 10, 18),
                datetime.date(2021, 10, 19),
                datetime.date(2021, 10, 20),
                datetime.date(2021, 10, 21),
                datetime.date(2021, 10, 22),
                datetime.date(2021, 10, 25),
                datetime.date(2021, 10, 26),
                datetime.date(2021, 10, 27),
                datetime.date(2021, 10, 28),
                datetime.date(2021, 10, 29),
                datetime.date(2021, 11, 1),
                datetime.date(2021, 11, 2),
                datetime.date(2021, 11, 3),
                datetime.date(2021, 11, 4),
                datetime.date(2021, 11, 5),
                datetime.date(2021, 11, 8),
                datetime.date(2021, 11, 9),
                datetime.date(2021, 11, 10),
                datetime.date(2021, 11, 11),
                datetime.date(2021, 11, 12),
                datetime.date(2021, 11, 15),
                datetime.date(2021, 11, 16),
                datetime.date(2021, 11, 17),
                datetime.date(2021, 11, 18),
                datetime.date(2021, 11, 19),
                datetime.date(2021, 11, 22),
                datetime.date(2021, 11, 23),
                datetime.date(2021, 11, 24),
                datetime.date(2021, 11, 25),
                datetime.date(2021, 11, 26),
                datetime.date(2021, 11, 29),
                datetime.date(2021, 11, 30),
                datetime.date(2021, 12, 1),
                datetime.date(2021, 12, 2),
                datetime.date(2021, 12, 3),
                datetime.date(2021, 12, 6),
                datetime.date(2021, 12, 7),
                datetime.date(2021, 12, 8),
                datetime.date(2021, 12, 9),
                datetime.date(2021, 12, 10),
                datetime.date(2021, 12, 13),
                datetime.date(2021, 12, 14),
                datetime.date(2021, 12, 15),
            ],
            dtype=object,
        )

        # test week frames
        week_frames = tf.resample_frames(trade_days, FrameType.WEEK)
        exp = [
            tf.int2date(x)
            for x in [
                20210903,
                20210910,
                20210917,
                20210924,
                20210930,
                20211008,
                20211015,
                20211022,
                20211029,
                20211105,
                20211112,
                20211119,
                20211126,
                20211203,
                20211210,
                20211215,
            ]
        ]

        self.assertListEqual(exp, week_frames)

        # test month framesP
        month_frames = tf.resample_frames(trade_days, FrameType.MONTH)
        exp = [
            tf.int2date(x) for x in [20210831, 20210930, 20211029, 20211130, 20211215]
        ]

        print(month_frames)

        self.assertListEqual(exp, month_frames)

        # test quarter frames
        quarter_frames = tf.resample_frames(trade_days, FrameType.QUARTER)
        exp = [tf.int2date(x) for x in [20210930, 20211215]]

        print(quarter_frames)

        self.assertListEqual(exp, quarter_frames)

        # test year frames
        trade_days = np.array(
            [
                datetime.date(2021, 12, 23),
                datetime.date(2021, 12, 24),
                datetime.date(2021, 12, 27),
                datetime.date(2021, 12, 28),
                datetime.date(2021, 12, 29),
                datetime.date(2021, 12, 30),
                datetime.date(2021, 12, 31),
                datetime.date(2022, 1, 4),
                datetime.date(2022, 1, 5),
                datetime.date(2022, 1, 6),
            ],
            dtype=object,
        )
        year_frames = tf.resample_frames(trade_days, FrameType.YEAR)
        exp = [tf.int2date(x) for x in [20211231, 20220106]]

        print(year_frames)

        self.assertListEqual(exp, year_frames)

    def test_shift_min1(self):
        X = [
            ("2020-03-26 09:31", 0, "2020-03-26 09:31"),
            ("2020-03-26 09:31", 1, "2020-03-26 09:32"),
            ("2020-03-26 11:30", 0, "2020-03-26 11:30"),
            ("2020-03-26 11:30", 1, "2020-03-26 13:01"),
            ("2020-03-26 11:30", 2, "2020-03-26 13:02"),
            ("2020-03-26 15:00", 0, "2020-03-26 15:00"),
            ("2020-03-26 15:00", 1, "2020-03-27 09:31"),
            ("2020-03-26 15:00", 241, "2020-03-30 09:31"),
        ]
        for i, (start, offset, expected) in enumerate(X):
            logger.debug("testing %s", X[i])
            actual = tf.shift(arrow.get(start).naive, offset, FrameType.MIN1)
            self.assertEqual(arrow.get(expected).naive, actual)

    def test_count_frames_min1(self):
        X = [
            ("2020-03-26 09:31", 1, "2020-03-26 09:31"),
            ("2020-03-26 09:31", 2, "2020-03-26 09:32"),
            ("2020-03-26 11:30", 1, "2020-03-26 11:30"),
            ("2020-03-26 11:30", 2, "2020-03-26 13:01"),
            ("2020-03-26 11:30", 3, "2020-03-26 13:02"),
            ("2020-03-26 15:00", 1, "2020-03-26 15:00"),
            ("2020-03-26 15:00", 2, "2020-03-27 09:31"),
            ("2020-03-26 15:00", 242, "2020-03-30 09:31"),
        ]
        for i, (start, expected, end) in enumerate(X):
            logger.debug("testing %s", X[i])
            actual = tf.count_frames(
                arrow.get(start).naive, arrow.get(end).naive, FrameType.MIN1
            )
            self.assertEqual(expected, actual)

    def test_shift_min5(self):
        X = [
            ("2020-03-26 09:35", 0, "2020-03-26 09:35"),
            ("2020-03-26 09:35", 1, "2020-03-26 09:40"),
            ("2020-03-26 09:35", 2, "2020-03-26 09:45"),
            ("2020-03-26 11:30", 0, "2020-03-26 11:30"),
            ("2020-03-26 11:30", 1, "2020-03-26 13:05"),
            ("2020-03-26 11:30", 2, "2020-03-26 13:10"),
            ("2020-03-26 15:00", 0, "2020-03-26 15:00"),
            ("2020-03-26 15:00", 1, "2020-03-27 09:35"),
            ("2020-03-26 15:00", 49, "2020-03-30 09:35"),
        ]
        for i, (start, offset, expected) in enumerate(X):
            logger.debug("testing %s", X[i])
            actual = tf.shift(arrow.get(start).naive, offset, FrameType.MIN5)
            self.assertEqual(arrow.get(expected).naive, actual)

    def test_shift_min15(self):
        X = [
            ["2020-03-26 09:45", 0, "2020-03-26 09:45"],
            ["2020-03-26 09:45", 5, "2020-03-26 11:00"],
            ["2020-03-26 09:45", 8, "2020-03-26 13:15"],
            ["2020-03-27 10:45", 14, "2020-03-30 10:15"],
            ["2020-03-26 13:15", -9, "2020-03-25 15:00"],
            ["2020-03-26 13:15", -18, "2020-03-25 11:15"],
            ["2020-03-26 13:15", -34, "2020-03-24 11:15"],
        ]

        fmt = "YYYY-MM-DD HH:mm"

        for i, (start, offset, expected) in enumerate(X):
            logger.debug("testing %s", X[i])
            actual = tf.shift(arrow.get(start, fmt).naive, offset, FrameType.MIN15)
            self.assertEqual(arrow.get(expected, fmt).naive, actual)

    def test_count_frames_min15(self):
        X = [
            ["2020-03-26 09:45", "2020-03-26 10:00", 2],
            ["2020-03-26 10:00", "2020-03-27 09:45", 16],
            ["2020-03-26 10:00", "2020-03-27 13:15", 24],
        ]

        for i, (start, end, expected) in enumerate(X):
            logger.debug("testing %s", X[i])
            start = arrow.get(start)
            end = arrow.get(end)
            actual = tf.count_frames(start, end, FrameType.MIN15)
            self.assertEqual(expected, actual)

    def test_shift_frame_min30(self):
        pass

    def test_shift(self):
        mom = arrow.get("2020-1-20")

        self.assertEqual(tf.shift(mom, 1, FrameType.DAY), tf.day_shift(mom, 1))
        self.assertEqual(tf.shift(mom, 1, FrameType.WEEK), tf.week_shift(mom, 1))
        self.assertEqual(tf.shift(mom, 1, FrameType.MONTH), tf.month_shift(mom, 1))

    def test_count_frames_min30(self):
        pass

    def test_count_day_frames(self):
        """
         [20191219, 20191220, 20191223, 20191224, 20191225, 20191226,

        20200117, 20200120, 20200121, 20200122, 20200123, 20200203,
        20200204, 20200205, 20200206, 20200207, 20200210, 20200211]

         [20200429, 20200430, 20200506, 20200507, 20200508, 20200511,
        """
        X = [
            ("2019-12-21", 1, "2019-12-21"),
            ("2020-01-23", 3, "2020-02-04"),
            ("2020-02-03", 1, "2020-02-03"),
            ("2020-02-08", 1, "2020-02-08"),
            ("2020-02-08", 1, "2020-02-09"),
            ("2020-02-08", 2, "2020-02-10"),
            ("2020-05-01", 20, "2020-06-01"),
        ]

        for i, (s, expected, e) in enumerate(X):
            logger.debug("testing %s", X[i])
            # cause 130 ± 3.34 µs, 130e-6 seconds
            actual = tf.count_day_frames(
                arrow.get(s, "YYYY-MM-DD").date(), arrow.get(e, "YYYY-MM-DD").date()
            )
            self.assertEqual(expected, actual)

    def test_week_shift(self):
        X = [
            ["2020-01-25", 0, "2020-01-23"],
            ["2020-01-23", 1, "2020-02-07"],
            ["2020-01-25", 2, "2020-02-14"],
            ["2020-05-06", 0, "2020-04-30"],
            ["2020-05-09", -3, "2020-04-17"],
        ]

        for i, (x, n, expected) in enumerate(X):
            logger.debug("testing %s", X[i])
            actual = tf.week_shift(arrow.get(x).date(), n)
            self.assertEqual(actual, arrow.get(expected).date())

    def test_day_shift(self):
        X = [  # of test case
            ["2019-12-13", 0, "2019-12-13"],  # should be 2019-12-13
            ["2019-12-15", 0, "2019-12-13"],  # should be 2019-12-13
            ["2019-12-15", 1, "2019-12-16"],  # 2019-12-16
            ["2019-12-13", 1, "2019-12-16"],  # should be 2019-12-16
            ["2019-12-15", -1, "2019-12-12"],  # 2019-12-12
        ]

        for i, (start, offset, expected) in enumerate(X):
            logger.debug("testing of %s", X[i])
            actual = tf.day_shift(arrow.get(start).date(), offset)
            self.assertEqual(arrow.get(expected).date(), actual)

    def test_count_frames_week(self):
        X = [
            ["2020-01-25", 1, "2020-01-23"],
            ["2020-01-23", 2, "2020-02-07"],
            ["2020-01-25", 3, "2020-02-14"],
            ["2020-05-06", 1, "2020-04-30"],
        ]

        for i, (start, expected, end) in enumerate(X):
            logger.debug("testing %s", X[i])
            actual = tf.count_frames(
                arrow.get(start).date(), arrow.get(end).date(), FrameType.WEEK
            )
            self.assertEqual(actual, expected)

    def test_count_frames_month(self):
        X = [
            ["2015-02-25", 1, "2015-01-30"],
            ["2015-02-27", 1, "2015-02-27"],
            ["2015-03-01", 1, "2015-02-27"],
            ["2015-03-01", 2, "2015-03-31"],
            ["2015-03-01", 1, "2015-03-30"],
            ["2015-03-01", 13, "2016-02-29"],
        ]

        for i, (start, expected, end) in enumerate(X):
            logger.debug("testing %s", X[i])
            actual = tf.count_frames(
                arrow.get(start).date(), arrow.get(end).date(), FrameType.MONTH
            )
            self.assertEqual(expected, actual)

    def test_count_frames_quarter(self):
        X = [
            ["2021-12-31", 1, "2021-12-31"],
            ["2021-12-30", 2, "2021-12-31"],  # 12-30 belongs 09-30
            ["2021-09-30", 2, "2021-12-31"],
            ["2021-09-29", 3, "2021-12-31"],
            ["2021-06-30", 3, "2021-12-31"],
            ["2021-06-30", 3, "2022-01-01"],
        ]

        for i, (start, expected, end) in enumerate(X):
            logger.info("testing %s", X[i])
            actual = tf.count_frames(
                arrow.get(start).date(), arrow.get(end).date(), FrameType.QUARTER
            )
            self.assertEqual(expected, actual)

    def test_count_frames_year(self):
        X = [
            ["2021-12-31", 1, "2021-12-31"],
            ["2020-12-31", 2, "2021-12-31"],
            ["2020-12-30", 3, "2021-12-31"],
            ["2018-12-28", 4, "2021-12-31"],
        ]

        for i, (start, expected, end) in enumerate(X):
            logger.info("testing %s", X[i])
            actual = tf.count_frames(
                arrow.get(start).date(), arrow.get(end).date(), FrameType.YEAR
            )
            self.assertEqual(expected, actual)

    def test_month_shift(self):
        X = [
            ["2015-02-25", 0, "2015-01-30"],
            ["2015-02-27", 0, "2015-02-27"],
            ["2015-03-01", 0, "2015-02-27"],
            ["2015-03-01", 1, "2015-03-31"],
            ["2015-03-01", 12, "2016-02-29"],
            ["2016-03-10", -12, "2015-02-27"],
        ]

        for i, (start, n, expected) in enumerate(X):
            logger.debug("testing %s", X[i])

            actual = tf.month_shift(arrow.get(start).date(), n)
            self.assertEqual(arrow.get(expected).date(), actual)

    def test_floor(self):
        X = [
            ("2005-01-09", FrameType.DAY, "2005-01-07"),
            ("2005-01-07", FrameType.DAY, "2005-01-07"),
            ("2005-01-08 14:00", FrameType.DAY, "2005-1-7"),
            ("2005-01-07 16:00:00", FrameType.DAY, "2005-01-07"),
            ("2005-01-07 14:59:00", FrameType.DAY, "2005-01-06"),
            ("2005-1-10 15:00:00", FrameType.WEEK, "2005-1-7"),
            ("2005-1-13 15:00:00", FrameType.WEEK, "2005-1-7"),
            ("2005-1-14 15:00:00", FrameType.WEEK, "2005-1-14"),
            ("2005-2-1 15:00:00", FrameType.MONTH, "2005-1-31"),
            ("2005-2-27 15:00:00", FrameType.MONTH, "2005-1-31"),
            ("2005-2-28 15:00:00", FrameType.MONTH, "2005-2-28"),
            ("2005-3-1 15:00:00", FrameType.MONTH, "2005-2-28"),
            ("2005-1-5 09:30", FrameType.MIN1, "2005-1-4 15:00"),
            ("2005-1-5 09:31", FrameType.MIN1, "2005-1-5 09:31"),
            ("2005-1-5 09:34", FrameType.MIN5, "2005-1-4 15:00"),
            ("2005-1-5 09:36", FrameType.MIN5, "2005-1-5 09:35"),
            ("2005-1-5 09:46", FrameType.MIN15, "2005-1-5 09:45"),
            ("2005-1-5 10:01", FrameType.MIN30, "2005-1-5 10:00"),
            ("2005-1-5 10:31", FrameType.MIN60, "2005-1-5 10:30"),
            # 如果moment为非交易日，则floor到上一交易日收盘
            ("2020-11-21 09:32", FrameType.MIN1, "2020-11-20 15:00"),
            # 如果moment刚好是frame结束时间，则floor(frame) == frame
            ("2005-1-5 10:00", FrameType.MIN30, "2005-1-5 10:00"),
        ]

        for i, (moment, frame_type, expected) in enumerate(X):
            logger.debug("testing %s", X[i])

            frame = arrow.get(moment).naive
            if frame_type in tf.day_level_frames and frame.hour == 0:
                frame = frame.date()

            actual = tf.floor(frame, frame_type)
            expected = arrow.get(expected).naive
            if frame_type in tf.day_level_frames:
                expected = arrow.get(expected).date()
            else:
                expected = arrow.get(expected).naive

            self.assertEqual(expected, actual)

    def test_ceiling(self):
        X = [
            ("2005-1-7", FrameType.DAY, "2005-1-7"),
            ("2005-1-9", FrameType.DAY, "2005-1-10"),
            ("2005-1-10", FrameType.DAY, "2005-1-10"),
            ("2005-1-4", FrameType.WEEK, "2005-1-7"),
            ("2005-1-7", FrameType.WEEK, "2005-1-7"),
            ("2005-1-9", FrameType.WEEK, "2005-1-14"),
            ("2005-1-1", FrameType.MONTH, "2005-1-31"),
            ("2005-1-5 14:59:00", FrameType.MIN1, "2005-1-5 14:59"),
            ("2005-1-5 14:59:00", FrameType.MIN5, "2005-1-5 15:00"),
            ("2005-1-5 14:59:00", FrameType.MIN15, "2005-1-5 15:00"),
            ("2005-1-5 14:59:00", FrameType.MIN30, "2005-1-5 15:00"),
            ("2005-1-5 14:59:00", FrameType.MIN60, "2005-1-5 15:00"),
            ("2005-1-5 14:55:00", FrameType.MIN5, "2005-1-5 14:55:00"),
            ("2005-1-5 14:30:00", FrameType.MIN5, "2005-1-5 14:30:00"),
            ("2005-1-9 14:59:00", FrameType.MIN5, "2005-1-10 09:35:00"),
        ]

        for i in range(0, len(X)):
            logger.debug("testing %s: %s", i, X[i])

            moment, frame_type, expected = X[i]
            if frame_type in tf.day_level_frames:
                actual = tf.ceiling(arrow.get(moment).date(), frame_type)
                expected = arrow.get(expected).date()
            else:
                moment = arrow.get(moment).naive
                actual = tf.ceiling(moment, frame_type)
                expected = arrow.get(expected).naive

            self.assertEqual(expected, actual)

    def test_get_frames_by_count(self):
        days = [
            20200117,
            20200120,
            20200121,
            20200122,
            20200123,
            20200203,
            20200204,
            20200205,
            20200206,
            20200207,
            20200210,
            20200211,
        ]

        for i in range(len(days)):
            end, n = tf.int2date(days[i]), i + 1
            expected = days[:n]
            actual = tf.get_frames_by_count(end, n, FrameType.DAY)
            logger.debug(
                "get_frames_by_count(%s, %s, %s)->%s", end, n, FrameType.DAY, actual
            )
            self.assertListEqual(expected, list(actual))

        X = [
            (202002041030, 1, [202002041030]),
            (202002041030, 2, [202002041000, 202002041030]),
            (202002041030, 3, [202002031500, 202002041000, 202002041030]),
            (202002041030, 4, [202002031430, 202002031500, 202002041000, 202002041030]),
            (
                202002041030,
                5,
                [202002031400, 202002031430, 202002031500, 202002041000, 202002041030],
            ),
            (
                202002041030,
                6,
                [
                    202002031330,
                    202002031400,
                    202002031430,
                    202002031500,
                    202002041000,
                    202002041030,
                ],
            ),
            (
                202002041030,
                7,
                [
                    202002031130,
                    202002031330,
                    202002031400,
                    202002031430,
                    202002031500,
                    202002041000,
                    202002041030,
                ],
            ),
            (
                202002041030,
                8,
                [
                    202002031100,
                    202002031130,
                    202002031330,
                    202002031400,
                    202002031430,
                    202002031500,
                    202002041000,
                    202002041030,
                ],
            ),
            (
                202002041030,
                9,
                [
                    202002031030,
                    202002031100,
                    202002031130,
                    202002031330,
                    202002031400,
                    202002031430,
                    202002031500,
                    202002041000,
                    202002041030,
                ],
            ),
            (
                202002041030,
                10,
                [
                    202002031000,
                    202002031030,
                    202002031100,
                    202002031130,
                    202002031330,
                    202002031400,
                    202002031430,
                    202002031500,
                    202002041000,
                    202002041030,
                ],
            ),
            (
                202002041030,
                11,
                [
                    202001231500,
                    202002031000,
                    202002031030,
                    202002031100,
                    202002031130,
                    202002031330,
                    202002031400,
                    202002031430,
                    202002031500,
                    202002041000,
                    202002041030,
                ],
            ),
        ]
        for i, (end, n, expected) in enumerate(X):
            end = tf.int2time(end)
            actual = tf.get_frames_by_count(end, n, FrameType.MIN30)
            logger.debug(
                "get_frames_by_count(%s, %s, %s)->%s", end, n, FrameType.DAY, actual
            )
            self.assertListEqual(expected, actual)

        actual = tf.get_frames_by_count(datetime.date(2020, 2, 12), 3, FrameType.MONTH)
        self.assertListEqual([20191129, 20191231, 20200123], actual.tolist())

        actual = tf.get_frames_by_count(datetime.date(2020, 2, 12), 3, FrameType.WEEK)
        self.assertListEqual([20200117, 20200123, 20200207], actual.tolist())

    def test_get_frames(self):
        days = [
            20200117,
            20200120,
            20200121,
            20200122,
            20200123,
            20200203,
            20200204,
            20200205,
            20200206,
            20200207,
            20200210,
            20200211,
        ]

        for i in range(len(days)):
            start = tf.int2date(days[0])
            end = tf.int2date(days[i])

            actual = tf.get_frames(start, end, FrameType.DAY)
            logger.debug(
                "get_frames(%s, %s, %s)->%s", start, end, FrameType.DAY, actual
            )
            self.assertListEqual(days[0 : i + 1], list(actual))

        X = [
            (202002041030, 1, [202002041030]),
            (202002041030, 2, [202002041000, 202002041030]),
            (202002041030, 3, [202002031500, 202002041000, 202002041030]),
            (202002041030, 4, [202002031430, 202002031500, 202002041000, 202002041030]),
            (
                202002041030,
                5,
                [202002031400, 202002031430, 202002031500, 202002041000, 202002041030],
            ),
            (
                202002041030,
                6,
                [
                    202002031330,
                    202002031400,
                    202002031430,
                    202002031500,
                    202002041000,
                    202002041030,
                ],
            ),
            (
                202002041030,
                7,
                [
                    202002031130,
                    202002031330,
                    202002031400,
                    202002031430,
                    202002031500,
                    202002041000,
                    202002041030,
                ],
            ),
            (
                202002041030,
                8,
                [
                    202002031100,
                    202002031130,
                    202002031330,
                    202002031400,
                    202002031430,
                    202002031500,
                    202002041000,
                    202002041030,
                ],
            ),
            (
                202002041030,
                9,
                [
                    202002031030,
                    202002031100,
                    202002031130,
                    202002031330,
                    202002031400,
                    202002031430,
                    202002031500,
                    202002041000,
                    202002041030,
                ],
            ),
            (
                202002041030,
                10,
                [
                    202002031000,
                    202002031030,
                    202002031100,
                    202002031130,
                    202002031330,
                    202002031400,
                    202002031430,
                    202002031500,
                    202002041000,
                    202002041030,
                ],
            ),
            (
                202002041030,
                11,
                [
                    202001231500,
                    202002031000,
                    202002031030,
                    202002031100,
                    202002031130,
                    202002031330,
                    202002031400,
                    202002031430,
                    202002031500,
                    202002041000,
                    202002041030,
                ],
            ),
        ]

        for i, (end, n, expected) in enumerate(X):
            start = tf.int2time(expected[0])
            end = tf.int2time(end)
            actual = tf.get_frames(start, end, FrameType.MIN30)
            logger.debug(
                "get_frames(%s, %s, %s)->%s", start, end, FrameType.MIN30, actual
            )
            self.assertListEqual(expected, actual)

        moments = [
            "2020-1-1",
            "2019-12-31",
            "2020-1-1 10:35",
            "2019-12-31 10:35",
            datetime.date(2019, 12, 31),
            arrow.get("2019-12-31 10:35").naive,
            arrow.get("2019-12-31 10:35").naive,
        ]

        for moment in moments:
            actual = tf.first_min_frame(moment, FrameType.MIN5)
            self.assertEqual(arrow.get("2019-12-31 09:35").naive, actual)

        moment = arrow.get("2019-12-31").date()

        expected = [
            arrow.get("2019-12-31 09:31").naive,
            arrow.get("2019-12-31 09:45").naive,
            arrow.get("2019-12-31 10:00").naive,
            arrow.get("2019-12-31 10:30").naive,
        ]
        for i, ft in enumerate(
            [FrameType.MIN1, FrameType.MIN15, FrameType.MIN30, FrameType.MIN60]
        ):
            actual = tf.first_min_frame(moment, ft)
            self.assertEqual(expected[i], actual)

    def test_last_min_frame(self):
        try:
            tf.last_min_frame(datetime.datetime.now(), FrameType.DAY)
            self.assertTrue(False)
        except ValueError as e:
            self.assertEqual("FrameType.DAY not supported", str(e))

        try:
            tf.last_min_frame(10, FrameType.DAY)
            self.assertTrue(False)
        except TypeError:
            self.assertTrue(True)

        actual = tf.last_min_frame(arrow.get("2020-1-24"), FrameType.MIN15)
        self.assertEqual(arrow.get("2020-1-23 15:00").naive, actual)

        actual = tf.last_min_frame("2020-1-24", FrameType.MIN15)
        self.assertEqual(arrow.get("2020-1-23 15:00").naive, actual)

    def test_frame_len(self):
        self.assertEqual(1, tf.frame_len(FrameType.MIN1))
        self.assertEqual(5, tf.frame_len(FrameType.MIN5))
        self.assertEqual(15, tf.frame_len(FrameType.MIN15))
        self.assertEqual(30, tf.frame_len(FrameType.MIN30))
        self.assertEqual(60, tf.frame_len(FrameType.MIN60))
        self.assertEqual(240, tf.frame_len(FrameType.DAY))

    def test_get_ticks(self):
        expected = [
            tf.ticks[FrameType.MIN1],
            tf.ticks[FrameType.MIN5],
            tf.ticks[FrameType.MIN15],
            tf.ticks[FrameType.MIN30],
            tf.ticks[FrameType.MIN60],
            tf.day_frames,
            tf.week_frames,
            tf.month_frames,
        ]

        for i, ft in enumerate(
            [
                FrameType.MIN1,
                FrameType.MIN5,
                FrameType.MIN15,
                FrameType.MIN30,
                FrameType.MIN60,
                FrameType.DAY,
                FrameType.WEEK,
                FrameType.MONTH,
            ]
        ):
            self.assertListEqual(list(expected[i]), list(tf.get_ticks(ft)))

    def test_replace_date(self):
        dtm = datetime.datetime(2020, 1, 1, 15, 35)
        dt = datetime.date(2021, 1, 1)
        self.assertEqual(
            datetime.datetime(2021, 1, 1, 15, 35), tf.replace_date(dtm, dt)
        )

    def test_is_closing_call_auction_time(self):
        for moment in ["2020-1-7 14:57", "2020-1-7 14:58", "2020-1-7 14:59"]:
            moment = arrow.get(moment).naive
            self.assertTrue(tf.is_closing_call_auction_time(moment))

        for moment in ["2020-1-7 14:56", "2020-1-7 15:00"]:
            moment = arrow.get(moment).naive
            self.assertEqual(False, tf.is_closing_call_auction_time(moment))

        # not in trade day
        self.assertTrue(not tf.is_closing_call_auction_time(arrow.get("2020-1-4")))

    def test_is_opening_call_auction_time(self):
        for moment in ["2020-1-7 09:16", "2020-1-7 09:25"]:
            moment = arrow.get(moment).naive
            self.assertTrue(tf.is_opening_call_auction_time(moment))

        self.assertTrue(not tf.is_opening_call_auction_time(arrow.get("2020-1-4")))

    def test_is_open_time(self):
        self.assertTrue(tf.is_open_time(arrow.get("2020-1-7 09:35").naive))

        with mock.patch("arrow.now", return_value=arrow.get("2020-1-7 09:35").naive):
            self.assertTrue(tf.is_open_time())

    def test_combine_time(self):
        moment = arrow.get("2020-1-1").date()
        expect = arrow.get("2020-1-1 14:30").naive

        self.assertEqual(expect, tf.combine_time(moment, 14, 30))

    async def test_save_calendar(self):
        day_frames = tf.day_frames[-200:].copy()

        trade_days = [tf.int2date(x) for x in day_frames]

        await tf.remove_calendar()
        await tf.save_calendar(trade_days)
        await tf.init()

        self.assertListEqual(day_frames.tolist(), tf.day_frames.tolist())

        week_frames = [
            20220422,
            20220429,
            20220506,
            20220513,
            20220520,
            20220527,
            20220602,
            20220610,
            20220617,
            20220624,
            20220701,
            20220708,
            20220715,
            20220722,
            20220729,
            20220805,
            20220812,
            20220819,
            20220826,
            20220902,
            20220909,
            20220916,
            20220923,
            20220930,
            20221014,
            20221021,
            20221028,
            20221104,
            20221111,
            20221118,
            20221125,
            20221202,
            20221209,
            20221216,
            20221223,
            20221230,
            20230106,
            20230113,
            20230120,
            20230203,
            20230209,
        ]
        self.assertListEqual(week_frames, tf.week_frames.tolist())

        month_frames = [
            20220429,
            20220531,
            20220630,
            20220729,
            20220831,
            20220930,
            20221031,
            20221130,
            20221230,
            20230131,
            20230209,
        ]
        self.assertListEqual(month_frames, tf.month_frames.tolist())

        quarter_frames = [20220630, 20220930, 20221230, 20230209]
        self.assertListEqual(quarter_frames, tf.quarter_frames.tolist())

        year_frames = [20221230, 20230209]
        self.assertListEqual(year_frames, tf.year_frames.tolist())

    def test_service_degrade(self):
        for k in ["day_frames", "week_frames", "month_frames"]:
            setattr(tf, k, [])

        tf.service_degrade()

        for k in ["day_frames", "week_frames", "month_frames"]:
            v = getattr(tf, k)
            self.assertTrue(len(v) > 0)

    def test_is_bar_closed(self):
        now = "2022-2-9 10:33:00"
        with mock.patch.object(arrow, "now", return_value=arrow.get(now)):
            actual = tf.is_bar_closed(datetime.date(2022, 2, 9), FrameType.DAY)
            self.assertFalse(actual)

            actual = tf.is_bar_closed(datetime.date(2022, 2, 9), FrameType.WEEK)
            self.assertFalse(actual)

            actual = tf.is_bar_closed(datetime.date(2022, 1, 28), FrameType.WEEK)
            self.assertTrue(actual)

            actual = tf.is_bar_closed(
                datetime.datetime(2022, 2, 9, 10, 30), FrameType.MIN5
            )
            self.assertTrue(actual)

            actual = tf.is_bar_closed(
                datetime.datetime(2022, 2, 9, 10, 31), FrameType.MIN5
            )
            self.assertFalse(actual)

            actual = tf.is_bar_closed(
                datetime.datetime(2022, 2, 9, 10, 33), FrameType.MIN5
            )
            self.assertFalse(actual)

        now = "2022-2-9 15:01:00"
        with mock.patch.object(arrow, "now", return_value=arrow.get(now)):
            actual = tf.is_bar_closed(datetime.date(2022, 2, 9), FrameType.DAY)
            self.assertTrue(actual)

    def test_get_previous_trade_day(self):
        now = arrow.get("2022-8-3 15:00:00").naive
        rc = tf.get_previous_trade_day(now)
        self.assertEqual(rc, datetime.date(2022, 8, 2))

        now = arrow.get("2022-8-1 15:00:00").naive
        rc = tf.get_previous_trade_day(now)
        self.assertEqual(rc, datetime.date(2022, 7, 29))

        now = arrow.get("2022-7-31 15:00:00").naive
        rc = tf.get_previous_trade_day(now)
        self.assertEqual(rc, datetime.date(2022, 7, 29))

    def test_frame_scope(self):
        now = arrow.get("2021-10-8 15:00:00").naive
        d0, d1 = tf.get_frame_scope(now, FrameType.WEEK)
        self.assertEqual(d0, datetime.date(2021, 10, 8))
        self.assertEqual(d1, datetime.date(2021, 10, 8))

        now = arrow.get("2021-10-9 15:00:00").naive
        d0, d1 = tf.get_frame_scope(now, FrameType.WEEK)
        self.assertEqual(d0, datetime.date(2021, 10, 8))
        self.assertEqual(d1, datetime.date(2021, 10, 8))

        now = arrow.get("2005-1-7 15:00:00").naive
        d0, d1 = tf.get_frame_scope(now, FrameType.WEEK)
        self.assertEqual(d0, datetime.date(2005, 1, 4))
        self.assertEqual(d1, datetime.date(2005, 1, 7))

        now = arrow.get("2022-2-1 15:00:00").naive
        d0, d1 = tf.get_frame_scope(now, FrameType.MONTH)
        self.assertEqual(d0, datetime.date(2022, 2, 7))
        self.assertEqual(d1, datetime.date(2022, 2, 28))

        now = arrow.get("2022-3-1 15:00:00").naive
        d0, d1 = tf.get_frame_scope(now, FrameType.MONTH)
        self.assertEqual(d0, datetime.date(2022, 3, 1))
        self.assertEqual(d1, datetime.date(2022, 3, 31))

        now = arrow.get("2005-1-31 15:00:00").naive
        d0, d1 = tf.get_frame_scope(now, FrameType.MONTH)
        self.assertEqual(d0, datetime.date(2005, 1, 4))
        self.assertEqual(d1, datetime.date(2005, 1, 31))
