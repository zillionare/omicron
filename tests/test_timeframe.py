import logging
import unittest

import arrow

from omicron.core.timeframe import tf
from omicron.core.types import FrameType
from tests import init_test_env

cfg = init_test_env()

logger = logging.getLogger(__name__)


class TimeFrameTest(unittest.TestCase):
    def setUp(self) -> None:
        pass

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
            logger.info("testing %s", X[i])
            actual = tf.shift(arrow.get(start, tzinfo=cfg.tz), offset, FrameType.MIN1)
            self.assertEqual(arrow.get(expected, tzinfo=cfg.tz).datetime, actual)

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
            logger.info("testing %s", X[i])
            actual = tf.count_frames(
                arrow.get(start, tzinfo=cfg.tz),
                arrow.get(end, tzinfo=cfg.tz),
                FrameType.MIN1,
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
            logger.info("testing %s", X[i])
            actual = tf.shift(arrow.get(start, tzinfo=cfg.tz), offset, FrameType.MIN5)
            self.assertEqual(arrow.get(expected, tzinfo=cfg.tz), actual)

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
            logger.info("testing %s", X[i])
            actual = tf.shift(
                arrow.get(start, fmt, tzinfo=cfg.tz), offset, FrameType.MIN15
            )
            self.assertEqual(arrow.get(expected, fmt, tzinfo=cfg.tz).datetime, actual)

    def test_count_frames_min15(self):
        X = [
            ["2020-03-26 09:45", "2020-03-26 10:00", 2],
            ["2020-03-26 10:00", "2020-03-27 09:45", 16],
            ["2020-03-26 10:00", "2020-03-27 13:15", 24],
        ]

        for i, (start, end, expected) in enumerate(X):
            logger.info("testing %s", X[i])
            start = arrow.get(start)
            end = arrow.get(end)
            actual = tf.count_frames(start, end, FrameType.MIN15)
            self.assertEqual(expected, actual)

    def test_shift_frame_min30(self):
        pass

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
            logger.info("testing %s", X[i])
            # cause 130 ± 3.34 µs, 130e-6 seconds
            actual = tf.count_day_frames(
                arrow.get(s, "YYYY-MM-DD").date(), arrow.get(e, "YYYY-MM-DD").date()
            )
            self.assertEqual(expected, actual)

    def test_day_shift(self):
        X = [  # of test case
            ["2019-12-13", 0, "2019-12-13"],  # should be 2019-12-13
            ["2019-12-15", 0, "2019-12-13"],  # should be 2019-12-13
            ["2019-12-15", 1, "2019-12-16"],  # 2019-12-16
            ["2019-12-13", 1, "2019-12-16"],  # should be 2019-12-13
            ["2019-12-15", -1, "2019-12-12"],  # 2019-12-12
        ]

        for i, (start, offset, expected) in enumerate(X):
            logger.info("testing of %s", X[i])
            actual = tf.day_shift(arrow.get(start).date(), offset)
            self.assertEqual(arrow.get(expected).date(), actual)

    def test_week_shift(self):
        X = [
            ["2020-01-25", 0, "2020-01-23"],
            ["2020-01-23", 1, "2020-02-07"],
            ["2020-01-25", 2, "2020-02-14"],
            ["2020-05-06", 0, "2020-04-30"],
            ["2020-05-09", -3, "2020-04-17"],
        ]

        for i, (x, n, expected) in enumerate(X):
            logger.info("testing %s", X[i])
            actual = tf.week_shift(arrow.get(x).date(), n)
            self.assertEqual(actual, arrow.get(expected).date())

    def test_count_frames_week(self):
        X = [
            ["2020-01-25", 1, "2020-01-23"],
            ["2020-01-23", 2, "2020-02-07"],
            ["2020-01-25", 3, "2020-02-14"],
            ["2020-05-06", 1, "2020-04-30"],
        ]

        for i, (start, expected, end) in enumerate(X):
            logger.info("testing %s", X[i])
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
            logger.info("testing %s", X[i])
            actual = tf.count_frames(
                arrow.get(start).date(), arrow.get(end).date(), FrameType.MONTH
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
            logger.info("testing %s", X[i])

            actual = tf.month_shift(arrow.get(start).date(), n)
            self.assertEqual(arrow.get(expected).date(), actual)

    def test_floor(self):
        X = [
            ("2005-01-09", FrameType.DAY, "2005-01-07"),
            ("2005-01-07", FrameType.DAY, "2005-01-07"),
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
        ]

        for i, (moment, frame_type, expected) in enumerate(X):
            logger.info("testing %s", X[i])

            frame = arrow.get(moment).datetime
            if frame_type in tf.day_level_frames and frame.hour == 0:
                frame = frame.date()

            actual = tf.floor(frame, frame_type)
            expected = arrow.get(expected)
            if frame_type in tf.day_level_frames:
                expected = arrow.get(expected).date()
            else:
                expected = arrow.get(expected).datetime

            self.assertEqual(expected, actual)

    def test_ceil(self):
        X = [
            ("2005-1-4", FrameType.WEEK, "2005-1-7"),
            ("2005-1-1", FrameType.MONTH, "2005-1-31"),
            ("2005-1-5", FrameType.MIN1, "2005-1-5 15:00"),
            ("2005-1-5", FrameType.MIN5, "2005-1-5 15:00"),
            ("2005-1-5", FrameType.MIN15, "2005-1-5 15:00"),
            ("2005-1-5", FrameType.MIN30, "2005-1-5 15:00"),
            ("2005-1-5", FrameType.MIN60, "2005-1-5 15:00"),
        ]

        for i, (day, frame_type, expected) in enumerate(X):
            logger.info("testing %s", X[i])

            actual = tf.last_frame(day, frame_type)
            if frame_type in tf.day_level_frames:
                expected = arrow.get(expected).date()
            else:
                expected = arrow.get(expected, tzinfo=cfg.tz).datetime

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
            logger.info(
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
            logger.info(
                "get_frames_by_count(%s, %s, %s)->%s", end, n, FrameType.DAY, actual
            )
            self.assertListEqual(expected, actual)

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
            logger.info("get_frames(%s, %s, %s)->%s", start, end, FrameType.DAY, actual)
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
            logger.info(
                "get_frames(%s, %s, %s)->%s", start, end, FrameType.MIN30, actual
            )
            self.assertListEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
