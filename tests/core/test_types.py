import logging
import unittest
from unittest.mock import patch

from coretypes import FrameType

logger = logging.getLogger(__name__)


class MyTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_frame_type_convert(self):
        for frame_type in FrameType:
            self.assertEqual(
                frame_type.to_int(), FrameType.from_int(frame_type.to_int()).to_int()
            )

    def test_comparison(self):
        day = FrameType("1d")
        week = FrameType("1w")
        month = FrameType("1M")
        quarter = FrameType("1Q")
        year = FrameType("1Y")
        min_1 = FrameType("1m")
        min_5 = FrameType("5m")

        self.assertTrue(week > day)
        self.assertTrue(day < week)
        self.assertTrue(week < month)
        self.assertTrue(month < quarter)
        self.assertTrue(quarter < year)
        self.assertTrue(min_1, min_5)

        self.assertTrue(day >= day)
        self.assertTrue(day <= day)
