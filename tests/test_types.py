import logging
import unittest
from unittest.mock import patch

from omicron.core.types import FrameType

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
