import os
import unittest

import cfg4py

from omicron.notify.dingtalk import DingTalkMessage, ding
from tests import init_test_env


class DingTalkTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        await init_test_env()

        return await super().asyncSetUp()

    async def test_send_dingtalk_msg(self):
        cfg = cfg4py.get_instance()

        rc = DingTalkMessage.text("UNIT TEST MESSAGE!")
        self.assertEqual(rc, '{"errcode":0,"errmsg":"ok"}')

    async def test_ding(self):
        await ding("hello world from unittest")
        await ding(
            {
                "title": "hello from unittest",
                "text": "# greetings from aaron!\n![](https://images.freeimages.com/images/large-previews/572/light-effect-1146280.jpg)",
            }
        )
