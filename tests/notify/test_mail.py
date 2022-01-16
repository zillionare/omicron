import os
import unittest
from email.message import EmailMessage

import cfg4py

from omicron.notify.mail import compose, mail_notify, send_mail
from tests import init_test_env, test_dir


class MailTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        await init_test_env()

        cfg4py.update_config(
            {
                "notify": {
                    "mail_from": "aaron_yang@jieyu.ai",
                    "mail_to": "code@jieyu.ai",
                    "mail_server": "smtp.ym.163.com",
                }
            }
        )

        return await super().asyncSetUp()

    async def test_send_mail(self):
        password = os.environ.get("MAIL_PASSWORD")
        receiver = "code@jieyu.ai"
        sender = "aaron_yang@jieyu.ai"
        body = "unitest for omicron/notify/mail"
        host = "smtp.ym.163.com"
        await send_mail(
            sender, receiver, password, subject="MailTest", body=body, host=host
        )

        # add attachment
        html = """
        <html>
        <header>
        </header>
        <body>
        <h2> This is an email with HTML content and attachment</h2>
        <p> this is content
        </body>
        </html>
        """

        txt = "you will not see this"

        file = os.path.join(test_dir(), "data/test.jpg")
        msg = compose(
            "mail test with attachment", plain_txt=txt, html=html, attachment=file
        )

        await send_mail(sender, receiver, password, msg, host=host)

        # 参数检查
        try:
            await send_mail(sender, receiver, password)
            self.assertTrue(False, "未进行参数检查")
        except TypeError:
            pass

        try:
            await send_mail(
                sender,
                receiver,
                password,
                subject="test",
                body=body,
                msg=EmailMessage(),
                host=host,
            )
            self.assertTrue(False, "未进行参数检查")
        except TypeError:
            pass

    async def test_mail_notify(self):
        body = """
        <html>
        <header>
        </header>
        <body>
        <h2> This is an email send by mail_notify</h2>
        <p> this is content
        </body>
        </html>
        """

        await mail_notify("test mail_notify", body=body, html=True)
