import os
import unittest
from email.message import EmailMessage

import cfg4py

from omicron.notify.mail import compose, mail_notify, send_mail
from tests import init_test_env, test_dir


class MailTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        await init_test_env()

        return await super().asyncSetUp()

    async def test_send_mail(self):
        cfg = cfg4py.get_instance()
        password = os.environ.get("MAIL_PASSWORD")
        receiver = cfg.notify.mail_to
        sender = cfg.notify.mail_from
        body = "unitest for omicron/notify/mail"
        host = cfg.notify.mail_server
        send_mail(sender, receiver, password, subject="MailTest", body=body, host=host)

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

        task = send_mail(sender, receiver, password, msg, host=host)

        # 参数检查
        with self.assertRaises(TypeError):
            send_mail(sender, receiver, password)
            self.assertTrue(False, "未进行参数检查")

        with self.assertRaises(TypeError):
            send_mail(
                sender,
                receiver,
                password,
                subject="test",
                body=body,
                msg=EmailMessage(),
                host=host,
            )

        await task
        print("please check if you have received 2 email message")

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

        # 参数检查
        with self.assertRaises(TypeError):
            await mail_notify("test", body=body, msg=EmailMessage())

        with self.assertRaises(TypeError):
            await mail_notify(None, None, None)

        # send plain txt
        await mail_notify("test mail_notify", body="plain text body")

    def test_compose(self):
        file = os.path.join(test_dir(), "data/test.jpg")
        msg = compose("test", plain_txt="plain text body", attachment=file)
        self.assertListEqual(["test", "1.0", "multipart/mixed"], msg.values())
