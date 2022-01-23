import logging
import mimetypes
import os
import time
from email.message import EmailMessage
from typing import List

import aiosmtplib
import cfg4py

logger = logging.getLogger(__name__)


async def mail_notify(
    subject: str = None,
    body: str = None,
    msg: EmailMessage = None,
    html=False,
    receivers=None,
):
    """发送邮件通知。

    发送者、接收者及邮件服务器等配置请通过cfg4py配置：

    ```
    notify:
        mail_from: aaron_yang@jieyu.ai
        mail_to
            - code@jieyu.ai
        mail_server: smtp.ym.163.com
    ```
    验证密码请通过环境变量`MAIL_PASSWORD`来配置。

    subject/body与msg必须提供其一。

    Args:
        msg (EmailMessage, optional): [description]. Defaults to None.
        subject (str, optional): [description]. Defaults to None.
        body (str, optional): [description]. Defaults to None.
        html (bool, optional): body是否按html格式处理？ Defaults to False.
        receivers (List[str], Optional): 接收者信息。如果不提供，将使用预先配置的接收者信息。
    """
    if all([msg, subject or body]):
        raise TypeError("msg参数与subject/body只能提供其中之一")
    elif all([msg is None, subject is None, body is None]):
        raise TypeError("必须提供msg参数或者subjecdt/body参数")

    if msg is None:
        if html:
            msg = compose(subject, html=body)
        else:
            msg = compose(subject, plain_txt=body)

    cfg = cfg4py.get_instance()
    if not receivers:
        receivers = cfg.notify.mail_to

    password = os.environ.get("MAIL_PASSWORD")
    await send_mail(
        cfg.notify.mail_from, receivers, password, msg, host=cfg.notify.mail_server
    )


async def send_mail(
    sender: str,
    receivers: List[str],
    password: str,
    msg: EmailMessage = None,
    host: str = None,
    port: int = 25,
    cc: List[str] = None,
    bcc: List[str] = None,
    subject: str = None,
    body: str = None,
    username: str = None,
):
    """发送邮件通知。

    如果只发送简单的文本邮件，请使用 send_mail(sender, receivers, subject=subject, plain=plain)。如果要发送较复杂的带html和附件的邮件，请先调用compose()生成一个EmailMessage,然后再调用send_mail(sender, receivers, msg)来发送邮件。

    Args:
        sender (str): [description]
        receivers (List[str]): [description]
        msg (EmailMessage, optional): [description]. Defaults to None.
        host (str, optional): [description]. Defaults to None.
        port (int, optional): [description]. Defaults to 25.
        cc (List[str], optional): [description]. Defaults to None.
        bcc (List[str], optional): [description]. Defaults to None.
        subject (str, optional): [description]. Defaults to None.
        plain (str, optional): [description]. Defaults to None.
        username (str, optional): the username used to logon to mail server. if not provided, then `sender` is used.
    """
    if all([msg is not None, subject is not None or body is not None]):
        raise TypeError("msg参数与subject/body只能提供其中之一")
    elif all([msg is None, subject is None, body is None]):
        raise TypeError("必须提供msg参数或者subjecdt/body参数")

    msg = msg or EmailMessage()

    if isinstance(receivers, str):
        receivers = [receivers]

    msg["From"] = sender
    msg["To"] = ", ".join(receivers)

    if subject:
        msg["subject"] = subject

    if body:
        msg.set_content(body)

    if cc:
        msg["Cc"] = ", ".join(cc)
    if bcc:
        msg["Bcc"] = ", ".join(bcc)

    username = username or sender

    if host is None:
        host = sender.split("@")[-1]

    t0 = time.time()
    await aiosmtplib.send(
        msg, hostname=host, port=port, username=sender, password=password
    )
    logger.info("send email in %s seconds", time.time() - t0)


def compose(
    subject: str, plain_txt: str = None, html: str = None, attachment: str = None
):
    """编写MIME邮件。

    Args:
        subject (str): [description]
        plain_txt (str): [description]
        html (str, optional): [description]. Defaults to None.
        attachment (str, optional): 附件文件名。 Defaults to None.
    """
    msg = EmailMessage()

    msg["Subject"] = subject

    if html:
        msg.preamble = plain_txt or ""
        msg.set_content(html, subtype="html")
    else:
        assert plain_txt, "Either plain_txt or html is required."
        msg.set_content(plain_txt)

    if attachment:
        ctype, encoding = mimetypes.guess_type(attachment)
        if ctype is None or encoding is not None:
            ctype = "application/octet-stream"

        maintype, subtype = ctype.split("/", 1)
        with open(attachment, "rb") as f:
            msg.add_attachment(
                f.read(), maintype=maintype, subtype=subtype, filename=attachment
            )

    return msg
