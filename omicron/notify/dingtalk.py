# -*- coding: utf-8 -*-
import asyncio
import base64
import hashlib
import hmac
import json
import logging
import time
import urllib.parse
from typing import Awaitable, Union

import cfg4py
import httpx
from deprecation import deprecated

from omicron.core.errors import ConfigError

logger = logging.getLogger(__name__)
cfg = cfg4py.get_instance()


class DingTalkMessage:
    """
    钉钉的机器人消息推送类，封装了常用的消息类型以及加密算法
    需要在配置文件中配置钉钉的机器人的access_token
    如果配置了加签，需要在配置文件中配置钉钉的机器人的secret
    如果配置了自定义关键词，需要在配置文件中配置钉钉的机器人的keyword，多个关键词用英文逗号分隔
    全部的配置文件示例如下, 其中secret和keyword可以不配置, access_token必须配置
    notify:
      dingtalk_access_token: xxxx
      dingtalk_secret: xxxx
    """

    url = "https://oapi.dingtalk.com/robot/send"

    @classmethod
    def _get_access_token(cls):
        """获取钉钉机器人的access_token"""
        if hasattr(cfg.notify, "dingtalk_access_token"):
            return cfg.notify.dingtalk_access_token
        else:
            logger.error(
                "Dingtalk not configured, please add the following items:\n"
                "notify:\n"
                "  dingtalk_access_token: xxxx\n"
                "  dingtalk_secret: xxxx\n"
            )
            raise ConfigError("dingtalk_access_token not found")

    @classmethod
    def _get_secret(cls):
        """获取钉钉机器人的secret"""
        if hasattr(cfg.notify, "dingtalk_secret"):
            return cfg.notify.dingtalk_secret
        else:
            return None

    @classmethod
    def _get_url(cls):
        """获取钉钉机器人的消息推送地址，将签名和时间戳拼接在url后面"""
        access_token = cls._get_access_token()
        url = f"{cls.url}?access_token={access_token}"
        secret = cls._get_secret()
        if secret:
            timestamp, sign = cls._get_sign(secret)
            url = f"{url}&timestamp={timestamp}&sign={sign}"
        return url

    @classmethod
    def _get_sign(cls, secret: str):
        """获取签名发送给钉钉机器人"""
        timestamp = str(round(time.time() * 1000))
        secret_enc = secret.encode("utf-8")
        string_to_sign = "{}\n{}".format(timestamp, secret)
        string_to_sign_enc = string_to_sign.encode("utf-8")
        hmac_code = hmac.new(
            secret_enc, string_to_sign_enc, digestmod=hashlib.sha256
        ).digest()
        sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
        return timestamp, sign

    @classmethod
    def _send(cls, msg):
        """发送消息到钉钉机器人"""
        url = cls._get_url()
        response = httpx.post(url, json=msg, timeout=30)
        if response.status_code != 200:
            logger.error(
                f"failed to send message, content: {msg}, response from Dingtalk: {response.content.decode()}"
            )
            return
        rsp = json.loads(response.content)
        if rsp.get("errcode") != 0:
            logger.error(
                f"failed to send message, content: {msg}, response from Dingtalk: {rsp}"
            )
        return response.content.decode()

    @classmethod
    async def _send_async(cls, msg):
        """发送消息到钉钉机器人"""
        url = cls._get_url()
        async with httpx.AsyncClient() as client:
            r = await client.post(url, json=msg, timeout=30)
            if r.status_code != 200:
                logger.error(
                    f"failed to send message, content: {msg}, response from Dingtalk: {r.content.decode()}"
                )
                return
            rsp = json.loads(r.content)
            if rsp.get("errcode") != 0:
                logger.error(
                    f"failed to send message, content: {msg}, response from Dingtalk: {rsp}"
                )
            return r.content.decode()

    @classmethod
    @deprecated("2.0.0", details="use function `ding` instead")
    def text(cls, content):
        msg = {"text": {"content": content}, "msgtype": "text"}
        return cls._send(msg)


def ding(msg: Union[str, dict]) -> Awaitable:
    """发送消息到钉钉机器人

    支持发送纯文本消息和markdown格式的文本消息。如果要发送markdown格式的消息，请通过字典传入，必须包含包含"title"和"text"两个字段。更详细信息，请见[钉钉开放平台文档](https://open.dingtalk.com/document/orgapp-server/message-type)

    ???+ Important
        必须在异步线程(即运行asyncio loop的线程）中调用此方法，否则会抛出异常。
        此方法返回一个Awaitable，您可以等待它完成，也可以忽略返回值，此时它将作为一个后台任务执行，但完成的时间不确定。

    Args:
        msg: 待发送消息。

    Returns:
        发送消息的后台任务。您可以使用此返回句柄来取消任务。
    """
    if isinstance(msg, str):
        msg_ = {"text": {"content": msg}, "msgtype": "text"}
    elif isinstance(msg, dict):
        msg_ = {
            "msgtype": "markdown",
            "markdown": {"title": msg["title"], "text": msg["text"]},
        }
    else:
        raise TypeError

    task = asyncio.create_task(DingTalkMessage._send_async(msg_))
    return task
