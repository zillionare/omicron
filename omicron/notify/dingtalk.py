# -*- coding: utf-8 -*-
import base64
import hashlib
import hmac
import json
import logging
import time
import urllib.parse

import cfg4py
import requests

logger = logging.getLogger(__name__)
cfg = cfg4py.get_instance()


def config_validate(func):
    def inner(cls, *args, **kwargs):
        access_token = cls._get_access_token()
        if not access_token:
            return False
        else:
            return func(cls, *args, **kwargs)

    return inner


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
            return None

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
        response = requests.post(url, json=msg)
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
    @config_validate
    def text(cls, content):
        msg = {"text": {"content": content}, "msgtype": "text"}
        return cls._send(msg)
