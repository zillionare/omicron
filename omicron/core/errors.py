#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Any


class Error(Exception):
    """错误基类"""

    def __init__(self, message: str, *args):
        self.message = message
        self.args = args

    def __str__(self):
        return f"{self.message}: {self.args}"


class ServiceNotReadyError(Error):
    """服务未就绪错误，比如依赖的服务未启动"""

    pass


class DataNotReadyError(Error):
    """数据未就绪错误，比如依赖的数据还未下载到本地，或者未加载到内存中"""

    pass


class DuplicateOperationError(Error):
    """重复操作错误。"""

    pass


class SerializationError(Error):
    """序列化错误。"""

    pass


class BadParameterError(Error):
    """函数传入了错误的参数、参数个数或者参数值。"""

    pass


class EmptyResult(Error):
    """返回了空的查询结果

    Args:
        Error : [description]
    """

    def __init__(self, msg="return empty result"):
        super().__init__(msg)

    pass


class ConfigError(Error):
    """错误的配置"""

    pass
