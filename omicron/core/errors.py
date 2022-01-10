#!/usr/bin/env python
# -*- coding: utf-8 -*-


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
