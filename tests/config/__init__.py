#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors:

"""
import os
from importlib.metadata import version

import cfg4py
import pytest
from pytest import fixture


def get_config_dir():
    return os.path.dirname(__file__)


def endpoint():
    cfg = cfg4py.get_instance()

    major, minor, *_ = version("zillionare-omega").split(".")
    prefix = cfg.server.prefix.rstrip("/")
    return f"{prefix}/v{major}.{minor}"
