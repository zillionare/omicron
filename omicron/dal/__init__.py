#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .cache import RedisCache

cache = RedisCache()

__all__ = ['cache']
