#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .cache import RedisCache, RedisDB

cache = RedisCache()

__all__ = ['cache', 'RedisDB']
