#!/usr/bin/env python
# -*- coding: utf-8 -*-
import asyncio
import functools


def singleton(cls):
    """Make a class a Singleton class

    Examples:
        >>> @singleton
        ... class Foo:
        ...     # this is a singleton class
        ...     pass

    """
    instances = {}

    @functools.wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


def static_vars(**kwargs):
    def decorator(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorator


def async_concurrent(executors):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            p = functools.partial(func, *args, **kwargs)
            return await asyncio.get_running_loop().run_in_executor(executors, p)

        return wrapper

    return decorator


def async_run(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(func(*args, **kwargs))

    return wrapper
