# noqa
from typing import Optional


class Config(object):
    __access_counter__ = 0

    def __cfg4py_reset_access_counter__(self):
        self.__access_counter__ = 0

    def __getattribute__(self, name):
        obj = object.__getattribute__(self, name)
        if name.startswith("__") and name.endswith("__"):
            return obj

        if callable(obj):
            return obj

        self.__access_counter__ += 1
        return obj

    def __init__(self):
        raise TypeError("Do NOT instantiate this class")

    tz: Optional[str] = None

    class redis:
        dsn: Optional[str] = None

    class postgres:
        dsn: Optional[str] = None

        enabled: Optional[bool] = None

    class omega:
        home: Optional[str] = None

        class urls:
            quotes_server: Optional[str] = None
