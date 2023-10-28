"""
回测时，打印时间一般要求为回测当时的时间，而非系统时间。这个模块提供了改写日志时间的功能。

使用方法：

```python
from omicron.core.backtestlog import BacktestLogger

logger = BacktestLogger.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()

# 通过bt_date域来设置日期，而不是asctime
handler.setFormatter(Formatter("%(bt_date)s %(message)s"))

logging.basicConfig(level=logging.INFO, handlers=[handler])

# 调用时与普通日志一样，但要增加一个date参数

logger.info("this is info", date=datetime.date(2022, 3, 1))
```
上述代码将输出：

```
2022-03-01 this is info
```

使用本日志的核心是上述代码中的第3行和第9行，最后，在输出日志时加上`date=...`，如第15行所示。

注意在第9行，通常是`logging.getLogger(__nam__)`，而这里是`BacktestLogger.getLogger(__name__)`

如果上述调用中没有传入`date`，则将使用调用时间，此时行为跟原日志系统一致。
!!! warning
    当调用logger.exception时，不能传入date参数。

"""

import datetime
import logging
from typing import Union

from coretypes import Frame


class BacktestLogger(object):
    logger = None

    def __init__(self, name):
        self._log = logging.getLogger(name)

    @classmethod
    def getLogger(cls, name: str):
        if cls.logger is None:
            cls.logger = BacktestLogger(name)

        return cls.logger

    @classmethod
    def debug(cls, msg, *args, date: Union[str, Frame, None] = None):
        cls.logger._log.debug(
            msg, *args, extra={"bt_date": date or datetime.datetime.now()}
        )

    @classmethod
    def info(cls, msg, *args, date: Union[str, Frame, None] = None):
        cls.logger._log.info(
            msg, *args, extra={"bt_date": date or datetime.datetime.now()}
        )

    @classmethod
    def warning(cls, msg, *args, date: Union[str, Frame, None] = None):
        cls.logger._log.warning(
            msg, *args, extra={"bt_date": date or datetime.datetime.now()}
        )

    @classmethod
    def error(cls, msg, *args, date: Union[str, Frame, None] = None):
        cls.logger._log.error(
            msg, *args, extra={"bt_date": date or datetime.datetime.now()}
        )

    @classmethod
    def critical(cls, msg, *args, date: Union[str, Frame, None] = None):
        cls.logger._log.critical(
            msg, *args, extra={"bt_date": date or datetime.datetime.now()}
        )

    @classmethod
    def exception(cls, e):
        cls.logger._log.exception(e, extra={"bt_date": ""})

    @classmethod
    def log(cls, level, msg, *args, date: Union[str, Frame, None] = None):
        cls.logger._log.log(
            level, msg, *args, extra={"bt_date": date or datetime.datetime.now()}
        )

    @classmethod
    def setLevel(cls, level):
        cls.logger._log.setLevel(level)