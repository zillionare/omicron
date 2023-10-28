import datetime
import logging
import unittest
from logging import Formatter

from omicron.core.backtestlog import BacktestLogger

logger = BacktestLogger.getLogger(__name__)
logger.setLevel(logging.INFO)

root = logging.getLogger()
root.handlers.clear()
handler = logging.StreamHandler()
handler.setFormatter(Formatter("%(bt_date)s %(message)s"))

logging.basicConfig(level=logging.INFO, handlers=[handler])


class BacktestLoggerTest(unittest.TestCase):
    def test_log(self):
        logger.info("this is info", date=datetime.date(2022, 3, 1))
        logger.debug("this is debug", date=datetime.date(2022, 3, 2))
        logger.warning("this is warning", date=datetime.date(2022, 3, 3))
        logger.error("this is error", date=datetime.date(2022, 3, 4))
        logger.critical("this is critical", date=datetime.date(2022, 3, 6))
        logger.log(
            logging.INFO, "this is log with info level", date=datetime.date(2022, 3, 7)
        )
        logger.info("this is info without date, using current time.")
        logger.info("log with args %s", "arg1", date=datetime.date(2022, 3, 8))

        try:
            raise ValueError("this is an ** MOCK ** error")
        except ValueError as e:
            logger.exception(e)
