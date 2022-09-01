import unittest

from omicron.core.errors import ServiceNotReadyError


class ErrorsTest(unittest.TestCase):
    def test_service_not_ready_error(self):
        try:
            raise ServiceNotReadyError("this is a test", "arg1", 0)
        except ServiceNotReadyError as e:
            print(e)
