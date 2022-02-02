import unittest

from omicron.dal.influx.escape import (
    KEY_ESCAPE,
    MEASUREMENT_ESCAPE,
    STR_ESCAPE,
    TAG_ESCAPE,
    escape,
)


class EscapeTest(unittest.TestCase):
    def test_escape(self):
        # measurement
        measurements = ["test", "test,test", "test test"]
        actual = [escape(m, MEASUREMENT_ESCAPE) for m in measurements]
        expected = ["test", "test\\,test", "test\\ test"]
        self.assertListEqual(actual, expected)

        # tag name and value
        tags = ["test", "test test", "test=test"]
        expected = ["test", "test\\ test", "test\\=test"]
        actual = [escape(t, TAG_ESCAPE) for t in tags]
        self.assertListEqual(expected, actual)

        # field key
        keys = ["test", "test test", "test=test"]
        expected = ["test", "test\\ test", "test\\=test"]
        actual = [escape(k, KEY_ESCAPE) for k in keys]
        self.assertListEqual(expected, actual)

        # field value
        values = ["test", 'test "test', "test\\test"]
        expected = ["test", 'test \\"test', "test\\\\test"]
        actual = [escape(v, STR_ESCAPE) for v in values]
        self.assertListEqual(expected, actual)
