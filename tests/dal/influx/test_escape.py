import unittest

from omicron.dal.influx.escape import (
    escape,
    key_escape,
    measurement_escape,
    str_escape,
    tag_escape,
)


class EscapeTest(unittest.TestCase):
    def test_escape(self):
        # measurement
        measurements = ["test", "test,test", "test test"]
        actual = [escape(m, measurement_escape) for m in measurements]
        expected = ["test", "test\\,test", "test\\ test"]
        self.assertListEqual(actual, expected)

        # tag name and value
        tags = ["test", "test test", "test=test"]
        expected = ["test", "test\\ test", "test\\=test"]
        actual = [escape(t, tag_escape) for t in tags]
        self.assertListEqual(expected, actual)

        # field key
        keys = ["test", "test test", "test=test"]
        expected = ["test", "test\\ test", "test\\=test"]
        actual = [escape(k, key_escape) for k in keys]
        self.assertListEqual(expected, actual)

        # field value
        values = ["test", 'test "test', "test\\test"]
        expected = ["test", 'test \\"test', "test\\\\test"]
        actual = [escape(v, str_escape) for v in values]
        self.assertListEqual(expected, actual)
