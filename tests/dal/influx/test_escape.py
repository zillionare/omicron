import unittest

from omicron.dal.influx.escape import (
    escape_field_name,
    escape_field_value,
    escape_measurement,
    escape_tag_name,
    escape_tag_value,
)


class EscapeTest(unittest.TestCase):
    def test_escape_measurement(self):
        # measurement
        self.assertEqual(escape_measurement("test"), "test")
        self.assertEqual(escape_measurement("test,test"), "test\\,test")
        self.assertEqual(escape_measurement("test test"), "test\\ test")

        # tag name
        tags = ["test", "test test", "test=test"]
        expected = ["test", "test\\ test", "test\\=test"]
        actual = escape_tag_name(tags)
        self.assertListEqual(expected, actual)

        # tag value
        values = ["test", "test test", "test=test"]
        expected = ["test", "test\\ test", "test\\=test"]
        actual = escape_tag_value(values)
        self.assertListEqual(expected, actual)

        # field name
        keys = ["test", "test test", "test=test"]
        expected = ["test", "test\\ test", "test\\=test"]
        actual = escape_field_name(keys)
        self.assertListEqual(expected, actual)

        # field value
        values = ["test", 'test "test', "test\\test"]
        expected = ["test", 'test \\"test', "test\\\\test"]
        actual = escape_field_value(values)
        self.assertListEqual(expected, actual)

        # field value with backslash
        values = ["test\test", 'test"test']
        expected = ["test\\\\test", 'test\\\\"test']
