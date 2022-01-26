import datetime
import unittest

from omicron.core.errors import DuplicateOperationError
from omicron.dal.flux import Flux


class FluxTest(unittest.TestCase):
    def test_format_time(self):
        end = datetime.date(1978, 7, 8)
        flux = Flux()

        # all default
        exp = "1978-07-08T00:00:00Z"
        actual = flux.format_time(end)
        self.assertEqual(exp, actual)

        # shift 1 second
        exp = "1978-07-08T00:00:01Z"
        actual = flux.format_time(end, shift_forward=True)
        self.assertEqual(exp, actual)

        # default, to millisecond
        end = datetime.datetime(1978, 7, 8, 12, 34, 56, 123456)
        exp = "1978-07-08T12:34:56.123Z"
        actual = flux.format_time(end, precision="ms")
        self.assertEqual(exp, actual)

    def test_tags(self):
        flux = Flux()

        # default
        expected = 'filter(fn: (r) => r["code"] == "000001.XSHE" or r["code"] == "000002.XSHE")'

        actual = flux.tags({"code": ["000001.XSHE", "000002.XSHE"]}).expressions["tags"]
        self.assertEqual(expected, actual)

        # duplicate
        with self.assertRaises(DuplicateOperationError):
            flux.tags({"code": ["000001", "000002"]})

        # with only one values
        actual = Flux().tags({"code": ["000001.XSHE"]}).expressions["tags"]

        expected = 'filter(fn: (r) => r["code"] == "000001.XSHE")'
        self.assertEqual(expected, actual)

        # with two tags
        actual = (
            Flux()
            .tags({"code": ["000001", "000002"], "name": ["浦发银行"]})
            .expressions["tags"]
        )
        expected = 'filter(fn: (r) => r["code"] == "000001" or r["code"] == "000002" or r["name"] == "浦发银行")'

        self.assertEqual(expected, actual)

        # with empty values
        with self.assertRaises(AssertionError):
            Flux().tags({"code": []})

    def test_fields(self):
        fields = ["open", "close", "high", "low"]
        flux = Flux()
        actual = flux.fields(fields).expressions["fields"]

        exp = 'filter(fn: (r) => r["_field"] == "open" or r["_field"] == "close" or r["_field"] == "high" or r["_field"] == "low")'
        self.assertEqual(exp, actual)

        with self.assertRaises(DuplicateOperationError):
            flux.fields(fields)

    def test_bucket(self):
        flux = Flux()
        actual = flux.bucket("test").expressions["bucket"]
        self.assertEqual('from(bucket: "test")', actual)

        with self.assertRaises(DuplicateOperationError):
            flux.bucket("test")

    def test_measurement(self):
        flux = Flux()
        actual = flux.measurement("test").expressions["measurement"]
        self.assertEqual('filter(fn: (r) => r["_measurement"] == "test")', actual)

        with self.assertRaises(DuplicateOperationError):
            flux.measurement("test")

    def test_limit(self):
        flux = Flux()
        actual = flux.limit(10).expressions["limit"]
        self.assertEqual("limit(n: 10)", actual)

        with self.assertRaises(DuplicateOperationError):
            flux.limit(10)

    def test_range(self):
        flux = Flux()
        start = datetime.date(1973, 3, 18)
        end = datetime.date(1978, 7, 8)
        actual = flux.range(start, end, right_close=False).expressions["range"]
        exp = "range(start: 1973-03-18T00:00:00Z, stop: 1978-07-08T00:00:00Z)"
        self.assertEqual(exp, actual)

        # unsupported precision
        with self.assertRaises(AssertionError):
            Flux().range(start, end, precision="m")

        # duplicate
        with self.assertRaises(DuplicateOperationError):
            flux.range(start, end)

        # right closed
        actual = Flux().range(start, end, precision="ms").expressions["range"]
        exp = "range(start: 1973-03-18T00:00:00.000Z, stop: 1978-07-08T00:00:00.001Z)"
        self.assertEqual(exp, actual)

    def test_flux(self):
        flux = Flux()
        flux.bucket("my-bucket").measurement("stock_bars_1d").tags(
            {"code": ["000001.XSHE", "000002.XSHE"]}
        ).range(datetime.date(2019, 1, 1), datetime.date(2019, 1, 2)).fields(
            ["open", "close", "high", "low"]
        ).limit(
            10
        )

        exp = [
            'from(bucket: "my-bucket")',
            "  |> range(start: 2019-01-01T00:00:00Z, stop: 2019-01-02T00:00:01Z)",
            '  |> filter(fn: (r) => r["_measurement"] == "stock_bars_1d")',
            '  |> filter(fn: (r) => r["code"] == "000001.XSHE" or r["code"] == "000002.XSHE")',
            '  |> filter(fn: (r) => r["_field"] == "open" or r["_field"] == "close" or r["_field"] == "high" or r["_field"] == "low")',
            "  |> limit(n: 10)",
        ]
        actual = str(flux).split("\n")
        self.assertListEqual(exp, actual)
