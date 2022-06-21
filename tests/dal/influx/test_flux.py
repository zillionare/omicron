import datetime
import unittest

import numpy as np

from omicron.core.errors import DuplicateOperationError
from omicron.dal.influx.flux import Flux


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
        expected = '  |> filter(fn: (r) => r["code"] == "000001.XSHE" or r["code"] == "000002.XSHE")'

        actual = flux.tags({"code": ["000001.XSHE", "000002.XSHE"]}).expressions["tags"]
        self.assertEqual(expected, actual)

        # duplicate
        with self.assertRaises(DuplicateOperationError):
            flux.tags({"code": ["000001", "000002"]})

        # with only one value
        actual = Flux().tags({"code": ["000001.XSHE"]}).expressions["tags"]

        expected = '  |> filter(fn: (r) => r["code"] == "000001.XSHE")'
        self.assertEqual(expected, actual)

        # with only one value, represented in str
        expected = '  |> filter(fn: (r) => r["code"] == "000001.XSHE")'
        actual = Flux().tags({"code": "000001.XSHE"}).expressions["tags"]
        self.assertEqual(expected, actual)

        # with two tags
        actual = (
            Flux()
            .tags({"code": ["000001.XSHE", "000002.XSHE"], "name": "浦发银行"})
            .expressions["tags"]
        )
        expected = '  |> filter(fn: (r) => r["code"] == "000001.XSHE" or r["code"] == "000002.XSHE" or r["name"] == "浦发银行")'

        self.assertEqual(expected, actual)

        # with empty values
        with self.assertRaises(AssertionError):
            Flux().tags({"code": []})

    def test_fields(self):
        fields = ["open", "close", "high", "low"]
        flux = Flux()
        actual = flux.fields(fields).expressions["fields"]

        exp = '  |> filter(fn: (r) => r["_field"] == "_time" or r["_field"] == "close" or r["_field"] == "high" or r["_field"] == "low" or r["_field"] == "open")'
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
        self.assertEqual('  |> filter(fn: (r) => r["_measurement"] == "test")', actual)

        with self.assertRaises(DuplicateOperationError):
            flux.measurement("test")

    def test_limit(self):
        flux = Flux()
        actual = flux.limit(10).expressions["limit"]
        self.assertEqual("  |> limit(n: 10)", actual)

        with self.assertRaises(DuplicateOperationError):
            flux.limit(10)

    def test_range(self):
        flux = Flux()
        start = datetime.date(1973, 3, 18)
        end = datetime.date(1978, 7, 8)
        actual = flux.range(start, end, right_close=False).expressions["range"]
        exp = "  |> range(start: 1973-03-18T00:00:00Z, stop: 1978-07-08T00:00:00Z)"
        self.assertEqual(exp, actual)

        # unsupported precision
        with self.assertRaises(AssertionError):
            Flux().range(start, end, precision="m")

        # duplicate
        with self.assertRaises(DuplicateOperationError):
            flux.range(start, end)

        # right closed
        actual = Flux().range(start, end, precision="ms").expressions["range"]
        exp = "  |> range(start: 1973-03-18T00:00:00.000Z, stop: 1978-07-08T00:00:00.001Z)"
        self.assertEqual(exp, actual)

    def test_flux(self):
        flux = Flux()
        flux.bucket("my-bucket").measurement("stock_bars_1d").tags(
            {"code": ["000001.XSHE", "000002.XSHE"]}
        ).range(datetime.date(2019, 1, 1), datetime.date(2019, 1, 2)).fields(
            ["open", "close", "high", "low"]
        ).limit(
            10
        ).pivot().group(
            "code"
        )

        exp = [
            'from(bucket: "my-bucket")',
            "  |> range(start: 2019-01-01T00:00:00Z, stop: 2019-01-02T00:00:01Z)",
            '  |> filter(fn: (r) => r["_measurement"] == "stock_bars_1d")',
            '  |> filter(fn: (r) => r["code"] == "000001.XSHE" or r["code"] == "000002.XSHE")',
            '  |> filter(fn: (r) => r["_field"] == "_time" or r["_field"] == "close" or r["_field"] == "high" or r["_field"] == "low" or r["_field"] == "open")',
            '  |> drop(columns: ["_start","_stop","_measurement"])',
            '  |> pivot(columnKey: ["_field"], rowKey: ["_time"], valueColumn: "_value")',
            '  |> group(columns: ["code"])',
            "  |> limit(n: 10)",
        ]
        actual = str(flux).split("\n")
        self.assertListEqual(exp, actual)

        # test all kinds of duplicate
        flux = Flux()
        with self.assertRaises(DuplicateOperationError):
            flux.bucket("my-bucket").bucket("my-bucket")

        with self.assertRaises(DuplicateOperationError):
            flux.measurement("my-measurement").measurement("my-measurement")

        with self.assertRaises(DuplicateOperationError):
            flux.fields(["open", "close", "high", "low"]).fields(["hell"])

        with self.assertRaises(DuplicateOperationError):
            start, end = datetime.date(2019, 1, 1), datetime.date(2019, 1, 2)
            flux.range(start, end).range(start, end)

        with self.assertRaises(DuplicateOperationError):
            flux.limit(10).limit(5)

        with self.assertRaises(DuplicateOperationError):
            flux.pivot().pivot()

        with self.assertRaises(DuplicateOperationError):
            flux.tags({"code": ["000001.XSHE", "000002.XSHE"]}).tags(
                {"code": "000001.XSHE"}
            )

    def test_delete(self):
        cmd = Flux().delete(
            "unittest",
            datetime.datetime(2019, 1, 1),
            {"code": "000001.XSHE", "name": ["平安", "平安银行"]},
        )

        expected = {
            "start": "1970-01-01T00:00:00Z",
            "stop": "2019-01-01T00:00:00Z",
            "predicate": '_measurement="unittest" AND code = "000001.XSHE" AND name = "平安" AND name = "平安银行"',
        }
        self.assertDictEqual(expected, cmd)

    def test_to_timestamp(self):
        tm = "2022-02-08T01:02:03"

        expected = [1644282123, 1644282123000, 1644282123000000]
        for i, p in enumerate(["s", "ms", "us"]):
            tmp1 = expected[i]
            tmp2 = Flux.to_timestamp(tm, p)
            self.assertEqual(tmp1, tmp2)

        tm = datetime.datetime(2022, 2, 8, 1, 2, 3).replace(
            tzinfo=datetime.timezone.utc
        )
        expected = [1644282123, 1644282123000, 1644282123000000]
        for i, p in enumerate(["s", "ms", "us"]):
            self.assertEqual(expected[i], Flux.to_timestamp(tm, p))

        tm = np.datetime64("2022-02-08T01:02:03")
        expected = [1644282123, 1644282123000, 1644282123000000]
        for i, p in enumerate(["s", "ms", "us"]):
            self.assertEqual(expected[i], Flux.to_timestamp(tm, p))

    def test_drop(self):
        start = datetime.datetime(2019, 1, 1)
        end = datetime.datetime(2019, 1, 2)
        flux = (
            Flux()
            .bucket("my-bucket")
            .measurement("stock_bars_1d")
            .drop_sys_cols()
            .range(start, end)
        )

        exp = [
            'from(bucket: "my-bucket")',
            "  |> range(start: 2019-01-01T00:00:00Z, stop: 2019-01-02T00:00:01Z)",
            '  |> filter(fn: (r) => r["_measurement"] == "stock_bars_1d")',
            '  |> drop(columns: ["_start","_stop","_measurement"])',
            '  |> pivot(columnKey: ["_field"], rowKey: ["_time"], valueColumn: "_value")',
        ]

        actual = str(flux).split("\n")
        self.assertListEqual(exp, actual)
