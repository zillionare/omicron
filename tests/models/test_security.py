import datetime
import unittest

import arrow
import cfg4py

import omicron
from omicron.dal import cache
from omicron.dal.influx.influxclient import InfluxClient
from omicron.models.security import Security, convert_nptime_to_datetime
from omicron.models.stock import Stock
from tests import init_test_env, set_security_data_to_db

cfg = cfg4py.get_instance()


class SecurityTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        await init_test_env()
        await omicron.init()

        url, token, bucket, org = (
            cfg.influxdb.url,
            cfg.influxdb.token,
            cfg.influxdb.bucket_name,
            cfg.influxdb.org,
        )
        self.client = InfluxClient(url, token, bucket, org)

        # 关于数据准备，请参阅: data/README.md
        await Stock.reset_cache()
        await self.client.delete_bucket()
        await self.client.create_bucket()
        await set_security_data_to_db(self.client)

        return await super().asyncSetUp()

    async def asyncTearDown(self) -> None:
        await omicron.close()
        return await super().asyncTearDown()

    async def test_select_eval(self):
        dt = datetime.date(2022, 5, 20)
        query = Security.select(dt)
        query.types([]).exclude_st().exclude_kcb()
        codes = set(await query.eval())

        exp = {"000001.XSHE", "300001.XSHE", "600000.XSHG", "000001.XSHG"}
        self.assertSetEqual(exp, codes)

        # 允许ST
        query = Security.select(dt)
        query.types([]).exclude_kcb()
        codes = set(await query.eval())
        exp = {
            "000001.XSHE",
            "000005.XSHE",
            "300001.XSHE",
            "600000.XSHG",
            "000007.XSHE",
            "000001.XSHG",
        }
        self.assertSetEqual(exp, codes)

        # 允许科创板
        query = Security.select(dt)
        query.types([]).exclude_st()
        codes = set(await query.eval())
        exp = {
            "000001.XSHE",
            "300001.XSHE",
            "600000.XSHG",
            "688001.XSHG",
            "000001.XSHG",
        }
        self.assertSetEqual(exp, codes)

        # stock only
        query = Security.select(dt)
        query.types(["stock"]).exclude_st().exclude_kcb()
        codes = set(await query.eval())

        exp = {"000001.XSHE", "300001.XSHE", "600000.XSHG"}
        self.assertSetEqual(exp, codes)

        # index only
        query = Security.select(dt)
        query.types(["index"]).exclude_st().exclude_kcb()
        codes = set(await query.eval())

        exp = {"000001.XSHG"}
        self.assertSetEqual(exp, codes)

        # 排除创业板
        query = Security.select(dt)
        query.types([]).exclude_cyb().exclude_st().exclude_kcb()
        codes = set(await query.eval())

        exp = {"000001.XSHE", "000001.XSHG", "600000.XSHG"}
        self.assertSetEqual(exp, codes)

    async def test_choose_cyb(self):
        dt = datetime.date(2022, 5, 20)
        query = Security.select(dt)
        query.types([]).exclude_st().only_cyb()
        actual = await query.eval()
        self.assertListEqual(["300001.XSHE"], actual)

        # to check if we could omit `types` method
        query.exclude_st().only_cyb()
        actual = await query.eval()
        self.assertListEqual(["300001.XSHE"], actual)

    async def test_choose_kcb(self):
        dt = datetime.date(2022, 5, 20)

        query = Security.select(dt)
        query.types([]).exclude_st().only_kcb()
        actual = await query.eval()
        self.assertListEqual(["688001.XSHG"], actual)

    async def test_eval(self):
        dt = datetime.date(2022, 5, 20)

        query = Security.select()
        query.types([]).exclude_st().exclude_kcb()
        codes = set(await query.eval())
        exp = {"000001.XSHE", "300001.XSHE", "600000.XSHG", "000001.XSHG"}
        self.assertSetEqual(exp, codes)

        query = Security.select()
        query.types([]).only_st()
        codes = set(await query.eval())
        self.assertSetEqual({"000005.XSHE", "000007.XSHE"}, codes)

        query = Security.select()
        query.types([]).include_exit().name_like("DM")
        codes = set(await query.eval())
        self.assertSetEqual({"000406.XSHE"}, codes)

        query = Security.select()
        query.types([]).alias_like("银行")
        codes = set(await query.eval())
        self.assertSetEqual({"000001.XSHE", "600000.XSHG"}, codes)

    async def test_eval_2(self):
        # dt = datetime.date(2022, 5, 20)
        await cache.security.delete("security:latest_date")

        query = Security.select()
        query.types([]).exclude_st().exclude_kcb()
        codes = set(await query.eval())
        exp = {"000001.XSHE", "300001.XSHE", "600000.XSHG", "000001.XSHG"}
        self.assertSetEqual(exp, codes)

        await cache.security.set("security:latest_date", "2022-05-20")

    async def test_query_info(self):
        dt = datetime.date(2022, 5, 20)
        rc = await Security.info("688001.XSHG", dt)
        self.assertEqual(rc["display_name"], "华兴源创")
        self.assertEqual(rc["alias"], "华兴源创")

        rc = await Security.name("688001.XSHG", dt)
        self.assertEqual(rc, "HXYC")

        rc = await Security.alias("688001.XSHG", dt)
        self.assertEqual(rc, "华兴源创")
        rc = await Security.display_name("688001.XSHG", dt)
        self.assertEqual(rc, "华兴源创")

        rc = await Security.start_date("688001.XSHG", dt)
        self.assertEqual(rc, datetime.date(2019, 7, 22))
        rc = await Security.end_date("688001.XSHG", dt)
        self.assertEqual(rc, datetime.date(2200, 1, 1))

        rc = await Security.security_type("688001.XSHG", dt)
        self.assertEqual(rc, "stock")

        rc = await Security.security_type("688001.XSHG", None)
        self.assertEqual(rc, "stock")

    def test_fuzzy_match_ex(self):
        exp = set(["600000.XSHG"])
        self.assertSetEqual(exp, set(Security.fuzzy_match_ex("600").keys()))

        exp = set(["000001.XSHE", "600000.XSHG"])
        self.assertSetEqual(exp, set(Security.fuzzy_match_ex("P").keys()))

        exp = set(["000001.XSHE"])
        self.assertSetEqual(exp, set(Security.fuzzy_match_ex("平").keys()))

    async def test_update_securities(self):
        stocks = [
            ("000001.XSHE", "平安银行", "PAYH", "1991-04-03", "2200-01-01", "stock"),
            ("000001.XSHG", "上证指数", "SZZS", "1991-07-15", "2200-01-01", "index"),
            ("000406.XSHE", "石油大明", "SYDM", "1996-06-28", "2006-04-20", "stock"),
            ("000005.XSHE", "ST星源", "STXY", "1990-12-10", "2200-01-01", "stock"),
            ("300001.XSHE", "特锐德", "TRD", "2009-10-30", "2200-01-01", "stock"),
            ("600000.XSHG", "浦发银行", "PFYH", "1999-11-10", "2200-01-01", "stock"),
            ("688001.XSHG", "华兴源创", "HXYC", "2019-07-22", "2200-01-01", "stock"),
            ("000007.XSHE", "*ST全新", "*STQX", "1992-04-13", "2200-01-01", "stock"),
        ]

        dt = arrow.now().naive
        await Security.update_secs_cache(dt, stocks)

        # make sure no duplicate
        await Security.update_secs_cache(dt, stocks)

        stocks = await Security.load_securities()
        self.assertEqual(len(stocks), 8)
        t1 = convert_nptime_to_datetime(stocks[0]["ipo"]).date()
        self.assertEqual(t1, datetime.date(1991, 4, 3))

    async def test_save_securitie(self):
        dt = datetime.date(2022, 8, 1)
        secs = [["000001.XSHE", "平安银行", "PAYH", "1991-04-03", "2200-01-01", "stock"]]
        await Security.save_securities(secs, dt)

        items = await Security.load_securities_from_db(dt, "000001.XSHE")
        sec_data = items[0]
        self.assertEqual(sec_data[0], "000001.XSHE")

        dt = datetime.date(2022, 8, 2)
        secs = [["000001.XSHE", "平安银行", "PAYH", "1991-04-03", "2200-01-01", "stock"]]
        await Security.save_securities(secs, dt)

        dt1, dt2 = await Security.get_datescope_from_db()
        self.assertEqual(dt1, datetime.date(2022, 5, 20))
        self.assertEqual(dt2, datetime.date(2022, 8, 2))

        rc = await Security.get_security_types()
        self.assertTrue(rc)

        await cache.security.set("security:latest_date", "2022-08-02")
        query = Security.select(dt)
        query.types([]).name_like("PAY")
        codes = set(await query.eval())
        exp = {"000001.XSHE"}
        self.assertSetEqual(exp, codes)
        await cache.security.set("security:latest_date", "2022-05-20")

    async def test_save_xrxd(self):
        dt = datetime.date(2022, 8, 1)
        # code(0), a_xr_date, board_plan_bonusnote, bonus_ratio_rmb(3), dividend_ratio, transfer_ratio(5),
        # at_bonus_ratio_rmb(6), report_date, plan_progress, implementation_bonusnote, bonus_cancel_pub_date(10)

        secs = [
            [
                "000001.XSHE",
                datetime.date(2022, 8, 1),
                "note",
                10.0,
                5.0,
                0,
                0,
                datetime.date(2021, 12, 31),
                "progress",
                "impl note",
                datetime.date(2099, 1, 1),
            ]
        ]
        await Security.save_xrxd_reports(secs, dt)

        secs = [
            [
                "600000.XSHG",
                datetime.date(2022, 8, 1),
                "流通",
                10.0,
                5.0,
                0,
                0,
                datetime.date(2021, 12, 31),
                "progress",
                "impl note",
                datetime.date(2099, 1, 1),
            ],
            [
                "000001.XSHE",
                datetime.date(2022, 8, 2),
                "note",
                10.0,
                5.0,
                0,
                0,
                datetime.date(2021, 12, 31),
                "progress",
                "impl note",
                datetime.date(2099, 1, 1),
            ],
        ]
        await Security.save_xrxd_reports(secs, dt)

        secs = [
            [
                "000406.XSHE",
                datetime.date(2022, 8, 1),
                "note",
                10.0,
                5.0,
                0,
                0,
                datetime.date(2021, 12, 31),
                "progress",
                "impl note",
                datetime.date(2022, 8, 1),
            ]
        ]
        await Security.save_xrxd_reports(secs, dt)

        items = await Security.get_xrxd_info(dt, "000001.XSHE")
        item = items[0]
        self.assertEqual(item["xr_date"], dt)

    async def test_get_stock(self):
        item = Security.get_stock("000001.XSHE")
        self.assertEqual(item["alias"], "平安银行")
