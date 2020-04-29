import os
import unittest
import cfg4py

from pyemit import emit

from omicron.core import SecurityType
from omicron.models.security import Security
from omicron.models.securities import Securities
from omicron.core.lang import async_run

from omicron.dal import cache


class MyTestCase(unittest.TestCase):
    @async_run
    async def setUp(self) -> None:
        os.environ[cfg4py.envar] = 'TEST'
        src_dir = os.path.dirname(__file__)
        config_path = os.path.join(src_dir, '../omicron/config')

        cfg = cfg4py.init(config_path)
        await cache.init()
        await emit.start(emit.Engine.REDIS, dsn=cfg.redis.dsn, exchange='zillionare-omega')
        self.securities = Securities()
        await self.securities.load()

    def test_000_properties(self):
        sec = Security('000001.XSHE')
        for key, value in zip('display_name name ipo_date end_date'.split(' '),
                              '平安银行 PAYH 1991-04-03 2200-01-01'.split(' ')):
            self.assertEqual(str(getattr(sec, key)), value)

        sec = Security('399001.XSHE')
        print(sec)

    def test_001_parse_security_type(self):
        codes = [
            '600001.XSHG',  # 浦发银行
            '000001.XSHG',  # 上证指数
            '880001.XSHG',  # 总市值
            '999999.XSHG',  # 上证指数
            '511010.XSHG',  # 国债ETF
            '100303.XSHG',  # 国债0303
            '110031.XSHG',  # 航信转债
            '120201.XSHG',  # 02三峡债
            '000001.XSHE',  # 平安银行
            '300001.XSHE',  # 特锐德
            '399001.XSHE',  # 深成指
            '150001.XSHE',  # 福锐进取
            '131800.XSHE',  # 深圳债券
            '200011.XSHE',  # B股
        ]

        expected = [
            SecurityType.STOCK,
            SecurityType.INDEX,
            SecurityType.INDEX,
            SecurityType.INDEX,
            SecurityType.ETF,
            SecurityType.BOND,
            SecurityType.BOND,
            SecurityType.BOND,
            SecurityType.STOCK,
            SecurityType.STOCK,
            SecurityType.INDEX,
            SecurityType.ETF,
            SecurityType.BOND,
            SecurityType.STOCK_B
        ]

        for i, code in enumerate(codes):
            self.assertEqual(Security.parse_security_type(code), expected[i])


if __name__ == '__main__':
    unittest.main()
