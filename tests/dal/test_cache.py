import unittest

import cfg4py

from omicron.dal.cache import cache
from tests.config import get_config_dir


class CacehTest(unittest.IsolatedAsyncioTestCase):
    async def test_cache(self):
        cfg4py.init(get_config_dir())
        await cache.init()
        assert "_security_" == (await cache.security.get("__meta__.database"))
        await cache.close()

        with self.assertRaises(AssertionError):
            cache.security.get("__meta__.database")
