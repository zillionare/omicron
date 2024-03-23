import logging
from asyncio import Lock

import cfg4py
import redis.asyncio as redis
from redis.asyncio import Redis

logger = logging.getLogger(__file__)



class RedisCache:
    databases = ["_sys_", "_security_", "_temp_", "_feature_"]

    _security_: Redis|None = None
    _sys_: Redis|None = None
    _temp_: Redis|None = None
    _feature_: Redis|None = None
    _app_: Redis|None = None

    _initialized = False
    _cache_lock = Lock()

    @property
    def security(self) -> Redis:
        """represents security database, include today's bars, security info and etc"""
        assert self._security_
        return self._security_

    @property
    def sys(self) -> Redis:
        """represents sys database, which stores jobs, status and etc."""
        assert self._sys_
        return self._sys_

    @property
    def temp(self) -> Redis:
        """represents the temp database"""
        assert self._temp_
        return self._temp_

    @property
    def feature(self) -> Redis:
        """represents the feature database"""
        assert self._feature_
        return self._feature_

    @property
    def app(self) -> Redis:
        """represent the app database"""
        assert self._app_
        return self._app_

    async def close(self):
        """shutdown and close the connection"""
        async with self._cache_lock:
            if self._initialized is False:
                return True

            logger.info("closing redis cache...")
            if self._sys_ is not None:
                await self._sys_.aclose()
                self._sys_ = None

            if self._security_ is not None:
                await self._security_.aclose()
                self._security_ = None

            if self._temp_ is not None:
                await self._temp_.aclose()
                self._temp_ = None

            if self._feature_ is not None:
                await self._feature_.aclose()
                self._feature_ = None

            if self._app_ is not None:
                await self._app_.aclose()
                self._app_ = None

            self._initialized = False
            logger.info("redis caches are all closed")

    async def init(self, app: int = 5):
        """初始化"""
        async with self._cache_lock:
            if self._initialized:
                return True

            logger.info("init redis cache...")
            cfg = cfg4py.get_instance()
            for i, name in enumerate(self.databases):
                auto_decode = True
                if name == "_temp_":
                    auto_decode = False
                db = redis.from_url(
                    cfg.redis.dsn,
                    encoding="utf-8",
                    decode_responses=auto_decode,
                    max_connections=10,
                    db=i,
                )
                await db.set("__meta__.database", name)
                setattr(self, name, db)

            # init app pool
            if app < 5 or app > 15:
                app = 5
            db = redis.from_url(
                cfg.redis.dsn,
                encoding="utf-8",
                decode_responses=True,
                max_connections=10,
                db=app,
            )
            await db.set("__meta__.database", "__app__")
            setattr(self, "_app_", db)

            self._initialized = True
            logger.info("redis cache is inited")

        return True


cache = RedisCache()
__all__ = ["cache"]
