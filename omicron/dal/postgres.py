from gino import Gino

db = Gino()


async def init(dsn: str):
    global db

    await db.set_bind(dsn, min_size=2, max_size=3)


__all__ = ["db", "init"]
