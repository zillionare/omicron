from gino import Gino

db = Gino()


async def init(dsn: str):
    global db

    await db.set_bind(dsn)


__all__ = ["db", "init"]
