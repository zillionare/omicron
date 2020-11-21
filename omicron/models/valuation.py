import datetime
from typing import List, Union

import numpy as np
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.types import Date, Float, Integer, Numeric, String

import omicron
from omicron import db
from omicron.client.quotes_fetcher import get_valuation


class Valuation(db.Model):
    __tablename__ = "valuation"
    id = db.Column(Integer, primary_key=True)
    code = db.Column(String, nullable=False)
    pe = db.Column(Float)
    turnover = db.Column(Float)
    pb = db.Column(Float)
    ps = db.Column(Float)
    pcf = db.Column(Float)
    capital = db.Column(Numeric)
    market_cap = db.Column(Numeric)
    circulating_cap = db.Column(Numeric)
    circulating_market_cap = db.Column(Numeric)
    pe_lyr = db.Column(Float)
    frame = db.Column(Date, nullable=False)

    types = {
        "code": "O",
        "pe": "f4",
        "turnover": "f4",
        "pb": "f4",
        "ps": "f4",
        "pcf": "f4",
        "capital": "f4",
        "market_cap": "f4",
        "circulating_cap": "f4",
        "circulating_market_cap": "f4",
        "pe_lyr": "f4",
        "frame": "O",
    }

    @classmethod
    async def get(
        cls,
        codes: Union[List[str], str],
        frame: datetime.date,
        fields: List[str] = None,
        n: int = 1,
    ) -> np.array:
        """获取一支或者多支证券的直到`date`的`n`条数据

        尽管本函数提供了同时查询多支证券、多个日期市值数据的能力，但为后续处理方便，建议一次仅
        查询多支证券同一日的数据；或者一支证券多日的数据。

        请调用者保证截止`date`日，证券已存在`n`条市值数据。否则，每次调用都会产生一次无效的数据
        库查询：函数在查询数据库后，得不到满足条件的n条数据（无论此前数据是否存到本地，都不满足
        此条件），查询无效，还要再次请求上游服务器，但上游服务器的答复数据很可能就是在数据库中
        已存在的数据。

        无论查询条件如果，返回数据均为numpy structured array。证券代码和日期为该array的index,
        记录按date字段升序排列。有多只证券的，证券之间顺序由上游服务器决定。
        Args:
            codes (Union[List[str], str]): [description]
            frame (datetime.date): [description]
            fields (List[str]): if None, then returns all columns/fields from
            database/remote
            n (int):

        Returns:
            np.array: 返回数据为numpy structured array数组，包含以下字段:
            "code", "pe","turnover","pb","ps","pcf","capital","market_cap",
            "circulating_cap","circulating_market_cap","pe_lyr", "date",
        """
        if omicron.has_db():
            fields = fields or [
                "code",
                "pe",
                "turnover",
                "pb",
                "ps",
                "pcf",
                "capital",
                "market_cap",
                "circulating_cap",
                "circulating_market_cap",
                "pe_lyr",
                "frame",
            ]

            if isinstance(codes, str):
                codes = [codes]

            # 通过指定要查询的字段（即使是全部字段），避免了生成Valuation对象
            query = (
                cls.select(*fields).where(cls.code.in_(codes)).where(cls.frame <= frame)
            )
            query = query.order_by(cls.frame.desc()).limit(len(codes) * n)

            records = await query.gino.all()
            if records and len(records) == n * len(codes) and records[0].frame == frame:
                return cls.to_numpy(records, fields)[::-1]

        # if no db connection, or no result from database, then try remote fetch
        return await get_valuation(codes, frame, fields, n)

    @classmethod
    def to_numpy(cls, records: List, fields: List[str]) -> np.array:
        """将数据库返回的查询结果转换为numpy structured array

        Args:
            records (List): [description]
            keys (List[str]): [description]

        Returns:
            np.array: [description]
        """
        dtypes = [(name, cls.types[name]) for name in fields]
        return np.array(
            [tuple(rec[name] for name in fields) for rec in records], dtype=dtypes
        )

    @classmethod
    async def save(cls, recs: np.array):
        data = [dict(zip(recs.dtype.names, x)) for x in recs]
        qs = insert(cls.__table__).values(data)
        return await (
            qs.on_conflict_do_update(
                index_elements=[cls.code, cls.frame],
                set_={col: qs.excluded[col] for col in recs.dtype.names},
            )
            .returning(cls.id)
            .gino.all()
        )

    @classmethod
    async def get_circulating_cap(cls, code: str, frame: datetime.date, n: int):
        fields = ["frame", "circulating_cap"]

        return await cls.get(code, frame, fields, n)

    @classmethod
    async def truncate(cls):
        """truncate table in database."""
        # db.bind: `https://python-gino.org/docs/en/master/explanation/engine.html`
        await db.bind.status(f"truncate table {cls.__tablename__}")
