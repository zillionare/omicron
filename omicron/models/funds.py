import numpy as np
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.types import Date, Float, Integer, String, BigInteger

from omicron import db


class Funds(db.Model):
    __tablename__ = "funds"

    id = db.Column(Integer, primary_key=True)
    code = db.Column(String, nullable=False)
    name = db.Column(String)
    advisor = db.Column(String)
    trustee = db.Column(String)
    operate_mode_id = db.Column(Integer)
    operate_mode = db.Column(String)
    start_date = db.Column(Date, nullable=False)
    end_date = db.Column(Date, nullable=False)

    @classmethod
    async def save(cls, recs: np.array):
        page = 0
        result = []
        while page * 3000 < len(recs):
            _recs = recs[page * 3000 : (page + 1) * 3000]
            data = [dict(zip(recs.dtype.names, x)) for x in _recs]
            qs = insert(cls.__table__).values(data)
            _result = await (
                qs.on_conflict_do_update(
                    index_elements=[cls.code],
                    set_={col: qs.excluded[col] for col in recs.dtype.names},
                )
                .returning(cls.id)
                .gino.all()
            )
            result.extend(_result)
            page += 1
        return result


class FundNetValue(db.Model):
    __tablename__ = "fund_net_value"

    id = db.Column(Integer, primary_key=True)
    code = db.Column(String)
    net_value = db.Column(Float)
    sum_value = db.Column(Float)
    factor = db.Column(Float)
    acc_factor = db.Column(Float)
    refactor_net_value = db.Column(Float)
    day = db.Column(Date)

    @classmethod
    async def save(cls, recs: np.array):
        page = 0
        result = []
        while page * 3000 < len(recs):
            _recs = recs[page * 3000 : (page + 1) * 3000]
            data = [dict(zip(recs.dtype.names, x)) for x in _recs]
            qs = insert(cls.__table__).values(data)
            result.extend(
                await (
                    qs.on_conflict_do_update(
                        index_elements=[cls.code, cls.day],
                        set_={col: qs.excluded[col] for col in recs.dtype.names},
                    )
                    .returning(cls.id)
                    .gino.all()
                )
            )
            page += 1
        return result


class FundShareDaily(db.Model):
    __tablename__ = "fund_share_daily"

    id = db.Column(Integer, primary_key=True)
    code = db.Column(String)
    name = db.Column(String)
    exchange_code = db.Column(String)
    date = db.Column(Date)
    shares = db.Column(BigInteger)

    @classmethod
    async def save(cls, recs: np.array):
        page = 0
        result = []
        while page * 3000 < len(recs):
            _recs = recs[page * 3000 : (page + 1) * 3000]
            data = [dict(zip(recs.dtype.names, x)) for x in _recs]
            qs = insert(cls.__table__).values(data)
            result.extend(
                await (
                    qs.on_conflict_do_update(
                        index_elements=[cls.code, cls.date],
                        set_={col: qs.excluded[col] for col in recs.dtype.names},
                    )
                    .returning(cls.id)
                    .gino.all()
                )
            )
            page += 1
        return result


class FundPortfolioStock(db.Model):
    __tablename__ = "fund_portfolio_stock"

    id = db.Column(Integer, primary_key=True)
    code = db.Column(String)
    period_start = db.Column(Date, nullable=False)
    period_end = db.Column(Date, nullable=False)
    pub_date = db.Column(Date, nullable=False)
    report_type_id = db.Column(Integer)
    report_type = db.Column(String)
    rank = db.Column(Integer)
    symbol = db.Column(String)
    name = db.Column(String)
    shares = db.Column(Float)
    market_cap = db.Column(Float)
    proportion = db.Column(Float)

    @classmethod
    async def save(cls, recs: np.array):
        page = 0
        result = []
        while page * 1000 < len(recs):
            _recs = recs[page * 1000 : (page + 1) * 1000]
            data = [dict(zip(recs.dtype.names, x)) for x in _recs]
            qs = insert(cls.__table__).values(data)
            result.extend(
                await (
                    qs.on_conflict_do_update(
                        index_elements=[
                            cls.code,
                            cls.pub_date,
                            cls.symbol,
                            cls.report_type,
                        ],
                        set_={col: qs.excluded[col] for col in recs.dtype.names},
                    )
                    .returning(cls.id)
                    .gino.all()
                )
            )
            page += 1
        return result
