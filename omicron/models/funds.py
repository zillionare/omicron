import datetime
from typing import List, Union

import numpy as np
import sqlalchemy
from coretypes import FrameType
from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.types import BigInteger, Date, Float, Integer, String

from omicron import db
from omicron.models.timeframe import TimeFrame


class Funds(db.Model):
    __tablename__ = "funds"

    id = db.Column(Integer, primary_key=True)
    code = db.Column(String, nullable=False)
    name = db.Column(String)
    advisor = db.Column(String)
    trustee = db.Column(String)
    operate_mode_id = db.Column(Integer)
    operate_mode = db.Column(String)
    underlying_asset_type_id = db.Column(Integer)
    underlying_asset_type = db.Column(String)
    start_date = db.Column(Date, nullable=False)
    end_date = db.Column(Date, nullable=False)
    total_tna = db.Column(BigInteger)
    net_value = db.Column(Float)
    quote_change_weekly = db.Column(Float)
    quote_change_monthly = db.Column(Float)

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

    @classmethod
    async def get(
        cls,
        name: str = None,
        code: Union[str, List[str]] = None,
        operate_mode_ids: List[int] = None,
        total_tna_min: Union[int, float] = None,
        total_tna_max: Union[int, float] = None,
        position_stock: str = None,
        underlying_asset_type: str = None,
        position_symbol: List[str] = None,
        fund_range: int = None,
        position_stock_percent: float = None,
        orders: List[dict] = None,
        page: int = 1,
        page_size: int = 10,
    ):
        if code and isinstance(code, str):
            code = [code]
        columns = [
            "id",
            "code",
            "name",
            "advisor",
            "trustee",
            "operate_mode_id",
            "operate_mode",
            "start_date",
            "end_date",
            "total_tna",
            "net_value",
            "quote_change_monthly",
            "quote_change_weekly",
            "underlying_asset_type_id",
            "underlying_asset_type",
        ]
        fund_q = Funds.select(*columns)
        deadline = (await db.func.max(FundPortfolioStock.deadline).gino.all())[0][0]
        q = (
            db.select(
                [FundPortfolioStock.symbol, func.count(FundPortfolioStock.symbol)]
            )
            .where(FundPortfolioStock.deadline == deadline)
            .group_by(FundPortfolioStock.symbol)
            .having(func.count(FundPortfolioStock.symbol) == 1)
        )
        stocks = await q.gino.all()
        stock_symbols = [stock[0] for stock in stocks]
        if position_stock or position_symbol or fund_range or position_stock_percent:
            stock_q = FundPortfolioStock.select("code")
            if fund_range == 1:
                stock_q = stock_q.where(FundPortfolioStock.symbol.in_(stock_symbols))
            if position_stock_percent:
                q = (
                    db.select(
                        [
                            FundPortfolioStock.code,
                            func.sum(FundPortfolioStock.proportion),
                        ]
                    )
                    .where(FundPortfolioStock.deadline == deadline)
                    .group_by(FundPortfolioStock.code)
                    .having(
                        func.sum(FundPortfolioStock.proportion)
                        >= position_stock_percent
                    )
                )
                stocks = await q.gino.all()
                fund_codes = [stock[0] for stock in stocks]
                stock_q = stock_q.where(FundPortfolioStock.code.in_(fund_codes))
            if position_stock:
                stock_q = stock_q.where(
                    FundPortfolioStock.name.like(f"%{position_stock}%")
                )
            if position_symbol:
                stock_q = stock_q.where(FundPortfolioStock.symbol.in_(position_symbol))
            if code:
                stock_q = stock_q.where(FundPortfolioStock.code.in_(code))
            stock_records = await stock_q.where(
                FundPortfolioStock.deadline == deadline
            ).gino.all()
            code = [stock_record["code"] for stock_record in stock_records]
        if code:
            fund_q = fund_q.where(Funds.code.in_(code))
        if name:
            fund_q = fund_q.where(Funds.name.like(f"%{name}%"))
        if operate_mode_ids:
            fund_q = fund_q.where(Funds.operate_mode_id.in_(operate_mode_ids))
        if total_tna_min:
            fund_q = fund_q.where(Funds.total_tna >= total_tna_min)
        if total_tna_max:
            fund_q = fund_q.where(Funds.total_tna <= total_tna_max)
        if underlying_asset_type:
            fund_q = fund_q.where(
                Funds.underlying_asset_type_id == underlying_asset_type
            )

        for order in orders or []:
            fund_q = fund_q.order_by(
                getattr(sqlalchemy, order.get("order"))(
                    getattr(Funds, order.get("field"))
                )
            )

        fund_records = (
            await fund_q.offset((page - 1) * page_size).limit(page_size).gino.all()
        )
        fund_results = []
        for record in fund_records or []:
            result = {}
            for column in columns:
                result[column] = record[column]
            result["start_date"] = record["start_date"] and record[
                "start_date"
            ].strftime("%Y-%m-%d")
            result["end_date"] = record["end_date"] and record["end_date"].strftime(
                "%Y-%m-%d"
            )
            stocks = await FundPortfolioStock.get(
                record["code"], deadline=deadline, stock_symbols=stock_symbols
            )
            result["stocks"] = stocks
            fund_results.append(result)
        count = await fund_q.alias("q").count().gino.all()
        return {"count": count[0].values()[0], "items": fund_results}


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
    async def save(cls, recs: np.array, day: datetime.date = None):
        page = 0
        result = []

        today = day or datetime.date.today()
        last_week_trade_day = TimeFrame.floor(
            today - datetime.timedelta(today.weekday() + 1), frame_type=FrameType.DAY
        )
        last_month_trade_day = TimeFrame.floor(
            datetime.date(today.year, today.month, 1) - datetime.timedelta(1),
            frame_type=FrameType.DAY,
        )
        last_week_closing_price_records = (
            await cls.select("code", "net_value")
            .where(cls.day == last_week_trade_day)
            .gino.all()
        )
        last_week_closing_prices = {
            record["code"]: record["net_value"]
            for record in last_week_closing_price_records
        }
        last_month_closing_price_records = (
            await cls.select("code", "net_value")
            .where(cls.day == last_month_trade_day)
            .gino.all()
        )
        last_month_closing_prices = {
            record["code"]: record["net_value"]
            for record in last_month_closing_price_records
        }
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
            for _rec in data:
                net_value = _rec.get("net_value") or 0

                last_week_closing_price = last_week_closing_prices.get(_rec.get("code"))
                last_month_closing_price = last_month_closing_prices.get(
                    _rec.get("code")
                )
                quote_change_weekly = (
                    0
                    if not last_week_closing_price
                    else (net_value - last_week_closing_price) / last_week_closing_price
                )
                quote_change_monthly = (
                    0
                    if not last_month_closing_price
                    else (net_value - last_month_closing_price)
                    / last_month_closing_price
                )
                await Funds.update.values(
                    net_value=_rec.get("net_value") or 0,
                    quote_change_weekly=quote_change_weekly,
                    quote_change_monthly=quote_change_monthly,
                ).where(Funds.code == _rec.get("code")).gino.status()
            page += 1
        return result


class FundShareDaily(db.Model):
    __tablename__ = "fund_share_daily"

    id = db.Column(Integer, primary_key=True)
    code = db.Column(String)
    name = db.Column(String)
    date = db.Column(Date)
    total_tna = db.Column(BigInteger)

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

            for _rec in data:
                await Funds.update.values(total_tna=_rec.get("total_tna") or 0).where(
                    Funds.code == _rec.get("code")
                ).gino.status()
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
    deadline = db.Column(Date, nullable=False)

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

    @classmethod
    async def get(
        cls,
        codes: Union[str, List[str]],
        symbol: Union[str, List[str]] = None,
        deadline: datetime.date = None,
        stock_symbols: List[str] = None,
    ):
        if codes and isinstance(codes, str):
            codes = [codes]
        if symbol and isinstance(symbol, str):
            symbol = [symbol]
        if not deadline:
            deadline = (await db.func.max(FundPortfolioStock.deadline).gino.all())[0][0]
        if not stock_symbols:
            q = (
                db.select(
                    [FundPortfolioStock.symbol, func.count(FundPortfolioStock.symbol)]
                )
                .where(FundPortfolioStock.deadline == deadline)
                .group_by(FundPortfolioStock.symbol)
                .having(func.count(FundPortfolioStock.symbol) == 1)
            )
            stocks = await q.gino.all()
            stock_symbols = [stock[0] for stock in stocks]
        fields = [
            "id",
            "code",
            "period_start",
            "period_end",
            "pub_date",
            "report_type_id",
            "report_type",
            "rank",
            "symbol",
            "name",
            "shares",
            "market_cap",
            "proportion",
            "deadline",
        ]
        q = (
            cls.select(*fields)
            .where(cls.code.in_(codes))
            .where(cls.deadline == deadline)
        )
        if symbol:
            q = q.where(cls.symbol.in_(symbol))
        records = await q.gino.all()
        items = []
        for record in records:
            item = dict()
            for column in fields:
                item[column] = record[column]
            item["period_start"] = record["period_start"] and record[
                "period_start"
            ].strftime("%Y-%m-%d")
            item["period_end"] = record["period_end"] and record["period_end"].strftime(
                "%Y-%m-%d"
            )
            item["pub_date"] = record["pub_date"] and record["pub_date"].strftime(
                "%Y-%m-%d"
            )
            item["deadline"] = record["deadline"] and record["deadline"].strftime(
                "%Y-%m-%d"
            )
            if item["symbol"] in stock_symbols:
                item["is_single_stock"] = True
            else:
                item["is_single_stock"] = False
            items.append(item)
        return items
