create table funds
(
    id              serial not null
        constraint funds_pk primary key,
    code            varchar(6) not null,
    name            text not null,
    trustee         text not null,
    operate_mode_id integer       not null,
    operate_mode    varchar(20) not null,
    start_date      date,
    end_date        date,
    advisor         text
);

alter table funds
    owner to zillionare;

create unique index if not exists funds_code_uindex
    on funds (code);


-- auto-generated definition
create table fund_net_value
(
    id                 serial
        constraint fund_net_value_pk primary key,
    code               varchar(6) not null,
    net_value          numeric not null,
    sum_value          numeric not null,
    factor             numeric not null,
    acc_factor         numeric not null,
    refactor_net_value numeric not null,
    day                date             not null,
    shares             bigint
);

alter table fund_net_value
    owner to zillionare;

create unique index fund_net_value_code_day_uindex
    on fund_net_value (code, day);




-- auto-generated definition
create table fund_share_daily
(
    id            serial
        constraint fund_share_daily_pk primary key,
    code          varchar(6) not null,
    name          text not null,
    exchange_code varchar(6) not null,
    date          date          not null,
    shares        double precision,
    report_type   varchar(20)
);

alter table fund_share_daily
    owner to zillionare;

create unique index fund_share_daily_code_date_report_type_uindex
    on fund_share_daily (code, date, report_type);


-- auto-generated definition
create table fund_portfolio_stock
(
    id             serial
        primary key,
    code           varchar(1000)    not null,
    period_start   date             not null,
    period_end     date             not null,
    pub_date       date             not null,
    report_type_id integer          not null,
    report_type    varchar(1000)    not null,
    rank           integer          not null,
    symbol         varchar(1000)    not null,
    name           varchar(1000)    not null,
    shares         double precision not null,
    market_cap     double precision not null,
    proportion     double precision not null
);

alter table fund_portfolio_stock
    owner to zillionare;

create unique index fund_portfolio_stock_code_pub_date_symbol_report_type_uindex
    on fund_portfolio_stock (code, pub_date, symbol, report_type);

