# History


## 2.0.0-alpha.46 (2022-09-10)
* [#40](https://github.com/zillionare/omicron/issues/40) 增加k线图绘制功能。
* 本次修订增加了对plotly, ckwrap的依赖。
* 将原属于omicron.talib包中的bars_since, find_runs等跟数组相关的操作，移入omicron.extensions.np中。
## 2.0.0-alpha.45 (2022-09-08)
* [#39](https://github.com/zillionare/omicron/issues/39) fixed.
* removed dependency of postgres
* removed funds
* update arrow's version to be great than 1.2
* lock aiohttp's version to >3.8, <4.0>
## 2.0.0-alpha.35 (2022-07-13)

* fix issue in security exit date comparison, Security.eval().

## 2.0.0-alpha.34 (2022-07-13)

* change to sync call for Security.select()
* date parameter of Security.select(): if date >= today, it will use the data in cache, otherwise, query from database.

## 0.3.1 (2020-12-11)

this version introduced no features, just a internal amendment release, we're migrating to poetry build system.

## 0.3.0 (2020-11-22)

* Calendar, Triggers and time frame calculation
* Security list
* Bars with turnover
* Valuation
## 0.1.0 (2020-04-28)


* First release on PyPI.
