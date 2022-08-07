# History

## 0.1.0 (2020-04-28)


* First release on PyPI.

!!! tip
    this is a tip

## 0.3.0 (2020-11-22)

* Calendar, Triggers and time frame calculation
* Security list
* Bars with turnover
* Valuation

## 0.3.1 (2020-12-11)

this version introduced no features, just a internal amendment release, we're migrating to poetry build system.

## 2.0.0-alpha.34 (2022-07-13)

* change to sync call for Security.select()
* date parameter of Security.select(): if date >= today, it will use the data in cache, otherwise, query from database.

## 2.0.0-alpha.35 (2022-07-13)

* fix issue in security exit date comparison, Security.eval().
