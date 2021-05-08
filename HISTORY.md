# History

## 0.1.0 (2020-04-28)


* First release on PyPI.

## 0.3.0 (2020-11-22)

* Calendar, Triggers and time frame calculation
* Security list
* Bars with turnover
* Valuation

## 0.3.1 (2020-12-11)

this version introduced no features, just a internal amendment release, we're migrating to poetry build system.

## 1.0 (2021-4-25)
### fixed:
* No turnover retured if cache is empty #8
* tf.floor incorrect calculation on non-trade day #7
* postgres FATAL: sorry, too many clients already #6
* TypeError: <class 'datetime.date'> is not supported #6
