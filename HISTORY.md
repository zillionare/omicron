# History

## 2.0.0-alpha79
*增加get_buy_limit_secs和get_sell_limit_secs接口，以查询区间内涨停个股，增加此类查询的性能。
## 2.0.0-alpha78
* backtest中捕获异常时，如果是TradeError类型，打印该对象自带的stack
* Candlestick中判断峰谷时使用2倍标准差参数，以实现自适应
* 修复当行情数据缺失时，造成的backtest迭代frame与cursor指向不一致问题
## 2.0.0-alpha77
* strategy增加lifecycle
* 保留最后一个回测周期仅供交易使用，不调用`predict`
* Security获取股票列表时，如果不调用`types`，将获取股票列表，调用`types()`不传参数将获取带指数、股票的列表。
## 2.0.0-alpha76
* 增加backtestlog模块，用于输出回测日志时，将时间替换为回测时间
* 增加行情预取功能
* 增加回测报告中绘制自定义指标功能（仅支持Scatter)
## 2.0.0-alpha.69
* BaseStrategy增加`available_shares`方法
## 2.0.0-alpha.68
* 增加了MetricsGraph
* 增加Strategy基类
* Candlestick增加了布林带指标
## 2.0.0-alpha.49 (2022-09-16)
* 修订了安装文档。
* 移除了windows下对ta-lib的依赖。请参考[安装指南](docs/installation.md)以获取在windows下安装ta-lib的方法。
* 更新了poetry.lock文件。在上一版中，该文件与pyproject.toml不同步，导致安装时进行版本锁定，延长了安装时间。
* 修复了k线图标记顶和底时，标记离被标注的点太远的问题。
## 2.0.0-alpha.46 (2022-09-10)
* [#40](https://github.com/zillionare/omicron/issues/40) 增加k线图绘制功能。
* 本次修订增加了对plotly, ckwrap, ta-lib的依赖。
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
