
![](http://images.jieyu.ai/images/hot/zillionbanner.jpg)

<h1 align="center">Omicron - Core Library for Zillionare</h1>


[![Version](http://img.shields.io/pypi/v/zillionare-omicron?color=brightgreen)](https://pypi.python.org/pypi/zillionare-omicron)
[![CI Status](https://github.com/zillionare/omicron/actions/workflows/release.yml/badge.svg)](https://github.com/zillionare/omicron)
[![Code Coverage](https://img.shields.io/codecov/c/github/zillionare/omicron)](https://app.codecov.io/gh/zillionare/omicron)
[![Downloads](https://pepy.tech/badge/zillionare-omicron)](https://pepy.tech/project/zillionare-omicron)
[![License](https://img.shields.io/badge/License-MIT.svg)](https://opensource.org/licenses/MIT)
[![Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Contents

## 简介

Omicron是Zillionare的核心模块，提供以下功能：

1. 行情数据读取（需要启动[zillionare-omega](https://github.com/zillionare/omega)服务。
2. [概念板块数据](/api/board)，也需要启动[zillionare-omega](https://github.com/zillionare/omega)服务。
3. [交易日历及时间帧相关操作](/api/timeframe)
4. [证券列表及相关查询操作](/api/security)
5. [numpy数组功能扩展](/api/omicron/#extensions-package)
6. [技术指标及形态分析功能](/api/talib])
   1. 各种均线、曲线拟合、直线斜率和夹解计算、曲线平滑函数等。
   2. 形态分析功能，如交叉、顶底搜索、平台检测、RSI背离等。
7. [策略编写框架](/usage#stragety)，不修改代码即可同时用于实盘与回测。
8. 绘图功能。提供了[交互式k线图](/usage#candlestick)及[回测报告](/usage#metrics)。
9. 其它
   1. 修正Python的round函数错误，改用[math_round](/api/omicron/#extensions.decimal)
   2. 判断价格是否相等的函数：[price_equal](/api/omicron/#extensions.decimal)

Omicron是大富翁量化框架的一部分。您必须至少安装并运行[Omega](https://zillionare.github.io/omega)，然后才能利用omicron来访问上述数据。

[使用文档](https://zillionare.github.io/omicron)

## Credits

Zillionare-Omicron采用[Python Project Wizard](https://zillionare.github.io/python-project-wizard)构建。

