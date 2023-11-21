## 1. 配置、初始化和关闭 OMICRON

Omicron 依赖于 [zillionare-omega](https://github.com/zillionare/omega) 服务来获取数据。但它并不直接与 Omega 服务通讯，相反，它直接读取 Omega 服务器会写入数据的`Influxdb`和`redis`数据库。因此，在使用 Omicron 之前，我们需要提供这两个服务器的连接地址，并进行初始化。

### 1.1. 配置和初始化
Omicron 使用 [cfg4py](https://pypi.org/project/cfg4py/) 来管理配置。

cfg4py 使用 yaml 文件来保存配置项。在使用 cfg4py 之前，您需要在某处初始化 cfg4py，然后再初始化 omicron:

???+ tip

    为了简洁起见，我们在顶层代码中直接使用了 async/await。通常，这些代码能够直接在 notebook 中运行，但如果需要在普通的 python 脚本中运行这些代码，您通常需要将其封装到一个异步函数中，再通过`asyncio.run`来运行它。

    ```python
    import asyncio
    import cfg4py
    import omicron
    async def main():
        cfg4py.init('path/to/your/config/dir')
        await omicron.init()
        # DO YOUR GREAT JOB WITH OMICRON

    asyncio.run(main())
    ```

```python
import cfg4py
import omicron
cfg4py.init('path/to/your/config/dir')

await omicron.init()
```

注意初始化 cfg4py 时，需要提供包含配置文件的**文件夹**的路径，而**不是配置文件**的路径。配置文件名必须为 defaults.yml。

您至少应该为 omicron 配置 Redis 连接串和 influxdb 连接串。下面是常用配置示例：

```yaml
# DEFAULTS.YAML
redis:
  dsn: redis://${REDIS_HOST}:${REDIS_PORT}

influxdb:
  url: http://${INFLUXDB_HOST}:${INFLUXDB_PORT}
  token: ${INFLUXDB_TOKEN}
  org: ${INFLUXDB_ORG}
  bucket_name: ${INFLUXDB_BUCKET_NAME}
  enable_compress: true
  max_query_size: 150000

notify:
    mail_from: ${MAIL_FROM}
    mail_to:
        - ${MAIL_TO}
    mail_server: ${MAIL_SERVER}
    dingtalk_access_token: ${DINGTALK_ACCESS_TOKEN}
    dingtalk_secret: ${DINGTALK_SECRET}
```

请根据您实际环境配置来更改上述文件。上述配置中，${{REDIS_HOST}}意味着环境变量。如果是 windows，您需要在系统 > 环境变量中进行设置。如果是 Linux 或者 Mac，您需要修改.bashrc，例如：
```
export REDIS_HOST=localhost
```

### 1.2. 关闭 omicron
在您的进程即将退出之前，请记得关闭 omicron。如果您是在 notebook 中使用 omicron, 则可以忽略此步聚。

```python
    await omicron.close()
```

## 2. 数据读取
### 2.1. 证券列表

[Security](/api/security) 和 [Query](/api/security/#omicron.models.security.Query) 提供了证券列表和查询操作。查询被设计成为链式 API。通常，我们通过调用 `Security.select()`来生成一个`Query`对象，然后可以针对此对象，进行各种过查询过滤，最后，我们调用`query.eval()`方法结束链式调用，并返回结果。

#### 2.1.1. 查询所有证券代码
您可以通过以下方法来获取某一天的证券列表：

```python
# 4. ASSUME YOU HAVE OMICRON INIT
dt = datetime.date(2022, 5, 20)

query = Security.select(dt)
codes = await query.eval()
print(codes)
# THE OUTPUTS IS LIKE ["000001.XSHE", "000004.XSHE", ...]
```
这里的`dt`如果没有提供的话，将使用最新的证券列表。但在回测中，您通常不同时间的证券列表，因此，`dt`在这种情况下是必须的，否则，您将引入未来数据。

#### 2.1.2. 返回所有股票或者指数
```python
query = Security.select(dt)
codes = await query.types(["stock"]).eval()
print(codes)
```

#### 2.1.3. 排除某种股票（证券）
```python
query = Security.select(dt)
codes = await query.exclude_st().exclude_kcb().exclude_cyb().eval()
print(codes)
```

#### 2.1.4. 如果只要求某种股票（证券）
```python
query = Security.select(dt)
codes = await query.only_kcb().only_st().only_cyb().eval()
print(codes)
#得到空列表
```

#### 2.1.5. 按别名进行模糊查询

A 股的证券在标识上，一般有代码（code 或者 symbol)、拼音简写 (name) 和汉字表示名 (display_name) 三种标识。比如中国平安，其代码为 601318.XSHG; 其拼音简写为 ZGPA；而中国平安被称为它的别名 (`alias`)。

如果要查询所有中字头的股票：

```python
query = Security.select(dt)
codes = await query.alias_like("中").eval()
print(codes)
```

#### 2.1.6. 通过代码查询其它信息

通过前面的查询我们可以得到一个证券列表，如果要得到具体的信息，可以通过`info`接口来查询：

```python
    dt = datetime.date(2022, 5, 20)
    info = await Security.info("688001.XSHG", dt)
    print(info)
```
输出为：
```json
{
    'type': 'stock',
    'display_name': '华兴源创',
    'alias': '华兴源创',
    'end': datetime.date(2200, 1, 1),
    'start': datetime.date(2019, 7, 22),
    'name': 'HXYC'
}
```
### 2.2. 交易日历及时间帧计算
Omicron 不仅提供了交易日历，与其它量化框架相比，我们还提供了丰富的与时间相关的运算操作。这些操作都有详细的文档和示例，您可以通过 [TimeFrame](/api/timeframe) 来进一步阅读。

omicron 中，常常会遇到时间帧 (Time Frame) 这个概念。因为行情数据都是按一定的时间长度组织的，比如 5 分钟，1 天，等等。因此，在 omicron 中，我们经常使用某个时间片结束的时间，来标识这个时间片，并将其称之为帧 (Time Frame)。

omicron 中，我们支持的时间帧包括日内的分钟帧 (FrameType.MIN1), 5 分钟帧 (FrameType.MIN5), 15 分钟帧、30 分钟帧和 60 分钟帧，以及日线级别的 FrameType.DAY, FrameType.WEEK 等。关于详细的类型说明，请参见 [coretypes](https://zillionare.github.io/core-types/)

omicron 提供的交易日历起始于 2005 年 1 月 4 日。提供的行情数据，最早从这一天起。

大致上，omicron 提供了以下时间帧操作：
#### 2.2.1. 交易时间的偏移
如果今天是 2022 年 5 月 20 日，您想得到 100 天前的交易日，则可以使用 day_shift:
```python
from omicron import tf
dt = datetime.date(2022, 5, 20)

tf.day_shift(dt, -100)
```
输出是 datetime.date(2021, 12, 16)。在这里，day_shift 的第二个参数`n`是偏移量，当它小于零时，是找`dt`前`n`个交易日；当它大于零时，是找`dt`之后的`n`个交易日。

比如有意思的是`n` == 0 的时候。对上述`dt`，day_shift(dt, 0) 得到的仍然是同一天，但如果`dt`是 2022 年 5 月 21 日是周六，则 day_shift(datetime.date(2022, 5, 21)) 将返回 2022 年 5 月 20 日。因为 5 月 21 日这一天是周六，不是交易日，day_shift 将返回其对应的交易日，这在多数情况下会非常方便。

除了`day_shift`外，timeframe 还提供了类似函数比如`week_shift`等。一般地，您可以用 shift(dt, n, frame_type) 来对任意支持的时间进行偏移。

#### 2.2.2. 边界操作 ceiling 和 floor
很多时候我们需要知道具体的某个时间点 (moment) 所属的帧。如果要取其上一帧，则可以用 floor 操作，反之，使用 ceiling。
```python
tf.ceiling(datetime.date(2005, 1, 4), FrameType.WEEK)
# OUTPUT IS DATETIME.DATE(2005, 1, 7)
```

#### 2.2.3. 时间转换
为了加快速度，以及方便持久化存储，在 timeframe 内部，有时候使用整数来表示时间。比如 20220502 表示的是 2022 年 5 月 20 日，而 202205220931 则表示 2022 年 5 月 20 日 9 时 31 分钟。

这种表示法，有时候要求我们进行一些转换：
```python
# 将整数表示的日期转换为日期
tf.int2date(20220522) # datetime.date(2022, 5, 22)
# 将整数表示的时间转换为时间
tf.int2time(202205220931) # datetime.datetime(2022, 5, 22, 9, 31)

# 将日期转换成为整数
tf.date2int(datetime.date(2022, 5, 22)) # 20220520

# 将时间转换成为时间
tf.date2time(datetime.datetime(2022, 5, 22, 9, 21)) # 202205220921
```

#### 2.2.4. 列出区间内的所有时间帧
有时候我们需要得到`start`和`end`之间某个时间帧类型的所有时间帧：
```python
start = arrow.get('2020-1-13 10:00').naive
end = arrow.get('2020-1-13 13:30').naive
tf.get_frames(start, end, FrameType.MIN30)
[202001131000, 202001131030, 202001131100, 202001131130, 202001131330]
```

???+ Important
    上面的示例中，出现了可能您不太熟悉的`naive`属性。它指的是取不带时区的时间。在 python 中，时间可以带时区（timezone-aware) 和不带时区 (naive)。

    如果您使用 datetime.datetime(2022, 5, 20)，它就是不带时区的，除非您专门指定时区。

    在 omicron 中，我们在绝大多数情况下，仅使用 naive 表示的时间，即不带时区，并且假定时区为东八区（即北京时间）。

如果您只知道结束时间，需要向前取`n`个时间帧，则可以使用`get_frames_by_count`。

如果您只是需要知道在`start`和`end`之间，总共有多少个帧，请使用 `count_frames`:
```python
start = datetime.date(2019, 12, 21)
end = datetime.date(2019, 12, 21)
tf.count_frames(start, end, FrameType.DAY)
```
输出将是 1。上述方法还有一个快捷方法，即`count_day_frames`，并且，对 week, month, quaters 也是一样。

### 2.3. 读取行情数据
现在，让我们来获取一段行情数据：
```python
code = "000001.XSHE"

end = datetime.date(2022, 5, 20)
bars = await Stock.get_bars(code, 10, FrameType.DAY, end)
```
返回的`bars`将是一个 numpy structured array, 其类型为 [bars_dtype](https://zillionare.github.io/core-types/)。一般地，它包括了以下字段：

    * frame（帧）
    * open（开盘价）
    * high（最高价）
    * low（最低价）
    * close（收盘价）
    * volume（成交量，股数）
    * amount（成交额）
    * factor（复权因子）

缺省情况下，返回的数据是到`end`为止的前复权数据。你可以通参数`fq = False`关闭它，来获得不复权数据，并以此自行计算后复权数据。

如果要获取某个时间段的数据，可以使用[get_bars_in_range](/api/stock/#omicron.models.stock.Stock.get_bars_in_range)。

上述方法总是尽最大可能返回实时数据，如果`end`为当前时间的话，但由于`omega`同步延时是一分钟，所以行情数据最多可能慢一分钟。如果要获取更实时的数据，可以通过[get_latest_price](/api/stock/#omicron.models.stock.Stock.get_latest_price)方法。

要获涨跌停价格和标志，请使用:

* [get_trade_price_limits](/api/stock/#omicron.models.stock.Stock.get_trade_price_limits)
* [trade_price_limits_flags](/api/stock/#omicron.models.stock.Stock.trade_price_limit_flags)
* [trade_price_limit_flags_ex](/api/stock/#omicron.models.stock.Stock.trade_price_limit_flags_ex)

### 2.4. 板块数据

提供同花顺板块行业板块和概念板块数据。在使用本模块之前，需要进行初始化：

```python
# 请先进行omicron初始化，略
from omicron.models.board import Board, BoardType
Board.init('192.168.100.101')
```

此处的IP为安装omega服务器的ip。

通过[board_list](/api/board/#omicron.models.board.Board.board_list)来查询所有的板块。

其它方法请参看[API文档](/api/board/#omicron.models.board.Board)


## 3. 策略编写

omicron 通过 [strategy](/api/strategy) 来提供策略框架。通过该框架编写的策略，可以在实盘和回测之间无缝转换 -- 根据初始化时传入的服务器不同而自动切换。

omicron 提供了一个简单的 [双均线策略](/latest/api/strategy/#omicron.strategy.sma) 作为策略编写的示范，可结合其源码，以及本文档中的[完整策略示例](/latest/usage/#完整sma回测示例)在notebook中运行查看。


策略框架提供了回测驱动逻辑及一些基本函数。要编写自己的策略，您需要从基类[BaseStrategy](/api/strategy/#omicron.strategy.base.BaseStrategy)派生出自己的子类，并改写它的`predict`方法来实现调仓换股。

策略框架依赖于[zillionare-trader-client](https://zillionare.github.io/trader-client/)，在回测时，需要有[zillionare-backtesting](https://zillionare.github.io/backtesting/)提供回测服务。在实盘时，需要[zilllionare-gm-adaptor](https://github.com/zillionare/trader-gm-adaptor)或者其它实盘交易网关提供服务。

策略代码可以不加修改，即可使用于回测和实盘两种场景。


### 3.1. 回测场景

实现策略回测，一般需要进行以下步骤：
1. 从此基类派生出一个策略子类，比如sma.py
2. 子类需要重载`predict`方法，根据当前传入的时间帧和帧类型参数，获取数据并进行处理，评估出交易信号。
3. 子类根据交易信号，在`predict`方法里，调用基类的`buy`和`sell`方法来进行交易
4. 生成策略实例，通过实例调用`backtest`方法来进行回测，该方法将根据策略构建时指定的回测起始时间、终止时间、帧类型，逐帧生成各个时间帧，并调用子类的`predict`方法。如果调用时指定了`prefetch_stocks`参数，`backtest`还将进行数据预取（预取的数据长度由`warmup_peroid`决定），并将截止到当前回测帧时的数据传入。
5. 在交易结束时，调用`plot_metrics`方法来获取如下所示的回测指标图
![](https://images.jieyu.ai/images/2023/05/20230508160012.png?2)

如何派生子类，可以参考[sma][omicron.strategy.sma.SMAStrategy]源代码。


```python
from omicron.strategy.sma import SMAStrategy
sma = SMAStrategy(
    url="", # the url of either backtest server, or trade server
    is_backtest=True,
    start=datetime.date(2023, 2, 3),
    end=datetime.date(2023, 4, 28),
    frame_type=FrameType.DAY,
    warmup_period = 20
)

await sma.backtest(prefetch_stocks=["600000.XSHG"])
```
在回测时，必须要指定`is_backtest=True`和`start`, `end`参数。
### 3.2. 回测报告

在回测结束后，可以通过以下方法，在notebook中绘制回测报告：

```python
await sma.plot_metrics()
```

这将绘制出类似以下图：

![](https://images.jieyu.ai/images/2023/05/20230508160012.png?2)

#### 3.2.1. 在回测报告中添加技术指标

!!! info
    Since 2.0.0.a76

首先，我们可以在策略类的predict方法中计算出技术指标，并保存到成员变量中。在下面的示例代码中，我们将技术指标及当时的时间保存到了一个indicators数组中（注意顺序！），然后在回测结束后，在调用 plot_metrics时，将其传入即可。

```
indicators = [
    (datetime.date(2021, 2, 3), 20.1),
    (datetime.date(2021, 2, 4), 20.2),
    ...,
    (datetime.date(2021, 4, 1), 20.3)
    ]
await sma.plot_metrics(indicator)
```
时间只能使用主周期的时间，否则可能产生无法与坐标轴对齐的情况。

加入的指标默认只显示在legend中，如果要显示在主图上，需要点击legend进行显示。

指标除可以叠加在主图上之外，还会出现在基准线的hoverinfo中（即使指标的计算与基准线无关），参见上图中的“指标”行。


### 3.3. 使用数据预取

!!! info
    since version 2.0.0-alpha76

在回测中，可以使用主周期的数据预取，以加快回测速度。工作原理如下：

如果策略指定了`warmup_period`，并在调用`backtest`时传入了`prefetch_stocks`参数，则`backtest`将会在回测之前，预取从[start - warmup_period * frame_type, end]间的portfolio行情数据，并在每次调用`predict`方法时，通过`barss`参数，将[start - warmup_period * frame_type, start + i * frame_type]间的数据传给`predict`方法。传入的数据已进行前复权。

如果在回测过程中，需要偷看未来数据，可以使用peek方法。

### 3.4. 完整SMA回测示例

以下策略需要在notebook中运行，并且需要事先安装omega服务器同步数据，并正确配置omicron。

该示例在《大富翁量化课程》课件环境下可运行。

```python
import cfg4py
import omicron
import datetime
from omicron.strategy.sma import SMAStrategy
from coretypes import FrameType

cfg = cfg4py.init("/etc/zillionare")
await omicron.init()

sec = "600000.XSHG"
start = datetime.date(2022, 1, 4)
end = datetime.date(2023, 1,1)

sma = SMAStrategy(sec, url=cfg.backtest.url, is_backtest=True, start=start, end=end, frame_type=FrameType.DAY, warmup_period=10)
await sma.backtest(portfolio=[sec], stop_on_error=False)
await sma.plot_metrics(sma.indicators)

```

### 3.5. 实盘
在实盘环境下，你还需要在子类中加入周期性任务(比如每分钟执行一次），在该任务中调用`predict`方法来完成交易，如以下示例所示：

```python
import cfg4py
import omicron
import datetime
from omicron.strategy.sma import SMAStrategy
from coretypes import FrameType
from apscheduler.schedulers.asyncio import AsyncIOScheduler


cfg = cfg4py.init("/etc/zillionare")
await omicron.init()

async def daily_job():
    sma = SMAStrategy(sec, url=cfg.traderserver.url, is_backtest=False,frame_type=FrameType.DAY)
    bars = await Stock.get_bars(sma._sec, 20, FrameType.DAY)
    await sma.predict(barss={sma._sec: bars})

async def main():
    scheduler = AsyncIOScheduler()
    scheduler.add_job(daily_job, 'cron', hour=14, minute=55)
    scheduler.start()
```
策略代码无须修改。

该策略将自动在每天的14：55运行，以判断是否要进行调仓换股。您需要额外判断当天是否为交易日。

## 4. 绘图

omicron 通过 [Candlestick](/api/plotting/candlestick/) 提供了 k 线绘制功能。默认地，它将绘制一幅显示 120 个 bar，可拖动（以加载更多 bar)，并且可以叠加副图、主图叠加各种指标的 k 线图：

![](https://images.jieyu.ai/images/2023/05/20230508164848.png)

上图显示了自动检测出来的平台。此外，还可以进行顶底自动检测和标注。

!!! note
    通过指定`width`参数，可以影响初始加载的bar的数量。

omicron 通过 [metris](/api/plotting/metrics) 提供回测报告。该报告类似于：

![](https://images.jieyu.ai/images/2023/05/20230508160012.png)

它同样提供可拖动的绘图，并且在买卖点上可以通过鼠标悬停，显示买卖点信息。

omicron 的绘图功能只能在 notebook 中使用。
## 5. 评估指标
omicron 提供了 mean_absolute_error 函数和 pct_error 函数。它们在 scipy 或者其它库中也能找到，为了方便不熟悉这些第三方库的使用者，我们内置了这个常指标。

对一些常见的策略评估函数，我们引用了 empyrical 中的相关函数，比如 alpha, beta, shapre_ratio， calmar_ratio 等。

## 6. TALIB 库
您应该把这里提供的函数当成实验性的。这些 API 也可能在某天被废弃、重命名、修改，或者这些 API 并没有多大作用，或者它们的实现存在错误。

但是，如果我们将来会抛弃这些 API 的话，我们一定会通过 depracted 方法提前进行警告。

## 7. 扩展
Python当中的四舍五入用于证券投资，会带来严重的问题，比如，像`round(0.3/2)`，我们期望得到`0.2`，但实际上会得到`0.1`。这种误差一旦发生成在一些低价股身上，将会带来非常大的不确定性。比如，1.945保留两位小数，本来应该是1.95，如果被误舍入为1.94，则误差接近0.5%，这对投资来说是难以接受的。

!!! info
    如果一天只进行一次交易，一次交易误差为0.5%，一年累积下来，误差将达到2.5倍。

我们在[decimals](/api/omicron/#omicron.extensions.decimals)中提供了适用于证券交易领域的版本，`math_round`和价格比较函数`price_equal`。

我们还在[np](/api/omicron/#omicron.extensions.np)中，对numpy中缺失的一些功能进行了补充，比如`numpy_append_fields`, `fill_nan`等。


