# 配置和初始化omicron

Omicron 使用 [cfg4py](https://pypi.org/project/cfg4py/) 来管理配置。

cfg4py 使用 yaml 文件来保存配置项。在使用 cfg4py 之前，您需要在某处初始化 cfg4py，然后再初始化omicron:

???+ tip

    为了简洁起见，我们在顶层代码中直接使用了async/await。通常，这些代码能够直接在notebook中运行，但如果需要在普通的python脚本中运行这些代码，您通常需要将其封装到一个异步函数中，再通过`asyncio.run`来运行它。

    ```python
    import asyncio
    import cfg4py
    import omicron
    async def main():
        cfg4py.init('path/to/your/config/dir')
        await omicron.init()
        # do your great job with omicron

    asyncio.run(main())
    ```

```python
import cfg4py
import omicron
cfg4py.init('path/to/your/config/dir')

await omicron.init()
```

注意初始化 cfg4py 时，需要提供包含配置文件的**文件夹**的路径，而**不是配置文件**的路径。配置文件名必须为 defaults.yml。

您至少应该为omicron配置Redis 连接串和influxdb连接串。下面是常用配置示例：

```yaml
# defaults.yaml
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

请根据您实际环境配置来更改上述文件。上述配置中，${{REDIS_HOST}}意味着环境变量。如果是windows，您需要在系统 > 环境变量中进行设置。如果是Linux或者Mac，您需要修改.bashrc，例如：
```
export REDIS_HOST=localhost
```

Omicron提供了证券列表、交易日历、行情数据及其它功能。

# 2. 关闭omicron
在您的进程即将退出之前，请记得关闭omicron。如果您是在notebook中使用omicron,则可以忽略此步聚。

```python
    await omicron.close()
```

# 3. 证券列表
您可以通过以下方法来获取某一天的证券列表
```python
# 4. assume you have omicron init
dt = datetime.date(2022, 5, 20)

query = Security.select(dt)
codes = await query.eval()
print(codes)
# the outputs is like ["000001.XSHE", "000004.XSHE", ...]
```
这里的`dt`如果没有提供的话，将使用最新的证券列表。但在回测中，您通常不同时间的证券列表，因此，`dt`在这种情况下是必须的，否则，您将会使用最新的证券列表来回测过去。

这里的`Security.select()`方法返回一个`Query`对象，用以按查询条件进行过滤。该对象支持链式操作。它的方法中，除了`eval`，基本都是用来指定过滤条件，构建查询用的。如果要得到最终结果，请使用`Query.eval`方法。

## 返回所有股票或者指数
```python
query = Security.select(dt)
codes = await query.types(["stock"]).eval()
print(codes)
```

## 排除某种股票（证券）
```python
query = Security.select(dt)
codes = await query.exclude_st().exclude_kcb().exclude_cyb().eval()
print(codes)
```

## 如果只要求某种股票（证券）
```python
query = Security.select(dt)
codes = await query.only_kcb().only_st().only_cyb().eval()
print(codes)
#得到空列表
```

## 按别名进行模糊查询
A股的证券在标识上，一般有代码（code或者symbol)、拼音简写(name)和汉字表示名(display_name)三种标识。比如中国平安，其代码为601318.XSHG;其拼音简写为ZGPA；而中国平安被称为它的别名(`alias`)。

如果要查询所有中字头的股票：
```python
query = Security.select(dt)
codes = await query.alias_like("中").eval()
print(codes)
```

## 通过代码查询其它信息
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
# TimeFrame时间计算
Omicron不仅提供了交易日历，与其它量化框架相比，我们还提供了丰富的时间相关的运算操作。这些操作都有详细的文档和示例，您可以通过[TimeFrame](/api/timeframe)来进一步阅读。

omicron中，常常会遇到时间帧(Time Frame)这个概念。因为行情数据都是按一定的时间长度组织的，比如5分钟，1天，等等。因此，在omicron中，我们经常使用某个时间片结束的时间，来标识这个时间片，并将其称之为帧(Time Frame)。

omicron中，我们支持的时间帧是有限的，主要是日内的分钟帧(FrameType.MIN1), 5分钟帧(FrameType.MIN5), 15分钟帧、30分钟帧和00分钟帧，以及日线级别的FrameType.DAY, FrameType.WEEK等。关于详细的类型说明，请参见[coretypes](https://zillionare.github.io/core-types/)

omicron提供的交易日历起始于2005年1月4日。提供的行情数据，最早从这一天起。

大致上，omicron提供了以下操作：
## 交易时间的偏移
如果今天是2022年5月20日，您想得到100天前的交易日，则可以使用day_shift:
```python
from omicron import tf
dt = datetime.date(2022, 5, 20)

tf.day_shift(dt, -100)
```
输出是datetime.date(2021, 12, 16)。在这里，day_shift的第二个参数`n`是偏移量，当它小于零时，是找`dt`前`n`个交易日;当它大于零时，是找`dt`之后的`n`个交易日。

比如有意思的是`n` == 0的时候。对上述`dt`，day_shift(dt, 0)得到的仍然是同一天，但如果`dt`是2022年5月21日是周六，则day_shift(datetime.date(2022, 5, 21))将返回2022年5月20日。因为5月21日这一天是周六，不是交易日，day_shift将返回其对应的交易日，这在多数情况下会非常方便。

除了`day_shift`外，timeframe还提供了类似函数比如`week_shift`等。一般地，您可以用shift(dt, n, frame_type)来对任意支持的时间进行偏移。

## 边界操作 ceiling和floor
很多时候我们需要知道具体的某个时间点(moment)所属的帧。如果要取其上一帧，则可以用floor操作，反之，使用ceiling。
```python
tf.ceiling(datetime.date(2005, 1, 4), FrameType.WEEK)
# output is datetime.date(2005, 1, 7)
```

## 时间转换
为了加快速度，以及方便持久化存储，在timeframe内部，有时候使用整数来表示时间。比如20220502表示的是2022年5月20日，而202205220931则表示2022年5月20日9时31分钟。

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

## 列时间帧
有时候我们需要得到`start`和`end`之间某个时间帧类型的所有时间帧：
```python
start = arrow.get('2020-1-13 10:00').naive
end = arrow.get('2020-1-13 13:30').naive
tf.get_frames(start, end, FrameType.MIN30)
[202001131000, 202001131030, 202001131100, 202001131130, 202001131330]
```

???+ Important
    上面的示例中，出现了可能您不太熟悉的`naive`属性。它指的是取不带时区的时间。在python中，时间可以带时区（timezone-aware)和不带时区(naive)。

    如果您使用datetime.datetime(2022, 5, 20)，它就是不带时区的，除非您专门指定时区。

    在omicron中，我们在绝大多数情况下，仅使用naive表示的时间，即不带时区，并且假定时区为东八区（即北京时间）。

如果您只知道结束时间，需要向前取`n`个时间帧，则可以使用`get_frames_by_count`。

如果您只是需要知道在`start`和`end`之间，总共有多少个帧，请使用 `count_frames`:
```python
start = datetime.date(2019, 12, 21)
end = datetime.date(2019, 12, 21)
tf.count_frames(start, end, FrameType.DAY)
```
输出将是1。上述方法还有一个快捷方法，即`count_day_frames`，并且，对week, month, quaters也是一样。

# 取行情数据
现在，让我们来获取一段行情数据：
```python
code = "000001.XSHE"

end = datetime.date(2022, 5, 20)
bars = await Stock.get_bars(code, 10, FrameType.DAY, end)
```
返回的`bars`将是一个numpy structured array, 其类型为[bars_dtype](https://zillionare.github.io/core-types/)。一般地，它包括了以下字段：

    * frame（帧）
    * open（开盘价）
    * high（最高价）
    * low（最低价）
    * close（收盘价）
    * volume（成交量，股数）
    * amount（成交额）
    * factor(复权因子)

# 评估指标
omicron提供了mean_absolute_error函数和pct_error函数。它们在scipy或者其它库中也能找到，为了方便不熟悉这些第三方库的使用者，我们内置了这个常指标。

对一些常见的策略评估函数，我们引用了empyrical中的相关函数，比如alpha, beta, shapre_ratio， calmar_ratio等。

# talib库
您应该把这里提供的函数当成实验性的。这些API也可能在某天被废弃、重命名、修改，或者这些API并没有多大作用，或者它们的实现存在错误。

但是，如果我们将来会抛弃这些API的话，我们一定会通过depracted方法提前进行警告。
