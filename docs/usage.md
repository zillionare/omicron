# 1. 快速上手

## 1.1. 配置文件

Omicron 使用 [cfg4py](https://pypi.org/project/cfg4py/) 来管理配置。

cfg4py 使用 yaml 文件来保存配置项。在使用 cfg4py 之前，您需要在某处初始化 cfg4py:

```python
import cfg4py
cfg4py.init('path_to_your_config_folder')
```

注意初始化 cfg4py 时，需要提供包含配置文件的文件夹的路径，而不是配置文件的路径。配置文件名必须为 defaults.yml。

您至少应该为 Omicron 配置时区、Redis 连接串、Postgres 连接串和 Omega 服务器连接地址：

```yaml
# path_to_config/defaults.yaml
tz: Asia/Shanghai
redis:
    dsn: redis://localhost:6379
postgres:
    # 请修改服务器名称，并在环境变量中增加pg_account, pg_password
    dsn: postgres://${pg_account}:${pg_password}@localhost/zillionare
    enabled: false
omega:
    urls:
        quotes_server: http://localhost:3181
```

请根据您实际环境配置来更改上述文件。缺省地，Omicron 是不使用 Postgres 数据库的。如果您打算使用 Postgres 数据库的话，除了要配置正确的连接串之外，还要将这里的 `enabled` 改为 `true`。

关于 Postgres 数据库的作用，请参见[Omega 文档](https://zillionare-omega.readthedocs.io)

Omicron 的最基础的作用，就是访问行情数据。我们通过下面的例子来看如何实现这个功能：

## 1.2. 获取行情数据

```python
import arrow
import cfg4py
import omicron
from omicron.models.stock import Stock
from omicron.models.timeframe import TimeFrame as tf
from coretypes import FrameType

asynd def main():
    cfg4py.init('path_to_config')

    # 初始化omicron, 建立与 redis, postgres及omega server的连接
    await omicron.init()

    s = Stock()

    # 加载全市场证券列表
    await s.load()

    print(s[0]['code]) # this should output '000001.XSHE'

    # 加载行情数据
    start = arrow.get('2020-10-10').date()
    end = arrow.get('2020-10-20').date()
    sec = Stock('000001.XSHE')

    assert sec.display_name == '平安银行'
    frame_type = FrameType.DAY
    bars = await Stock.get_bars('000001.XSHE', 5, frame_type, stop)
    print(bars)

    # 日期转换
    n = tf.count_day_frames(start, end)
    start = tf.day_shift(end, -n)
    bars = await Stock.get_bars_in_range(['000001.XSHE', start, stop, frame_type)
    print(bars)
```
