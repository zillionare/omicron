# 快速上手

## 配置文件

Omicron使用 [cfg4py](https://pypi.org/project/cfg4py/) 来管理配置。

cfg4py使用yaml文件来保存配置项。在使用cfg4py之前，您需要在某处初始化cfg4py:

```python
import cfg4py
cfg4py.init('path_to_your_config_folder')
```

---
**注意**

cfg4py要求您通过环境变量来声明当前机器的角色是开发、测试还是生产环境。为不同的场景应用不同的配置，被认为是确保安全性的最佳实践之一。

关于如何设置，请参见 [Omega部署指南](https://zillionare-omega.readthedocs.io/zh_CN/latest/deployment.html#id14)

---

Omicron需要从配置文件中读取到以下信息：

```yaml
# path_to_config/defaults.yaml
tz: Asia/Shanghai
redis:
  dsn: redis://localhost:6379
postgres:
  dsn: postgres://${pg_account}:${pg_password}@localhost/zillionare
  enabled: false
omega:
  urls:
    quotes_server: http://localhost:3181
```

您的工程可能使用了其它的方式来管理配置，比如configparser。推荐您将配置统一使用cfg4py来管理。
## 获取行情数据
```python
import arrow
import cfg4py
import omicron
from omicron.models.securities import Securities
from omicron.models.security import Security
from omicron.core.timeframe import tf
from omicron.core.types import FrameType

asynd def main():
    cfg4py.init('path_to_config')

    # 初始化omicron, 建立与 redis, postgres及omega server的连接
    await omicron.init()

    s = Securities()

    # 加载全市场证券列表
    await s.load()

    print(s[0]['code]) # this should output '000001.XSHE'

    # 加载行情数据
    start = arrow.get('2020-10-10').date()
    end = arrow.get('2020-10-20').date()
    sec = Security('000001.XSHE')

    assert sec.display_name == '平安银行'
    frame_type = FrameType.DAY
    bars = await sec.load_bars(start, stop, frame_type)
    print(bars)

    # 日期转换
    n = tf.count_day_frames(start, end)
    start = tf.day_shift(end, -n)
    bars = await sec.load_bars(start, stop, frame_type)
    print(bars)
```



