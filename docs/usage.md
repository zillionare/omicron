# 配置和初始化omicron

Omicron 使用 [cfg4py](https://pypi.org/project/cfg4py/) 来管理配置。

cfg4py 使用 yaml 文件来保存配置项。在使用 cfg4py 之前，您需要在某处初始化 cfg4py，然后再初始化omicron:

???+ Tips
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
cfg4py.init('path_to_your_config_folder')

await omicron.init()
```

注意初始化 cfg4py 时，需要提供包含配置文件的文件夹的路径，而不是配置文件的路径。配置文件名必须为 defaults.yml。

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

omega:
  home: ~/.zillionare/omega
  urls:
    quotes_server: http://localhost:3181

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

这里的`Security.select()`方法返回一个
