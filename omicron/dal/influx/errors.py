from omicron.core.errors import Error


class InfluxDBWriteError(Error):
    """写influx db错误"""

    pass


class InfluxDBQueryError(Error):
    """查询influx db错误"""

    pass


class InfluxDeleteError(Error):
    """删除纪录错误"""

    pass


class InfluxSchemaError(Error):
    """InfluxDBshema命令(如创建bucket, 查询org)错误"""

    pass
