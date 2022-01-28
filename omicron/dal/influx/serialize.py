import io
from typing import List

import pandas as pd


def unserialize(
    data: bytes,
    rename_time_field: str = None,
    keep_cols: List[str] = None,
    sort_by: str = None,
) -> pd.DataFrame:
    """反序列化influxdb返回的数据为dataframe

    如果指明了rename_time_field，且keep_cols中的列包含相关列，则keep_cols应该使用rename后的列。

    Args:
        data : 待反序列化的数据
        rename_time_field : flux查询默认通过_time字段返回时间戳。如果需要将字段重命名，则通过本字段传入. Defaults to None.
        keep_cols : 传入需要保留的列的名字，默认全部保留. Defaults to None.
        sort_by : 如果传入了列名字，则将按该列进行升序排列. Defaults to None.

    Returns:
        [description]
    """
    data = data.decode("utf-8")

    df = pd.read_csv(
        io.StringIO(data),
        parse_dates=["_time"],
        sep=",",
        header=0,
        engine="c",
        infer_datetime_format=True,
    )

    if rename_time_field:
        df.rename(columns={"_time": rename_time_field}, inplace=True)

    if keep_cols:
        df = df[keep_cols]

    if sort_by:
        df.sort_values(by=sort_by, inplace=True)

    return df
