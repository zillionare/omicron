import arrow
import numpy as np


def mock_data_for_influx(n=100):
    mock_data = []
    start = arrow.get("2019-01-01 09:30:00")
    names = ["平安银行", "国联证券", "上海银行", "中国银行", "中国平安"]
    for i in range(n):
        mock_data.append(
            (start.shift(minutes=i).naive, 0.1, 0.2, f"00000{i%5+1}.XSHE", names[i % 5])
        )

    mock_data = np.array(
        mock_data,
        dtype=[
            ("frame", "O"),
            ("open", "float32"),
            ("close", "float32"),
            ("code", "O"),
            ("name", "O"),
        ],
    )

    return mock_data
