from math import copysign


def math_round(x: float, digits: int):
    """由于浮点数的表示问题，很多语言的round函数与数学上的round函数不一致。下面的函数结果与数学上的一致。

    Args:
        x: 要进行四舍五入的数字
        digits: 小数点后保留的位数

    """

    return int(x * (10**digits) + copysign(0.5, x)) / (10**digits)


def price_equal(x: float, y: float) -> bool:
    """判断股价是否相等

    Args:
        x : 价格1
        y : 价格2

    Returns:
        如果相等则返回True，否则返回False
    """
    return abs(math_round(x, 2) - math_round(y, 2)) < 1e-2
