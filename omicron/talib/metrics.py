from typing import Iterable, List, Tuple, Union

import numpy as np
from bottleneck import nanmean
from scipy import stats as scipy_stats

APPROX_BDAYS_PER_YEAR = 252


def mean_absolute_error(y: np.array, y_hat: np.array) -> float:
    """返回预测序列相对于真值序列的平均绝对值差

    两个序列应该具有相同的长度。如果存在nan，则nan的值不计入平均值。

    Examples:

        >>> y = np.arange(5)
        >>> y_hat = np.arange(5)
        >>> y_hat[4] = 0
        >>> mean_absolute_error(y, y)
        0.0

        >>> mean_absolute_error(y, y_hat)
        0.8

    Args:
        y (np.array): 真值序列
        y_hat: 比较序列

    Returns:
        float: 平均绝对值差
    """
    return nanmean(np.abs(y - y_hat))


def relative_error(y: np.array, y_hat: np.array) -> float:
    """相对于序列算术均值的误差值

    Examples:
        >>> y = np.arange(5)
        >>> y_hat = np.arange(5)
        >>> y_hat[4] = 0
        >>> relative_error(y, y_hat)
        0.4

    Args:
        y (np.array): [description]
        y_hat (np.array): [description]

    Returns:
        float: [description]
    """
    mae = mean_absolute_error(y, y_hat)
    return mae / nanmean(np.abs(y))


def max_drawdown(returns: np.ndarray) -> Tuple:
    """计算最大资产回撤

    [代码引用](https://stackoverflow.com/a/22607546)
    Args:
        returns : 收益率（而不是资产净值）

    Returns:
        [Tuple]: mdd, start, send
    """
    if len(returns) < 1:
        raise ValueError("returns must have at least one values")

    equitity = np.nancumprod(returns + 1)
    i = np.nanargmax(np.fmax.accumulate(equitity) - equitity)
    if i == 0:
        return (0, 0, 0)

    j = np.nanargmax(equitity[:i])

    return (equitity[i] - equitity[j]) / equitity[j], i, j


def sharpe_ratio(
    returns: np.ndarray, rf: float = 0.0, annual_factor: int = APPROX_BDAYS_PER_YEAR
) -> float:
    """计算夏普比率(年化)

    平均超额收益（即收益减去无风险利率）除以标准差，即夏普比率。

    `rf`(risk-free利率)为年化无风险利率。

    关于年化因子，请参见[年化收益率][omicron.talib.metrics.annual_return]中的定义。

    Note:
        See [this article](https://towardsdatascience.com/sharpe-ratio-sorino-ratio-and-calmar-ratio-252b0cddc328) for more details.

    Args:
        returns: 回报率(一维数组).
        rf: risk free returns(年化). Defaults to 0.0.
        annual_factor: 年化因子，默认为`APPROX_BDAYS_PER_YEAR`。如果`returns`为日收益率，则`annual_factor`可使用默认值；否则，应该使用根据returns取得的周期，传入对应的年化因子。

    Raise:
        ValueError: 如果`returns`中的收益率少于1个有效值。
    Returns:
        夏普比率。
    """
    adj_returns = returns - rf / APPROX_BDAYS_PER_YEAR
    return (np.nanmean(adj_returns) * np.sqrt(annual_factor)) / np.nanstd(
        adj_returns, ddof=1
    )


def sortino_ratio(
    returns: np.ndarray, rf: float = 0.0, annual_factor: int = APPROX_BDAYS_PER_YEAR
) -> float:
    """计算Sortino比率

    Sortina比率是将收益与负收益的标准差进行比较。在这里，我们并非使用负收益的标准差，而是使用了一种称为[downside risk][omicron.talib.metrics.downside_risk]的方法，这种方法与[investopedia](https://www.investopedia.com/terms/s/sortinoratio.asp)、[Quantopian empyrical](https://github.com/quantopian/empyrical/blob/master/empyrical/stats.py)及[this article](https://towardsdatascience.com/sharpe-ratio-sorino-ratio-and-calmar-ratio-252b0cddc328)保持一致。

    关于年化因子，请参见[年化收益率][omicron.talib.metrics.annual_return]中的定义。

    Args:
        returns : 收益率
        rf ([]): 无风险利率，默认为0.0
        annual_factor: 年化因子，默认为252.

    Returns:
        [float]: Sortino比率
    """
    adj_returns = returns - rf / annual_factor

    annualized_dr = downside_risk(adj_returns, annual_factor)
    if annualized_dr == 0:
        return np.inf

    return np.nanmean(adj_returns) * annual_factor / annualized_dr


def downside_risk(adjust_returns: np.array, annual_factor: int = 252) -> float:
    """计算downside risk。downside risk在sortino ratio中使用。

    严格地说，sortino ratio中的downside risk应该是求其标准差，但[investopedia](https://www.investopedia.com/terms/d/downside-deviation.asp)中的算法如此，很多实现，比如[Quantopian](https://github.com/quantopian/empyrical/blob/master/empyrical/stats.py#downside_risk)，都是这样计算的，这里与他们保持一致。正因为这个原因，我们把这个函数定义为downside_risk，而不是downside_deviation。

    Examples:
        >>> returns = np.array([-0.02, 0.16, 0.31, 0.17, -0.11, 0.21, 0.26, -0.03, 0.38])
        >>> rf = 0.01
        >>> round(downside_risk(returns - rf, 1), 4)
        0.0433

    Args:
        adjust_returns: 已减去无风险利率的收益率。
        annual_factor: 年化因子

    Returns:
        downside risk
    """
    downside = np.clip(adjust_returns, -np.inf, 0)
    downside = np.nanmean(np.square(downside))
    return np.sqrt(downside * annual_factor)


def calmar_ratio(returns: np.ndarray, annual_factor: int = 1) -> float:
    """计算Calmar比率

    Calmar比率是绝对收益与最大回撤的比率，再进行年化的结果。

    关于年化因子，请参见[年化收益率][omicron.talib.metrics.annual_return]中的定义。

    Examples:
        >>> returns = np.array([np.nan, 1., 10., -4., 2., 3., 2., 1., -10.]) / 100
        >>> # 绝对收益是在len(returns)期间取得的，所以年化因子为 252/len(returns)
        >>> round(calmar_ratio(returns, 252/len(returns)), 4)
        19.1359

    Args:
        returns: 非累进收益率
        annual_factor: 年化因子，默认为1，即不进行年化

    Returns:
        calmar收益率
    """
    mdd, *_ = max_drawdown(returns)

    if mdd < 0:
        return annual_return(returns, annual_factor) / abs(mdd)
    else:
        return np.inf


def volatility(
    returns: np.ndarray, annual_factor: int = 1, alpha: float = 2.0
) -> float:
    """计算收益的年化波动率

    代码来自[reference](https://gist.github.com/Ousret/644330403f5677a248c9df3d1f1ca052)

    如果收益率包含`np.nan`，这种情况是允许的。

    请注意，此处的年化因子与**[年化收益率][omicron.talib.metrics.annual_return]中定义的不同**。如果不需要年化（或者数据周期已经是按年为单位），则可以使用默认的年化因子。否则，年化因子 = 年交易日 / 周期长度（以日为单位）。比如，如果周期为日，则年化因子为252，如果为周，则为52。

    Examples:
        >>> returns = np.array([np.nan, 1., 10., -4., 2., 3., 2., 1., -10.]) / 100
        >>> round(volatility(returns, 252), 4)
        0.9136

    Args:
        returns: 非累进收益率，一维数组
        annual_factor: 年化因子

    Returns:
        年化波动率
    """
    return np.nanstd(returns, ddof=1) * annual_factor ** (1.0 / alpha)


def alpha_beta(
    returns: np.ndarray, market: np.ndarray, rf: float = 0.0, annual_factor: float = 1.0
) -> float:
    """计算alpha和beta收益

    Args:
        returns: 日收益率，非累进
        market: 作为参照物的收益率。
        rf: risk-free利率
        annual_factor: 年化因子

    Returns:
        alpha
    """
    if len(returns) != len(market):  # pragma : no cover
        raise ValueError("returns and factor_returns must have the same length")

    adj_returns = returns - rf
    adj_market = market - rf

    beta, alpha = scipy_stats.linregress(adj_market, adj_returns)[:2]

    return alpha * annual_factor, beta * annual_factor


def cumulative_return(returns: Iterable) -> float:
    """将以数组形式存在的单周期收益率进行累计。

    比如，某资产日收益率曲线为[0.01, 0.005, 0.002, 0.02]，则其累进收益率为0.0374

    Examples:
        >>> returns = [0.01, 0.005, 0.002, 0.02]
        >>> round(cumulative_return(returns), 4)
        0.0374

    Args:
        returns (np.array): [description]

    Returns:
        float: [description]
    """
    return np.nanprod(np.array(returns) + 1) - 1


def annual_return(returns: Union[float, np.array], annual_factor: int = 1) -> float:
    """从日收益率(数组)计算年化收益率

    `returns`如果为数组，表示每次交易所取得的收益（非累进收益），允许出现`nan`。如果为浮点数，则表示期间所取得的累进收益。

    为了计算年化收益，需要传入年化因子。比如，如果`returns`是在持仓周期为9天的情况下取得的，则年化因子为252/9（这里认为一年为252个交易日，需要根据资产的具体情况进行调整）。如果是在持仓周期为1年的情况下取得的，则年化因子为1.

    本函数有的文献也称为cagr，即Compute compound annual growth rate。

    Examples:
        >>> # 通过日收益率数组计算年化。由于资产持有时间等于数组长度，所以我们可以不传入
        >>> # `n_hold_days`参数。
        >>> returns = np.array([np.nan, 1., 10., -4., 2., 3., 2., 1., -10.]) / 100
        >>> round(annual_return(returns, annual_factor=252/len(returns)), 3)
        1.914

        >>> # 已知9天的累进收益，计算年化收益率
        >>> returns = 0.03893109170048037
        >>> round(annual_return(returns, annual_factor=252/9), 3)
        1.914

    Args:
        returns (float): 期间(日）收益率
        annual_factor (int): 年化因子。默认为1。
    """

    if isinstance(returns, np.ndarray):
        returns = cumulative_return(returns)

    return (1 + returns) ** annual_factor - 1


def omega_ratio(
    returns: np.ndarray,
    rf: float = 0.0,
    required_return: float = 0.0,
    annual_factor: float = APPROX_BDAYS_PER_YEAR,
) -> float:
    """计算omega比率

    copied from [quantopian/empyrical](https://github.com/quantopian/empyrical/blob/40f61b4f22/empyrical/stats.py)

    Examples:
        >>> returns = np.array([np.nan, 1.0, 10.0, -4.0, 2.0, 3.0, 2.0, 1.0, -10.0]) / 100
        >>> round(omega_ratio(returns, 0.0, 0, annual_factor=1), 2)
        1.36

    Args:
        returns: 日收益率，非累进
        rf: risk-free利率
        required_return: Minimum acceptance return of the investor. Threshold over which to consider positive vs negative returns. It will be converted to a value appropriate for the period of the returns. E.g. An annual minimum acceptable return of 100 will translate to a minimum acceptable return of 0.018.

    Returns:
        omega比率
    """
    if len(returns) < 2:
        raise ValueError("returns must have at least 2 elements")

    if annual_factor == 1:
        return_threshold = required_return
    elif required_return <= -1:
        return np.nan
    else:
        return_threshold = (1 + required_return) ** (1.0 / annual_factor) - 1

    returns_less_thresh = returns - rf - return_threshold

    number = sum(returns_less_thresh[returns_less_thresh > 0.0])
    denom = -1.0 * sum(returns_less_thresh[returns_less_thresh < 0.0])

    if denom > 0.0:
        return number / denom
    else:
        return np.nan
