import unittest

import numpy as np

from omicron.talib import *

rand = np.random.RandomState(1337)

# test data from https://github.com/quantopian/empyrical/blob/40f61b4f229df10898d46d08f7b1bdc543c0f99c/empyrical/tests/test_stats.py#L68
# Simple benchmark, no drawdown
simple_benchmark = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]) / 100

# All positive returns, small variance
positive_returns = np.array([1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) / 100

# All negative returns
negative_returns = np.array([0.0, -6.0, -7.0, -1.0, -9.0, -2.0, -6.0, -8.0, -5.0]) / 100

# All negative returns
all_negative_returns = (
    np.array([-2.0, -6.0, -7.0, -1.0, -9.0, -2.0, -6.0, -8.0, -5.0]) / 100
)

# Positive and negative returns with max drawdown
mixed_returns = np.array([np.nan, 1.0, 10.0, -4.0, 2.0, 3.0, 2.0, 1.0, -10.0]) / 100

# Flat line
flat_line_1 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) / 100

# Weekly returns
weekly_returns = np.array([0.0, 1.0, 10.0, -4.0, 2.0, 3.0, 2.0, 1.0, -10.0]) / 100

# Monthly returns
monthly_returns = np.array([0.0, 1.0, 10.0, -4.0, 2.0, 3.0, 2.0, 1.0, -10.0]) / 100

# Series of length 1
one_return = np.array([1.0]) / 100

# Empty series
empty_returns = np.array([]) / 100

# Random noise
noise = rand.normal(0, 0.001, 1000)

noise_uniform = rand.uniform(-0.01, 0.01, 1000)

# Random noise inv
inv_noise = noise * -1

# Flat line
flat_line_0 = np.linspace(0, 0, num=1000)

# Flat line
flat_line_1_tz = np.linspace(0.01, 0.01, num=1000)

# Positive line
pos_line = np.linspace(0, 1, num=1000)

# Negative line
neg_line = np.linspace(0, -1, num=1000)

# Sparse noise, same as noise but with np.nan sprinkled in
replace_nan = rand.choice(np.arange(len(noise)), rand.randint(1, 10))
sparse_noise = noise.copy()
sparse_noise[replace_nan] = np.nan

# Sparse flat line at 0.01
replace_nan = rand.choice(np.arange(len(noise)), rand.randint(1, 10))
sparse_flat_line_1_tz = flat_line_1_tz.copy()
sparse_flat_line_1_tz[replace_nan] = np.nan

one = [
    -0.00171614,
    0.01322056,
    0.03063862,
    -0.01422057,
    -0.00489779,
    0.01268925,
    -0.03357711,
    0.01797036,
]
two = [
    0.01846232,
    0.00793951,
    -0.01448395,
    0.00422537,
    -0.00339611,
    0.03756813,
    0.0151531,
    0.03549769,
]


class MetricsTest(unittest.TestCase):
    def test_cumulative_return(self):
        returns = [0.01, 0.005, 0.002, 0.02]
        actual = cumulative_return(returns)
        self.assertAlmostEqual(actual, 0.0374, places=4)

    def test_annualized_return(self):
        returns = mixed_returns

        actual = annual_return(returns, annual_factor=252 / len(returns))
        self.assertAlmostEqual(actual, 1.913592, places=3)

        actual = annual_return(returns, annual_factor=252 / len(returns))
        self.assertAlmostEqual(actual, 1.913592, places=3)

        actual = annual_return(returns, annual_factor=252 / 5)
        self.assertAlmostEqual(actual, 5.854, places=3)

        returns = 0.03893109170048037
        actual = annual_return(returns, annual_factor=252 / 9)
        self.assertAlmostEqual(actual, 1.913592, places=3)

    def test_sharpe_ratio(self):
        # test data from https://github.com/quantopian/empyrical/blob/40f61b4f229df10898d46d08f7b1bdc543c0f99c/empyrical/tests/test_stats.py#L68

        actual = sharpe_ratio(mixed_returns, np.nan)
        self.assertTrue(np.isnan(actual))

        actual = sharpe_ratio(mixed_returns)
        self.assertAlmostEqual(actual, 1.724, places=3)

        actual = sharpe_ratio(positive_returns, rf=0.0)
        self.assertAlmostEqual(actual, 52.915026221291804)

        actual = sharpe_ratio(mixed_returns, rf=0.03)
        self.assertAlmostEqual(actual, 1.6910259410055302)

        actual = sharpe_ratio(negative_returns, rf=0.0)
        self.assertAlmostEqual(actual, -24.406808633910085)

        actual = sharpe_ratio(flat_line_1, rf=0.0)
        self.assertAlmostEqual(actual, np.inf)

    def test_sortino_ratio(self):
        actual = sortino_ratio(mixed_returns, np.nan)
        self.assertTrue(np.isnan(actual))

        # data from https://www.educba.com/sortino-ratio/,如果使用标准差算法
        # returns = np.array([0.81, 0.53, -0.50, 1.09, -0.13, 2.70, 3.65, 1.00, -0.59, 2.00, -5.65, -1.65]) / 100
        # actual = sortino_ratio(returns, annual_factor=12, rf=0.47/100)
        # self.assertAlmostEqual(actual, -1.238, places=3)

        actual = sortino_ratio(mixed_returns)
        self.assertAlmostEqual(actual, 2.605531251673693)

        actual = sortino_ratio(mixed_returns, rf=0.03)
        self.assertAlmostEqual(actual, 2.552234598077831)

        actual = sortino_ratio(positive_returns)
        self.assertTrue(np.isinf(actual))

        actual = sortino_ratio(negative_returns)
        self.assertAlmostEqual(actual, -13.532743075043401)

        actual = sortino_ratio(simple_benchmark)
        self.assertTrue(np.isinf(actual))

    def test_downside_risk(self):
        # data and result from ttps://www.investopedia.com/terms/d/downside-deviation.asp
        returns = np.array([-0.02, 0.16, 0.31, 0.17, -0.11, 0.21, 0.26, -0.03, 0.38])
        rf = 0.01

        actual = downside_risk(returns - rf, 1)
        self.assertAlmostEqual(actual, 0.0433, places=4)

        # data from empyrical
        annual_factor = 252
        actual = downside_risk(mixed_returns - 0.1, annual_factor)
        self.assertAlmostEqual(actual, 1.7161730681956295, places=4)

    def test_max_drawdown(self):
        with self.assertRaises(ValueError):
            mdd, *_ = max_drawdown(empty_returns)

        mdd, *_ = max_drawdown(one_return)
        self.assertEqual(mdd, 0.0)

        mdd, *_ = max_drawdown(mixed_returns)
        self.assertAlmostEqual(mdd, -0.1, places=4)

        mdd, *_ = max_drawdown(simple_benchmark)
        self.assertAlmostEqual(mdd, 0.0, places=4)

        mdd, *_ = max_drawdown(positive_returns)
        self.assertAlmostEqual(mdd, 0.0, places=4)

    def test_calmar(self):
        actual = calmar_ratio(mixed_returns, annual_factor=252 / len(mixed_returns))
        self.assertAlmostEqual(actual, 19.135925373194233)

        actual = calmar_ratio(positive_returns)
        self.assertTrue(np.isinf(actual))

    def test_volatility(self):
        actual = volatility(flat_line_1_tz, 252 / len(flat_line_1_tz))
        self.assertAlmostEqual(actual, 0.0)

        actual = volatility(mixed_returns, 252)
        self.assertAlmostEqual(actual, 0.9136465399704637, places=4)

    def test_alpha_beta(self):
        alpha, beta = alpha_beta(noise, noise)
        self.assertAlmostEqual(alpha, 0.0)
        self.assertAlmostEqual(beta, 1.0)

        alpha, beta = alpha_beta(noise, noise * -1)
        self.assertAlmostEqual(alpha, -0.0)
        self.assertAlmostEqual(beta, -1.0)

        alpha, beta = alpha_beta(mixed_returns[1:], negative_returns[1:], rf=0.0)
        self.assertAlmostEqual(alpha, -0.03296, places=4)
        self.assertAlmostEqual(beta, -0.712962, places=4)

    def test_omega_ratio(self):
        actual = omega_ratio(mixed_returns, 0.0, 10.0)
        self.assertAlmostEqual(actual, 0.83354263497557934, places=4)

        with self.assertRaises(ValueError):
            actual = omega_ratio(empty_returns, 0.0, 0)

        actual = omega_ratio(mixed_returns, 0.0, -1)
        self.assertTrue(np.isnan(actual))

        actual = omega_ratio(mixed_returns, 0.0, 0, annual_factor=1)
        self.assertAlmostEqual(actual, 1.357142857142857, places=4)

        actual = omega_ratio(positive_returns, 0.01, 0.0)
        self.assertTrue(np.isnan(actual))

        actual = omega_ratio(positive_returns, 0.011, 0.0)
        self.assertAlmostEqual(actual, 1.125, places=4)

        actual = omega_ratio(negative_returns, 0.01, 0.0)
        self.assertAlmostEqual(actual, 0)

    def test_mean_absolute_error(self):
        y = np.array([i for i in range(5)])
        y_hat = y.copy()
        y_hat[4] = 0

        self.assertEqual(0, mean_absolute_error(y, y))
        self.assertAlmostEquals(0.8, mean_absolute_error(y, y_hat))
        self.assertAlmostEquals(0.8, mean_absolute_error(y_hat, y))

    def test_relative_error(self):
        y = np.arange(5)
        y_hat = y.copy()
        y_hat[4] = 0

        print(relative_error(y, y_hat))
