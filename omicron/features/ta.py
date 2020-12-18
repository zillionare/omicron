import numpy as np

from omicron.features.maths import moving_average, polyfit


def predict_moving_average(ts, win: int, n: int = 1):
    # todo: move to other module
    """
    预测次n日的均线位置。如果均线能拟合到某种曲线，则按曲线进行预测，否则，在假定股价不变的前提下来预测次n日均线位置。
    """
    # 1. 曲线拟合法
    curve_len = 5
    ma = moving_average(ts, win)[-curve_len:]
    deg, coef, error = polyfit(ma)
    prediction = []

    if np.sqrt(error / curve_len) / np.mean(ma) < 0.01:
        p = np.poly1d(coef)
        for i in range(curve_len, curve_len + n):
            prediction.append(p(i))

        return prediction

    # 2. 如果曲线拟合不成功，则使用假定股价不变法
    _ts = np.append(ts, [ts[-1]] * n)
    ma = ma(_ts, win)
    return ma[-n:]


def parallel_show(ts, figsize=None):
    # todo: move to other modules
    """形态比较"""
    figsize = figsize or (20, 20 // len(ts))
    # fig, axes = plt.subplots(nrows=1, ncols=len(ts), figsize=figsize)
    # fig.tight_layout()  # Or equivalently,  "plt.tight_layout()"

    # for i, _ts in enumerate(ts):
    #     axes[i].plot(_ts)
