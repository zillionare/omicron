import unittest

import numpy as np

from omicron.features.momentum import ma_acc, ma_slope, vol_acc, vol_change


class TestMomentum(unittest.TestCase):
    def test_ma_slope(self):
        ts = np.arange(259)

        slopes = [0.003952, 0.003992, 0.004073, 0.004434, 0.005115, 0.007662]
        for i, win in enumerate([5, 10, 20, 60, 120, 250]):
            features = ma_slope(ts, win)
            self.assertAlmostEqual(0, features[f"mom_ma_slope_{win}_err"], places=10)
            self.assertAlmostEqual(slopes[i], features[f"mom_ma_slope_{win}"], places=4)

    def test_ma_acc(self):
        a = 0.2
        b = -5
        c = 100
        ts = np.array([(a * np.square(x) + b * x + c) for x in range(26)])

        exp_a = [0.00273, 0.00281, 0.00259]
        exp_x = [-0.567, -0.278, 0.404]
        for i, win in enumerate([5, 10, 20]):
            features = ma_acc(ts, win)
            print(features)
            self.assertAlmostEqual(0, features[f"mom_ma_acc_{win}_err"], places=10)
            self.assertAlmostEqual(exp_a[i], features[f"mom_ma_acc_{win}_a"], places=5)
            self.assertAlmostEqual(exp_x[i], features[f"mom_ma_acc_{win}_x"], places=3)

    def test_vol_change(self):
        vol = np.array([int(abs(np.sin(3 * x)) * 5) for x in range(40)])

        features = vol_change(vol)
        exp = {
            "mom_vol_change_5": 0.0449,
            "mom_vol_change_10": 0.1259,
            "mom_vol_change_20": 0.0464,
        }
        for cut in [5, 10, 20]:
            key = f"mom_vol_change_{cut}"
            self.assertAlmostEqual(exp[key], features[key], places=3)

    def test_vol_acc(self):
        vol = np.array([int(abs(np.sin(3 * x)) * 5) for x in range(40)])
        features = vol_acc(vol)
        exp = {"mom_vol_acc_5": -0.4218, "mom_vol_acc_10": 0.8023}

        for cut in [5, 10]:
            key = f"mom_vol_acc_{cut}"
            self.assertAlmostEqual(exp[key], features[key], places=3)
