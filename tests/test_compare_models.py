"""Tests for the ERA5 internal-variability reference. Run: python tests/test_compare_models.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compare_models_monthly import anomaly_rmse_per_month


def test_anomaly_rmse_single_month_deviation():
    # 1 point, 2 years, 12 months; only Jan differs (10 vs 14 -> clim 12, anom +-2)
    vals = np.zeros((1, 2, 12))
    vals[0, 0, 0] = 10.0
    vals[0, 1, 0] = 14.0
    per_month, annual = anomaly_rmse_per_month(vals)
    assert np.isclose(per_month[0], 2.0)            # sqrt(mean([(-2)^2,(2)^2]))
    assert np.allclose(per_month[1:], 0.0)
    assert np.isclose(annual, np.sqrt(8.0 / 24.0))  # 8 sq over 1pt*2yr*12mo pairs


def test_anomaly_rmse_pools_over_points_and_years():
    # 2 points, identical interannual spread of +-3 in July -> July rmse = 3
    vals = np.zeros((2, 2, 12))
    vals[:, 0, 6] = -3.0
    vals[:, 1, 6] = 3.0
    per_month, annual = anomaly_rmse_per_month(vals)
    assert np.isclose(per_month[6], 3.0)
    assert np.isclose(annual, np.sqrt((4 * 9.0) / (2 * 2 * 12)))


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
            except Exception as e:
                failures += 1
                print(f"ERROR {name}: {type(e).__name__}: {e}")
    print("=" * 40)
    print("ALL TESTS PASSED" if failures == 0 else f"{failures} FAILURE(S)")
    sys.exit(1 if failures else 0)
