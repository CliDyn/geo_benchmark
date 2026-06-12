"""Tests for monthly_analysis core stats. Run: python tests/test_monthly_analysis.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monthly_analysis import aggregate_repeats, compute_monthly_statistics


def test_aggregate_repeats_mean_std_count():
    # two repeats, per-month mean is midpoint, population std is half the gap
    r1 = [float(m) for m in range(12)]            # 0..11
    r2 = [float(m) + 2 for m in range(12)]        # 2..13
    mean, std, count = aggregate_repeats([r1, r2])
    assert count == 2
    assert np.allclose(mean, [m + 1 for m in range(12)])
    assert np.allclose(std, [1.0] * 12)           # np.std ddof=0


def test_aggregate_repeats_skips_none_and_empty():
    r1 = [float(m) for m in range(12)]
    mean, std, count = aggregate_repeats([None, r1, None])
    assert count == 1
    assert np.allclose(mean, r1)
    assert np.allclose(std, [0.0] * 12)


def test_aggregate_repeats_no_valid_returns_nan():
    mean, std, count = aggregate_repeats([None, None])
    assert count == 0
    assert np.isnan(mean).all() and np.isnan(std).all()


def _two_point_arrays():
    # month 0: diffs +2 and -2; all other months diff 0
    llm = np.zeros((2, 12))
    era5 = np.zeros((2, 12))
    llm[0, 0] = 2.0      # diff +2
    llm[1, 0] = -2.0     # diff -2
    return llm, era5


def test_monthly_stats_per_month_rmse_bias_mae():
    llm, era5 = _two_point_arrays()
    stats = compute_monthly_statistics(llm, era5)
    jan = stats["per_month"][0]
    assert jan["month"] == 1
    assert jan["n"] == 2
    assert np.isclose(jan["rmse"], 2.0)
    assert np.isclose(jan["mae"], 2.0)
    assert np.isclose(jan["bias"], 0.0)
    feb = stats["per_month"][1]
    assert np.isclose(feb["rmse"], 0.0)


def test_monthly_stats_annual_pool_all_point_month_pairs():
    llm, era5 = _two_point_arrays()
    stats = compute_monthly_statistics(llm, era5)
    ann = stats["annual"]
    assert ann["n"] == 24                          # 2 points x 12 months
    # only two non-zero diffs of magnitude 2 among 24 pairs
    assert np.isclose(ann["rmse"], np.sqrt(8.0 / 24.0))
    assert np.isclose(ann["mae"], 4.0 / 24.0)
    assert np.isclose(ann["bias"], 0.0)


def test_monthly_stats_ignores_nan_pairs():
    llm = np.full((2, 12), np.nan)
    era5 = np.zeros((2, 12))
    llm[0, 0] = 3.0                                # only one valid pair
    stats = compute_monthly_statistics(llm, era5)
    assert stats["per_month"][0]["n"] == 1
    assert np.isclose(stats["per_month"][0]["rmse"], 3.0)
    assert stats["annual"]["n"] == 1


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
