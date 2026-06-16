"""Tests for old-vs-new July comparison helpers. Run: python tests/test_compare_old_new.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compare_old_new_july import point_temp_mean


def _result(values):
    resps = []
    for v in values:
        resps.append(None if v is None else {"parsed_data": {"july_temp_mean": v}})
    return {"llm_responses": resps}


def test_point_temp_mean_averages_repeats():
    assert np.isclose(point_temp_mean(_result([10.0, 14.0])), 12.0)


def test_point_temp_mean_skips_none():
    assert np.isclose(point_temp_mean(_result([None, 8.0, None])), 8.0)


def test_point_temp_mean_no_valid_returns_nan():
    assert np.isnan(point_temp_mean(_result([None, None])))


def test_point_temp_mean_detects_any_month_key():
    r = {"llm_responses": [{"parsed_data": {"october_temp_mean": 5.0}}]}
    assert np.isclose(point_temp_mean(r), 5.0)


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
