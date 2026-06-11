"""Tests for monthly (all-12-months) query mode and mesh subsampling.

Run directly (no pytest needed):
    python tests/test_monthly_mode.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from climate_llm_benchmark import validate_and_parse_response, format_progress
from subsample_mesh import subsample_land_points

TWELVE = [-3.25, -1.87, 4.12, 9.65, 15.32, 19.78, 22.41, 21.93, 16.55, 10.02, 3.41, -1.96]
TWELVE_STR = ", ".join(str(v) for v in TWELVE)


def test_monthly_parses_12_comma_separated_floats():
    result = validate_and_parse_response(TWELVE_STR, simple_mode=True, monthly=True)
    assert result is not None, "expected parsed dict, got None"
    assert result == {"monthly_temp_mean": TWELVE}, f"unexpected result: {result}"


def test_monthly_tolerates_newlines_and_spaces():
    text = "\n".join(str(v) for v in TWELVE)
    result = validate_and_parse_response(text, simple_mode=True, monthly=True)
    assert result == {"monthly_temp_mean": TWELVE}, f"unexpected result: {result}"


def test_monthly_filters_out_of_range_junk_like_years():
    text = "Mean monthly temperatures 1991-2020: " + TWELVE_STR
    result = validate_and_parse_response(text, simple_mode=True, monthly=True)
    assert result == {"monthly_temp_mean": TWELVE}, f"unexpected result: {result}"


def test_monthly_strips_think_blocks():
    text = "<think>July is about 22.4 here, winter -3</think>" + TWELVE_STR
    result = validate_and_parse_response(text, simple_mode=True, monthly=True)
    assert result == {"monthly_temp_mean": TWELVE}, f"unexpected result: {result}"


def test_monthly_rejects_11_values():
    text = ", ".join(str(v) for v in TWELVE[:11])
    assert validate_and_parse_response(text, simple_mode=True, monthly=True) is None


def test_monthly_rejects_13_in_range_values():
    text = TWELVE_STR + ", 5.5"
    assert validate_and_parse_response(text, simple_mode=True, monthly=True) is None


def test_monthly_rejects_out_of_range_member():
    vals = TWELVE[:11] + [75.0]  # 75 C is outside plausible -100..60
    text = ", ".join(str(v) for v in vals)
    assert validate_and_parse_response(text, simple_mode=True, monthly=True) is None


def test_single_month_july_unchanged():
    result = validate_and_parse_response("25.4", simple_mode=True, month="July")
    assert result == {"july_temp_mean": 25.4}, f"unexpected result: {result}"


def test_single_month_february_key():
    result = validate_and_parse_response("-11.3", simple_mode=True, month="February")
    assert result == {"february_temp_mean": -11.3}, f"unexpected result: {result}"


def test_progress_fresh_start_shows_point_1_and_percent():
    line = format_progress(start_index=0, done=100, total=1540, successful_queries=950)
    assert "100/1540" in line
    assert "6.5%" in line
    assert "point 1" in line          # this run started at point 1
    assert "950" in line


def test_progress_resume_shows_real_start_point():
    line = format_progress(start_index=500, done=600, total=1540, successful_queries=5800)
    assert "600/1540" in line
    assert "point 501" in line        # resumed at land point 501
    assert "5800" in line


def test_progress_handles_zero_total_without_error():
    line = format_progress(start_index=0, done=0, total=0, successful_queries=0)
    assert "0/0" in line              # no ZeroDivisionError


def _toy_mesh(n_points=35):
    # land at every 3rd point -> 12 land points (indices 0,3,...,33)
    points = []
    for i in range(n_points):
        points.append({"lon": float(i), "lat": float(-i), "is_land": i % 3 == 0})
    return {
        "mesh_points": points,
        "resolution": 1.0,
        "mesh_info": {"resolution_degrees": 1.0},
    }


def test_subsample_picks_every_kth_land_point():
    mesh = _toy_mesh()
    land = [p for p in mesh["mesh_points"] if p["is_land"]]
    sub = subsample_land_points(mesh, k=10)
    pts = sub["mesh_points"]
    assert all(p["is_land"] for p in pts), "ocean points must be excluded"
    assert len(pts) == 2, f"expected 2 points (12 land, k=10), got {len(pts)}"
    assert pts[0] == land[0] and pts[1] == land[10], "must pick land points 0 and 10"
    assert sub["resolution"] == 1.0
    assert sub["mesh_info"]["subsample_k"] == 10
    assert sub["mesh_info"]["n_land_points_parent"] == 12


def test_subsample_is_deterministic():
    mesh = _toy_mesh()
    a = subsample_land_points(mesh, k=10)
    b = subsample_land_points(mesh, k=10)
    assert a == b


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
