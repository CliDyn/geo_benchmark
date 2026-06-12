#!/usr/bin/env python3
"""
Overlay monthly RMSE and bias of several models on one figure each, with an
ERA5 internal-variability reference (interannual RMS on the LLM grid).

Reads enriched result files (`..._monthly_era5.json` from monthly_analysis.py,
carrying metadata['monthly_statistics']). The ERA5 reference is the RMS of each
year's deviation from the 30-year (1991-2020) per-point mean, sampled at the same
points as the LLM answers, from the raw monthly file.

Usage:
    python compare_models_monthly.py A_era5.json B_era5.json [...] [--kurz data/...kurz.nc]
Outputs:
    png/monthly_compare_rmse.png  (RMSE line per model + dashed ERA5 reference)
    png/monthly_compare_bias.png  (bias line per model + grey +-ERA5 corridor)
"""
import json
import sys
from pathlib import Path

import numpy as np

MONTHS3 = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DEFAULT_KURZ = "data/data_stream-moda_stepType-avgua_kurz.nc"
DEFAULT_CLIM = "data/t2m_climatology_1991-2020.nc"


# ---------------------------------------------------------------- pure core

def anomaly_rmse_per_month(values):
    """values: (n_points, n_years, 12) in degC.
    RMS of (year value - per-point-per-month mean over years), pooled over points & years.
    Returns (per_month[12], annual)."""
    values = np.asarray(values, dtype=float)
    clim = values.mean(axis=1, keepdims=True)              # per-point per-month 30yr mean
    anom = values - clim
    per_month = np.sqrt(np.nanmean(anom ** 2, axis=(0, 1)))
    annual = float(np.sqrt(np.nanmean(anom ** 2)))
    return per_month, annual


# ---------------------------------------------------------------- ERA5 reference

def _point_indexers(lats, lons):
    import xarray as xr
    lons360 = [l if l >= 0 else l + 360 for l in lons]
    return (xr.DataArray(np.asarray(lats), dims="point"),
            xr.DataArray(np.asarray(lons360), dims="point"))


def era5_internal_rmse(kurz_file, lats, lons):
    """Interannual RMS from the raw monthly file, sampled at the LLM points."""
    import xarray as xr
    ds = xr.open_dataset(kurz_file)
    la, lo = _point_indexers(lats, lons)
    t = (ds["t2m"].sel(latitude=la, longitude=lo, method="nearest")
         .sortby("valid_time").transpose("point", "valid_time"))
    arr = np.asarray(t.values, dtype=float) - 273.15        # (point, time) -> degC
    P, T = arr.shape
    n_years = T // 12
    arr = arr[:, : n_years * 12].reshape(P, n_years, 12)    # (point, year, month)
    return anomaly_rmse_per_month(arr)


def era5_std_reference(clim_file, lats, lons):
    """Same reference from the precomputed climatology t2m_std (cross-check)."""
    import xarray as xr
    ds = xr.open_dataset(clim_file)
    la, lo = _point_indexers(lats, lons)
    std = (ds["t2m_std"].sel(latitude=la, longitude=lo, method="nearest")
           .sortby("month").transpose("point", "month"))
    std = np.asarray(std.values, dtype=float)               # (point, month)
    return np.sqrt(np.mean(std ** 2, axis=0))


# ---------------------------------------------------------------- io

def load_stats(path):
    with open(path) as f:
        d = json.load(f)
    meta = d.get("metadata", {})
    stats = meta.get("monthly_statistics")
    if stats is None:
        raise SystemExit(f"{path}: no metadata['monthly_statistics'] "
                         f"(run monthly_analysis.py on it first)")
    return meta.get("model_used", Path(path).stem), stats


def load_points(path):
    with open(path) as f:
        d = json.load(f)
    lats = [r["point_info"]["lat"] for r in d["results"]]
    lons = [r["point_info"]["lon"] for r in d["results"]]
    return np.array(lats), np.array(lons)


# ---------------------------------------------------------------- plots

def make_plots(series, ref_per_month, ref_annual):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path("png").mkdir(exist_ok=True)
    months = range(1, 13)

    # RMSE
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for model, stats in series:
        rmse = [s["rmse"] for s in stats["per_month"]]
        ax.plot(months, rmse, "o-", label=f"{model}  (annual {stats['annual']['rmse']:.2f})")
    if ref_per_month is not None:
        ax.plot(months, ref_per_month, "k--", lw=2,
                label=f"ERA5 interannual variability  (annual {ref_annual:.2f})")
    ax.set_xticks(months)
    ax.set_xticklabels(MONTHS3)
    ax.set_ylabel("RMSE (°C)")
    ax.set_ylim(bottom=0)
    ax.set_title("Monthly RMSE vs ERA5")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("png/monthly_compare_rmse.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Bias
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for model, stats in series:
        bias = [s["bias"] for s in stats["per_month"]]
        ax.plot(months, bias, "s-", label=f"{model}  (annual {stats['annual']['bias']:+.2f})")
    if ref_per_month is not None:
        ax.plot(months, ref_per_month, "k--", lw=2,
                label=f"ERA5 interannual variability  (annual {ref_annual:.2f})")
    ax.set_xticks(months)
    ax.set_xticklabels(MONTHS3)
    ax.set_ylabel("Bias, LLM − ERA5 (°C)")
    ax.set_ylim(bottom=0)
    ax.set_title("Monthly bias vs ERA5")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("png/monthly_compare_bias.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Saved png/monthly_compare_rmse.png, png/monthly_compare_bias.png")


# ---------------------------------------------------------------- main

def main():
    argv = sys.argv[1:]
    kurz_file = DEFAULT_KURZ
    files = []
    i = 0
    while i < len(argv):
        if argv[i] == "--kurz":
            kurz_file = argv[i + 1]
            i += 2
        else:
            files.append(argv[i])
            i += 1

    if not files:
        print("Usage: python compare_models_monthly.py A_era5.json B_era5.json [...] [--kurz file.nc]")
        sys.exit(1)

    series = [load_stats(f) for f in files]
    lats, lons = load_points(files[0])

    ref_per_month, ref_annual = None, None
    if Path(kurz_file).exists():
        print(f"Computing ERA5 interannual reference from {kurz_file} at {len(lats)} points...")
        ref_per_month, ref_annual = era5_internal_rmse(kurz_file, lats, lons)
        if Path(DEFAULT_CLIM).exists():
            chk = era5_std_reference(DEFAULT_CLIM, lats, lons)
            print(f"  cross-check vs climatology t2m_std: max|Δ| = "
                  f"{float(np.max(np.abs(ref_per_month - chk))):.4f} °C (should be ~0)")
        print("  ERA5 ref per month: " + ", ".join(f"{v:.2f}" for v in ref_per_month)
              + f"  | annual {ref_annual:.2f}")
    else:
        print(f"NOTE: {kurz_file} not found -> plotting without ERA5 reference "
              f"(gunzip {kurz_file}.gz first).")

    make_plots(series, ref_per_month, ref_annual)
    for model, stats in series:
        print(f"  {model}: annual RMSE {stats['annual']['rmse']:.2f}, "
              f"bias {stats['annual']['bias']:+.2f}, corr {stats['annual']['corr']:.3f}")


if __name__ == "__main__":
    main()
