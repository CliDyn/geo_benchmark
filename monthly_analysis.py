#!/usr/bin/env python3
"""
Monthly analysis for `monthly` benchmark results (parsed_data = {"monthly_temp_mean": [12]}).

Reproduces the paper's comparison method (nearest-neighbour ERA5, RMSE/MAE/bias/corr,
averaging the repeats per point) but per month, plus an annual pool over all
point x month pairs. Writes the full statistics back into the output file's metadata
and produces seasonal-RMSE, scatter and per-month difference-map figures.

Usage:
    python monthly_analysis.py results/..._monthly.json [data/t2m_climatology_1991-2020.nc]
Outputs:
    results/..._monthly_era5.json   (per-point ERA5/LLM/diff + metadata['monthly_statistics'])
    png/monthly_<model>_seasonal_rmse.png, _scatter.png, _diff_maps.png
"""
import json
import sys
from pathlib import Path

import numpy as np

MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]


# ---------------------------------------------------------------- pure core

def aggregate_repeats(monthly_lists):
    """Average the repeats of one point per month. Returns (mean[12], std[12], count)."""
    valid = [m for m in monthly_lists if m is not None and len(m) == 12]
    if not valid:
        nan = np.full(12, np.nan)
        return nan, nan, 0
    a = np.asarray(valid, dtype=float)            # (n_repeats, 12)
    return a.mean(axis=0), a.std(axis=0), a.shape[0]


def _metrics(diff, x, y, month=None):
    n = int(diff.size)
    out = {
        "rmse": float(np.sqrt(np.mean(diff ** 2))) if n else float("nan"),
        "mae": float(np.mean(np.abs(diff))) if n else float("nan"),
        "bias": float(np.mean(diff)) if n else float("nan"),
        "corr": float(np.corrcoef(x, y)[0, 1]) if n > 1 and x.std() > 0 and y.std() > 0 else float("nan"),
        "n": n,
    }
    return {"month": month, **out} if month is not None else out


def compute_monthly_statistics(llm, era5):
    """llm, era5: arrays (n_points, 12). Per-month and annual-pooled metrics."""
    llm = np.asarray(llm, dtype=float)
    era5 = np.asarray(era5, dtype=float)
    per_month = []
    for m in range(12):
        x, y = llm[:, m], era5[:, m]
        mask = np.isfinite(x) & np.isfinite(y)
        per_month.append(_metrics(x[mask] - y[mask], x[mask], y[mask], month=m + 1))
    mask = np.isfinite(llm) & np.isfinite(era5)
    per_month_annual = _metrics((llm - era5)[mask], llm[mask], era5[mask])
    return {"per_month": per_month, "annual": per_month_annual}


# ---------------------------------------------------------------- ERA5 + enrich

def sample_era5_monthly(ds, lat, lon):
    """12 ERA5 t2m_mean values (Jan..Dec) at the nearest grid point."""
    era5_lon = lon if lon >= 0 else lon + 360
    series = ds["t2m_mean"].sel(latitude=lat, longitude=era5_lon, method="nearest")
    return np.asarray(series.sortby("month").values, dtype=float)


def enrich(results_data, ds):
    """Add per-point llm/era5 monthly arrays + difference to point_info. Returns (llm, era5) arrays."""
    results = results_data["results"]
    P = len(results)
    llm_means = np.full((P, 12), np.nan)
    era5_means = np.full((P, 12), np.nan)
    repeat_std = np.full((P, 12), np.nan)
    for i, result in enumerate(results):
        pi = result["point_info"]
        lists = [r["parsed_data"]["monthly_temp_mean"]
                 for r in result.get("llm_responses", [])
                 if r and "parsed_data" in r and "monthly_temp_mean" in r["parsed_data"]]
        mean, std, count = aggregate_repeats(lists)
        era5 = sample_era5_monthly(ds, pi["lat"], pi["lon"])
        pi["llm_monthly_mean"] = mean.tolist()
        pi["llm_monthly_std"] = std.tolist()
        pi["llm_monthly_count"] = count
        pi["era5_monthly_mean"] = [float(v) for v in era5]
        pi["monthly_temp_difference"] = (mean - era5).tolist()
        llm_means[i], era5_means[i], repeat_std[i] = mean, era5, std
        if (i + 1) % 200 == 0:
            print(f"  enriched {i + 1}/{P} points")
    return llm_means, era5_means, repeat_std


def build_statistics(llm_means, era5_means, repeat_std):
    stats = compute_monthly_statistics(llm_means, era5_means)
    spread = np.nanmean(repeat_std, axis=0)
    for m in range(12):
        stats["per_month"][m]["mean_repeat_std"] = float(spread[m])
    stats["annual"]["mean_repeat_std"] = float(np.nanmean(spread))
    return stats


def print_table(stats, model):
    print(f"\nMonthly LLM-vs-ERA5 statistics: {model}")
    print("=" * 78)
    print(f"{'Month':<10}{'RMSE':>8}{'MAE':>8}{'Bias':>9}{'Corr':>8}{'RepStd':>9}{'N':>8}")
    print("-" * 78)
    for m, s in enumerate(stats["per_month"]):
        print(f"{MONTHS[m]:<10}{s['rmse']:>8.2f}{s['mae']:>8.2f}{s['bias']:>+9.2f}"
              f"{s['corr']:>8.3f}{s['mean_repeat_std']:>9.2f}{s['n']:>8d}")
    a = stats["annual"]
    print("-" * 78)
    print(f"{'ANNUAL':<10}{a['rmse']:>8.2f}{a['mae']:>8.2f}{a['bias']:>+9.2f}"
          f"{a['corr']:>8.3f}{a['mean_repeat_std']:>9.2f}{a['n']:>8d}")
    print("(ANNUAL = pooled over all point x month pairs)")


# ---------------------------------------------------------------- plots

def _land_boundary():
    try:
        import geopandas as gpd
        shp = Path("data/land/ne_10m_land.shp")
        if shp.exists():
            return gpd.read_file(shp)
    except Exception:
        pass
    return None


def make_plots(results_data, stats, model, out_prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path("png").mkdir(exist_ok=True)
    results = results_data["results"]
    lats = np.array([r["point_info"]["lat"] for r in results])
    lons = np.array([r["point_info"]["lon"] for r in results])
    diff = np.array([r["point_info"]["monthly_temp_difference"] for r in results])   # (P,12)
    llm = np.array([r["point_info"]["llm_monthly_mean"] for r in results])
    era5 = np.array([r["point_info"]["era5_monthly_mean"] for r in results])
    land = _land_boundary()

    months = range(1, 13)
    labels = [m[:3] for m in MONTHS]
    rmse = [s["rmse"] for s in stats["per_month"]]
    bias = [s["bias"] for s in stats["per_month"]]

    # 1a) RMSE per month
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(months, rmse, "o-", color="tab:red")
    ax.axhline(stats["annual"]["rmse"], color="tab:red", ls=":", alpha=0.6,
               label=f"Annual RMSE={stats['annual']['rmse']:.2f}")
    ax.set_xticks(months)
    ax.set_xticklabels(labels)
    ax.set_ylabel("RMSE (°C)")
    ax.set_ylim(bottom=0)
    ax.set_title(f"Monthly RMSE (LLM vs ERA5): {model}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"png/{out_prefix}_rmse.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 1b) bias per month
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(months, bias, "s-", color="tab:blue")
    ax.axhline(0, color="gray", lw=0.8)
    ax.axhline(stats["annual"]["bias"], color="tab:blue", ls=":", alpha=0.6,
               label=f"Annual bias={stats['annual']['bias']:+.2f}")
    ax.set_xticks(months)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Bias, LLM − ERA5 (°C)")
    ax.set_title(f"Monthly bias: {model}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"png/{out_prefix}_bias.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 2) pooled scatter LLM vs ERA5
    x = era5.flatten()
    y = llm.flatten()
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x, y, s=3, alpha=0.25)
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=2, label="1:1")
    c = np.polyfit(x, y, 1)
    ax.plot([lo, hi], np.poly1d(c)([lo, hi]), "b-", lw=1.5,
            label=f"fit y={c[0]:.2f}x+{c[1]:.2f}")
    a = stats["annual"]
    ax.text(0.05, 0.95, f"N={a['n']}\nRMSE={a['rmse']:.2f}\nMAE={a['mae']:.2f}\n"
            f"Bias={a['bias']:+.2f}\nCorr={a['corr']:.3f}",
            transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("ERA5 (°C)")
    ax.set_ylabel("LLM (°C)")
    ax.set_title(f"All months pooled: {model}")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"png/{out_prefix}_scatter.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 3) 12-panel per-month difference maps
    vmax = np.nanpercentile(np.abs(diff), 98)
    fig, axes = plt.subplots(3, 4, figsize=(20, 10))
    for m in range(12):
        ax = axes.flat[m]
        if land is not None:
            land.boundary.plot(ax=ax, color="0.6", linewidth=0.3)
        sc = ax.scatter(lons, lats, c=diff[:, m], cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax, s=6)
        ax.set_title(f"{MONTHS[m]}  RMSE={stats['per_month'][m]['rmse']:.2f}", fontsize=10)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 85)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(sc, ax=axes, shrink=0.6, label="LLM−ERA5 (°C)")
    fig.suptitle(f"Monthly LLM−ERA5 difference: {model}", fontsize=14)
    fig.savefig(f"png/{out_prefix}_diff_maps.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved png/{out_prefix}_rmse.png, _bias.png, _scatter.png, _diff_maps.png")


# ---------------------------------------------------------------- main

def main():
    if len(sys.argv) < 2:
        print("Usage: python monthly_analysis.py <results_monthly.json> [era5.nc]")
        sys.exit(1)
    results_file = sys.argv[1]
    era5_file = sys.argv[2] if len(sys.argv) > 2 else "data/t2m_climatology_1991-2020.nc"

    import xarray as xr

    with open(results_file) as f:
        results_data = json.load(f)
    model = results_data.get("metadata", {}).get("model_used", "model")
    print(f"Loaded {len(results_data['results'])} points | model {model}")

    ds = xr.open_dataset(era5_file)
    llm_means, era5_means, repeat_std = enrich(results_data, ds)
    stats = build_statistics(llm_means, era5_means, repeat_std)

    results_data["metadata"]["monthly_statistics"] = stats
    results_data["metadata"]["era5_file"] = era5_file
    results_data["metadata"]["era5_period"] = "1991-2020"

    out = Path(results_file).with_suffix("")
    out = out.parent / (out.name + "_era5.json")
    with open(out, "w") as f:
        json.dump(results_data, f, indent=2, default=str)
    print(f"Wrote enriched results + statistics -> {out}")

    print_table(stats, model)

    out_prefix = "monthly_" + model.replace(":", "_").replace("/", "_").replace(".", "_")
    make_plots(results_data, stats, model, out_prefix)


if __name__ == "__main__":
    main()
