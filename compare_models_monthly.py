#!/usr/bin/env python3
"""
Overlay monthly RMSE and bias of several models on one figure each.

Reads enriched result files (the `..._monthly_era5.json` produced by
monthly_analysis.py, which carry metadata['monthly_statistics']).

Usage:
    python compare_models_monthly.py A_monthly_era5.json B_monthly_era5.json [...]
Outputs:
    png/monthly_compare_rmse.png, png/monthly_compare_bias.png
"""
import json
import sys
from pathlib import Path

MONTHS3 = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def load_stats(path):
    with open(path) as f:
        d = json.load(f)
    meta = d.get("metadata", {})
    stats = meta.get("monthly_statistics")
    if stats is None:
        raise SystemExit(f"{path}: no metadata['monthly_statistics'] "
                         f"(run monthly_analysis.py on it first)")
    return meta.get("model_used", Path(path).stem), stats


def main():
    files = sys.argv[1:]
    if len(files) < 1:
        print("Usage: python compare_models_monthly.py A_era5.json B_era5.json [...]")
        sys.exit(1)

    series = [load_stats(f) for f in files]
    months = range(1, 13)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path("png").mkdir(exist_ok=True)

    # RMSE
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for model, stats in series:
        rmse = [s["rmse"] for s in stats["per_month"]]
        ax.plot(months, rmse, "o-", label=f"{model}  (annual {stats['annual']['rmse']:.2f})")
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
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(months)
    ax.set_xticklabels(MONTHS3)
    ax.set_ylabel("Bias, LLM − ERA5 (°C)")
    ax.set_title("Monthly bias vs ERA5")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("png/monthly_compare_bias.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Saved png/monthly_compare_rmse.png, png/monthly_compare_bias.png")
    for model, stats in series:
        print(f"  {model}: annual RMSE {stats['annual']['rmse']:.2f}, "
              f"bias {stats['annual']['bias']:+.2f}, corr {stats['annual']['corr']:.3f}")


if __name__ == "__main__":
    main()
