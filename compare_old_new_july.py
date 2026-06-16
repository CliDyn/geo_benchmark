#!/usr/bin/env python3
"""
Compare OLD full-grid July runs (paper, ~15395 land points) with the NEW sub10
single-month July runs, on the SAME 1540 sub10 points.

Each old full-grid file is subset to the sub10 coordinates (sub10 is an exact
subset of the 1deg grid), the 10 repeats are averaged, and ERA5-July
nearest-neighbour is subtracted (same postprocessing as the paper). The new side
reuses the already-postprocessed single-month July `_era5.json` files.

Framed as a generation-temperature comparison: T=0 (old _simple), T=0.3 (old _temp03),
default (the new provider-default-temperature single-month July runs).

Outputs:
    png/temperature_july_rmse_bias.png  per-model RMSE & bias: T=0, T=0.3, default
    png/temperature_july_scatter.png    per-point July LLM temperature, T=0.3 vs default

Run: python compare_old_new_july.py
"""
import gzip
import json
from pathlib import Path

import numpy as np

from compare_models_monthly import rmse_bias

ERA5 = "data/t2m_climatology_1991-2020.nc"
SUB10 = "meshes/mesh_data_1.0deg_sub10.json"
R = "results/climate_results_1.0deg_r10_"

MODELS = [
    {"name": "gpt-5", "color": "C0",
     "old": {"T=0": R + "gpt-5_simple.json.gz"},                       # no July temp03 file
     "new": R + "gpt-5_sub10_simple_July_era5.json"},
    {"name": "gpt-oss:120b", "color": "C1",
     "old": {"T=0": R + "gpt-oss_120b_combined_simple.json.gz",
             "T=0.3": R + "gpt-oss_120b_temp03_simple.json.gz"},
     "new": R + "gpt-oss_120b_sub10_simple_July_era5.json"},
    {"name": "mistral-small3.1:24b", "color": "C2",
     "old": {"T=0": R + "mistral-small3_1_24b_simple.json.gz",
             "T=0.3": R + "mistral-small3_1_24b_simple_temp03.json.gz"},
     "new": R + "mistral-small3_1_24b_sub10_simple_July_era5.json"},
    {"name": "gemma3:27b", "color": "C3",
     "old": {"T=0": R + "gemma3_27b_simple.json.gz",
             "T=0.3": R + "gemma3_27b_simple_temp_03.json.gz"},
     "new": R + "gemma3_27b_sub10_simple_July_era5.json"},
]


def _load(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        return json.load(f)


def point_temp_mean(result):
    """Mean of the *_temp_mean values across a point's valid repeats (NaN if none)."""
    vals = []
    for x in result.get("llm_responses", []):
        if x and "parsed_data" in x:
            k = next((k for k in x["parsed_data"] if k.endswith("_temp_mean")), None)
            if k is not None:
                vals.append(x["parsed_data"][k])
    return float(np.mean(vals)) if vals else float("nan")


def sub10_keys():
    m = _load(SUB10)
    return {(round(p["lat"], 4), round(p["lon"], 4)) for p in m["mesh_points"]}


def old_llm_map(path, keys):
    """{(lat,lon): mean July LLM temp} for the sub10 points present in an old full-grid file."""
    d = _load(path)
    out = {}
    for r in d["results"]:
        pi = r["point_info"]
        key = (round(pi["lat"], 4), round(pi["lon"], 4))
        if key in keys:
            out[key] = point_temp_mean(r)
    return out


def new_llm_map(path):
    """{(lat,lon): (llm_temp_mean, temp_difference)} from a new single-month _era5 file."""
    d = _load(path)
    out = {}
    for r in d["results"]:
        pi = r["point_info"]
        if "llm_temp_mean" in pi and "temp_difference" in pi:
            out[(round(pi["lat"], 4), round(pi["lon"], 4))] = (pi["llm_temp_mean"], pi["temp_difference"])
    return out


def era5_july_map(keys):
    import xarray as xr
    ds = xr.open_dataset(ERA5)
    keys = sorted(keys)
    lats = [k[0] for k in keys]
    lon360 = [k[1] if k[1] >= 0 else k[1] + 360 for k in keys]
    la = xr.DataArray(lats, dims="p")
    lo = xr.DataArray(lon360, dims="p")
    v = ds["t2m_mean"].sel(latitude=la, longitude=lo, month=7, method="nearest").values
    return {keys[i]: float(v[i]) for i in range(len(keys))}


def make_plots(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path("png").mkdir(exist_ok=True)
    names = [r["name"] for r in results]
    x = np.arange(len(results))

    # Fig A: RMSE / bias per model — old T=0 (square), old T=0.3 (triangle), new (open circle)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, idx, ylab in [(ax1, 0, "RMSE (°C)"), (ax2, 1, "Bias, LLM − ERA5 (°C)")]:
        for i, r in enumerate(results):
            if "T=0" in r["old"]:
                ax.plot(i - 0.16, r["old"]["T=0"][idx], "s", color=r["color"], markersize=9)
            if "T=0.3" in r["old"]:
                ax.plot(i, r["old"]["T=0.3"][idx], "^", color=r["color"], markersize=9)
            if r["new"]:
                ax.plot(i + 0.16, r["new"][idx], "o", color=r["color"], markersize=12,
                        markerfacecolor="none", markeredgewidth=2.4)
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace(":", "\n") for n in names], fontsize=9)
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3, axis="y")
        if idx == 1:
            ax.axhline(0, color="gray", lw=0.8)
    ax1.set_ylim(bottom=0)
    for marker, lab in [("ks", "T=0"), ("k^", "T=0.3")]:
        ax1.plot([], [], marker, label=lab)
    ax1.plot([], [], "o", markerfacecolor="none", markeredgecolor="k", markeredgewidth=2.4,
             label="default")
    ax1.legend()
    fig.suptitle("July (sub10, 1540 points): RMSE and bias by generation temperature")
    fig.tight_layout()
    fig.savefig("png/temperature_july_rmse_bias.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Fig B: per-point old-LLM vs new-LLM (models that have a new run)
    sc = [r for r in results if r["new_llm"]]
    n = len(sc)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, sc):
        tag = "T=0.3" if "T=0.3" in r["old_llm"] else "T=0"
        old, new = r["old_llm"][tag], r["new_llm"]
        common = sorted(set(old) & set(new))
        xo = np.array([old[k] for k in common])
        yn = np.array([new[k] for k in common])
        ax.scatter(xo, yn, s=4, alpha=0.3, color=r["color"])
        lo, hi = min(xo.min(), yn.min()), max(xo.max(), yn.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="1:1")
        rb = rmse_bias(yn - xo)
        ax.set_title(f"{r['name']}\n{tag} vs default:  RMSD={rb[0]:.2f}, mean Δ={rb[1]:+.2f} °C", fontsize=10)
        ax.set_xlabel(f"LLM July, {tag} (°C)")
        ax.set_ylabel("LLM July, default (°C)")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left")
    fig.suptitle("Per-point July LLM temperature: T=0.3 vs default (sub10, 1540 points)")
    fig.tight_layout()
    fig.savefig("png/temperature_july_scatter.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved png/temperature_july_rmse_bias.png, png/temperature_july_scatter.png")


def main():
    keys = sub10_keys()
    print(f"sub10 points: {len(keys)} | sampling ERA5 July...")
    era5 = era5_july_map(keys)

    results = []
    for m in MODELS:
        entry = {"name": m["name"], "color": m["color"], "old": {}, "old_llm": {},
                 "new": None, "new_llm": {}}
        for tag, path in m["old"].items():
            if not Path(path).exists():
                print(f"  WARN missing {path}")
                continue
            llm = old_llm_map(path, keys)
            entry["old_llm"][tag] = llm
            entry["old"][tag] = rmse_bias([llm[k] - era5[k] for k in llm])
        if m["new"] and Path(m["new"]).exists():
            nm = new_llm_map(m["new"])
            entry["new_llm"] = {k: v[0] for k, v in nm.items()}
            entry["new"] = rmse_bias([v[1] for v in nm.values()])
        results.append(entry)
        olds = "  ".join(f"{t}: {rb[0]:.2f}/{rb[1]:+.2f}" for t, rb in entry["old"].items())
        new = f"default: {entry['new'][0]:.2f}/{entry['new'][1]:+.2f}" if entry["new"] else "default: —"
        print(f"{m['name']:<22} {olds}   {new}   (RMSE/bias)")

    make_plots(results)


if __name__ == "__main__":
    main()
