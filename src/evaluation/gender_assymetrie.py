#!/usr/bin/env python3
"""Gender asymmetry chart: mean Δ per variation averaged across all models.

One horizontal bar pair (female / male) per variation, grouped by category.
To the right of each bar-pair the signed difference (female − male) is shown;
positive = female scored higher, negative = male scored higher.

Output: output/evaluation/eval_charts/gender_asymmetry.png

Usage:
  python3 src/gender_assymetrie.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EVALUATION_ROOT = Path("output/evaluation")
OUTPUT_DIR      = Path("output/evaluation/eval_charts")

CATEGORY_ORDER = [
    "fashion_style",
    "facial_hair_male",
    "lip_makeup_female",
    "makeup_female",
    "hair_style",
    "hair_color",
    "hair_length",
    "skin_irregularities",
    "eyewear",
    "accessories",
    "piercings",
    "tattoos",
]

CAT_DISPLAY = {
    "fashion_style":       "Fashion Style",
    "facial_hair_male":    "Facial Hair (Male)",
    "lip_makeup_female":   "Lip Makeup (Female)",
    "makeup_female":       "Makeup (Female)",
    "hair_style":          "Hair Style",
    "hair_color":          "Hair Color",
    "hair_length":         "Hair Length",
    "skin_irregularities": "Skin",
    "eyewear":             "Eyewear",
    "accessories":         "Accessories",
    "piercings":           "Piercings",
    "tattoos":             "Tattoos",
}

FEMALE_COLOR = "#EDA335"
MALE_COLOR   = "#6399C8"
BAR_H        = 0.34
Y_STEP       = 1.0
CAT_GAP      = 0.65


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _discover_model_dirs(root: Path) -> list[Path]:
    return [
        e for e in sorted(root.iterdir())
        if e.is_dir() and (e / "variation_impact_summary.csv").exists()
    ]


def _load_model_data(model_dir: Path) -> dict[str, dict[str, float]]:
    """Return {gender: {variation_name: mean_delta}} averaged across scenarios."""
    sums: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with (model_dir / "variation_impact_summary.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            sums[row["gender"]][row["variation_name"]].append(float(row["mean_delta"]))
    return {
        gender: {var: float(np.mean(vals)) for var, vals in var_dict.items()}
        for gender, var_dict in sums.items()
    }


def _average_across_models(
    all_model_data: dict[str, dict[str, dict[str, float]]]
) -> dict[str, dict[str, float]]:
    """Average each (gender, variation) mean_delta across all models."""
    collector: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for gender_data in all_model_data.values():
        for gender, var_vals in gender_data.items():
            for var, val in var_vals.items():
                collector[gender][var].append(val)
    return {
        gender: {var: float(np.mean(vals)) for var, vals in var_dict.items()}
        for gender, var_dict in collector.items()
    }


def _build_ordered_variations(
    avg_data: dict[str, dict[str, float]]
) -> list[tuple[str, str]]:
    """Variations ordered by category (CATEGORY_ORDER) then |avg delta| descending."""
    seen: set[str] = set()
    by_cat: dict[str, list[str]] = defaultdict(list)
    for var_vals in avg_data.values():
        for var in var_vals:
            if var not in seen:
                by_cat[var.split(":")[0]].append(var)
                seen.add(var)

    global_abs: dict[str, float] = {
        var: float(np.mean([abs(avg_data[g].get(var, 0.0)) for g in avg_data]))
        for var in seen
    }

    result: list[tuple[str, str]] = []
    for cat in CATEGORY_ORDER:
        for var in sorted(by_cat.get(cat, []), key=lambda v: global_abs[v], reverse=True):
            result.append((cat, var))
    for cat in sorted(by_cat):
        if cat not in CATEGORY_ORDER:
            for var in sorted(by_cat[cat], key=lambda v: global_abs[v], reverse=True):
                result.append((cat, var))
    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot(
    avg_data: dict[str, dict[str, float]],
    ordered_vars: list[tuple[str, str]],
    output_path: Path,
):
    female_vals = avg_data.get("female", {})
    male_vals   = avg_data.get("male",   {})

    # ---- Y positions with category gaps ----
    y_positions: list[float] = []
    separator_y: list[float] = []
    cat_y_map: dict[str, list[float]] = defaultdict(list)

    y = 0.0
    prev_cat = None
    for cat, _ in ordered_vars:
        if cat != prev_cat:
            if prev_cat is not None:
                separator_y.append(y + (Y_STEP - CAT_GAP) / 2)
                y -= CAT_GAP
            prev_cat = cat
        y_positions.append(y)
        cat_y_map[cat].append(y)
        y -= Y_STEP

    # ---- x limits ----
    all_vals = list(female_vals.values()) + list(male_vals.values())
    xmax = max((abs(v) for v in all_vals), default=0.3) * 1.18
    xlim_lo, xlim_hi = -xmax, xmax

    total_height = abs(y) + 1.0
    fig_height   = max(10, total_height * 0.38)
    fig, ax = plt.subplots(figsize=(10.5, fig_height))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # ---- Layer 1: red → white → green gradient ----
    cmap_bg = LinearSegmentedColormap.from_list("rg_bg", ["#F1C9C9", "#FCFCFD", "#CDEBD3"])
    gradient_data = np.linspace(-1, 1, 800).reshape(1, -1)
    y_top    = max(y_positions) + Y_STEP / 2
    y_bottom = min(y_positions) - Y_STEP / 2
    ax.imshow(
        gradient_data,
        aspect="auto",
        cmap=cmap_bg,
        extent=[xlim_lo, xlim_hi, y_bottom, y_top],
        origin="upper",
        vmin=-1, vmax=1,
        zorder=0,
    )

    # ---- Layer 2: alternating white stripe per category ----
    cat_items = list(cat_y_map.items())
    n_cats = len(cat_items)
    for i, (cat, ys) in enumerate(cat_items):
        y_stripe_top = separator_y[i - 1] if i > 0          else max(ys) + Y_STEP / 2
        y_stripe_bot = separator_y[i]     if i < n_cats - 1 else min(ys) - Y_STEP / 2
        if i % 2 == 0:
            ax.axhspan(y_stripe_bot, y_stripe_top, facecolor="white", alpha=0.38, zorder=1)

    # ---- Layer 3: bars + difference labels ----
    y_ticks: list[float] = []
    y_labels: list[str]  = []

    for yp, (cat, var) in zip(y_positions, ordered_vars):
        fval = female_vals.get(var, np.nan)
        mval = male_vals.get(var, np.nan)

        for val, color, offset in [
            (fval, FEMALE_COLOR, +BAR_H * 0.55),
            (mval, MALE_COLOR,   -BAR_H * 0.55),
        ]:
            if not np.isnan(val):
                ax.barh(
                    yp + offset, val,
                    height=BAR_H,
                    color=color,
                    edgecolor="white",
                    linewidth=0.4,
                    zorder=3,
                )

        # Difference annotation (female − male), placed just past the longer bar
        diff = fval - mval
        diff_color = FEMALE_COLOR
        prefix = "+" if diff > 0 else ""
        x_pos = max(fval, mval, 0) + xmax * 0.03
        ax.text(
            x_pos, yp,
            f"{prefix}{diff:.3f}",
            va="center", ha="left",
            fontsize=7.5,
            color=diff_color,
            fontweight="bold",
            clip_on=False,
            zorder=6,
        )

        y_ticks.append(yp)
        y_labels.append(var.partition(":")[2])

    # ---- Category separator lines ----
    for sy in separator_y:
        ax.axhline(sy, color="#999999", linewidth=0.9, linestyle="-", zorder=5)

    # ---- Category labels on the right (between bars and diff column) ----
    for cat, ys in cat_y_map.items():
        center_y = (ys[0] + ys[-1]) / 2
        ax.text(
            1.002, center_y,
            CAT_DISPLAY.get(cat, cat),
            transform=ax.get_yaxis_transform(),
            va="center", ha="left",
            fontsize=7.5, color="#555555", style="italic",
        )

    # ---- Axes decoration ----
    for xval in ax.get_xticks():
        ax.axvline(xval, color="white", linewidth=0.8, zorder=2)
    ax.axvline(0, color="#555555", linewidth=0.9, zorder=6)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=8.5)
    ax.set_xlim(xlim_lo, xlim_hi)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.set_xlabel("Mean Δ (female / male)", fontsize=10)
    ax.set_title("Gender asymmetry across appearance variations", fontsize=12.5, fontweight="bold")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")

    # ---- Legend ----
    legend_items = [
        mpatches.Patch(color=FEMALE_COLOR, label="female"),
        mpatches.Patch(color=MALE_COLOR, label="male"),
    ]
    ax.legend(handles=legend_items, loc="lower right", frameon=False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    model_dirs = _discover_model_dirs(EVALUATION_ROOT)
    if not model_dirs:
        raise FileNotFoundError("No model evaluation directories found.")

    all_model_data = {model_dir.name: _load_model_data(model_dir) for model_dir in model_dirs}
    avg_data = _average_across_models(all_model_data)
    ordered_vars = _build_ordered_variations(avg_data)
    _plot(avg_data, ordered_vars, OUTPUT_DIR / "gender_asymmetry.png")


if __name__ == "__main__":
    main()
