#!/usr/bin/env python3
"""Create one horizontal bar chart per MLLM model showing mean delta split by gender.

For each model the chart shows all variation types on the y-axis, grouped and
sorted by category, with horizontal separator lines between categories.
Bar color encodes the value direction: red (negative) → green (positive).

Output: output/evaluation/eval_charts/<model>_gender_split.png

Usage:
  python3 src/gender_split_per_model.py
  python3 src/gender_split_per_model.py --evaluation-root output/evaluation
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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

CAT_LABEL_SHORT = {
    "fashion_style":       "fashion",
    "facial_hair_male":    "facial_hair",
    "lip_makeup_female":   "lip_makeup",
    "makeup_female":       "makeup",
    "hair_style":          "hair_style",
    "hair_color":          "hair_color",
    "hair_length":         "hair_length",
    "skin_irregularities": "skin",
    "eyewear":             "eyewear",
    "accessories":         "accessories",
    "piercings":           "piercings",
    "tattoos":             "tattoos",
}

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

MODEL_DISPLAY = {
    "gemma3":    "Gemma-3",
    "gemma4":    "Gemma-4",
    "internvl":  "InternVL3",
    "llava_next": "LLaVA-1.6",
    "pixtral":   "Pixtral",
    "qwen3":     "Qwen3-VL",
}

# Red → light gray → green
CMAP = LinearSegmentedColormap.from_list("rg", ["#C0392B", "#F0F0F0", "#27AE60"])

FEMALE_COLOR = "#EDA335"
MALE_COLOR   = "#6399C8"
BAR_H        = 0.34
Y_STEP       = 1.0
CAT_GAP      = 0.65


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _var_display(variation_name: str) -> str:
    _, _, val = variation_name.partition(":")
    return val


def _load_model_data(model_dir: Path) -> dict[str, dict[str, tuple[float, float]]]:
    """Return {gender: {variation_name: (mean_delta, std_across_scenarios)}}.

    std_across_scenarios is the standard deviation of per-scenario mean_delta
    values. A large std means scenarios pull the variation in opposite directions
    (they would cancel in the plain mean).
    """
    sums: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    csv_path = model_dir / "variation_impact_summary.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            sums[row["gender"]][row["variation_name"]].append(float(row["mean_delta"]))
    return {
        gender: {
            var: (float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
            for var, vals in var_dict.items()
        }
        for gender, var_dict in sums.items()
    }


def _discover_model_dirs(evaluation_root: Path) -> list[Path]:
    dirs = []
    for entry in sorted(evaluation_root.iterdir()):
        if entry.is_dir() and (entry / "variation_impact_summary.csv").exists():
            dirs.append(entry)
    return dirs


def _build_ordered_variations(
    all_model_data: dict[str, dict[str, dict[str, float]]]
) -> list[tuple[str, str]]:
    """Return [(category, variation_name), ...] ordered by category then by
    global avg |delta| descending within each category."""
    by_cat: dict[str, list[str]] = defaultdict(list)
    seen: set[str] = set()
    for gender_data in all_model_data.values():
        for gender_vals in gender_data.values():
            for var in gender_vals:
                if var not in seen:
                    cat = var.split(":")[0]
                    by_cat[cat].append(var)
                    seen.add(var)

    # Global avg |delta| per variation for stable sort across models
    global_abs: dict[str, float] = {}
    for var in seen:
        vals = [
            abs(gender_data[g].get(var, (0.0, 0.0))[0])
            for gender_data in all_model_data.values()
            for g in gender_data
            if var in gender_data[g]
        ]
        global_abs[var] = float(np.mean(vals)) if vals else 0.0

    result: list[tuple[str, str]] = []
    for cat in CATEGORY_ORDER:
        variations = sorted(by_cat.get(cat, []), key=lambda v: global_abs[v], reverse=True)
        for var in variations:
            result.append((cat, var))
    # Append any unlisted categories at the end
    for cat in sorted(by_cat):
        if cat not in CATEGORY_ORDER:
            variations = sorted(by_cat[cat], key=lambda v: global_abs[v], reverse=True)
            for var in variations:
                result.append((cat, var))
    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_model(
    model_name: str,
    gender_data: dict[str, dict[str, float]],
    ordered_vars: list[tuple[str, str]],
    output_path: Path,
):
    # ---- Build y positions with a gap between category blocks ----
    # At the moment we detect a new category, `y` has already been decremented
    # by Y_STEP past the last variation of the previous category.
    # So: last_yp = y + Y_STEP, first_new_yp = y - CAT_GAP
    # Correct separator midpoint: y + (Y_STEP - CAT_GAP) / 2
    y_positions: list[float] = []
    separator_y: list[float] = []
    cat_y_map: dict[str, list[float]] = defaultdict(list)  # cat -> list of y positions

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

    female_vals = gender_data.get("female", {})
    male_vals   = gender_data.get("male", {})

    # Compute x limits from data so background spans don't blow out the axis
    all_bar_vals = list(female_vals.values()) + list(male_vals.values())
    xmax = max((abs(mean) + std for mean, std in all_bar_vals), default=0.3) * 1.18
    xlim_lo, xlim_hi = -xmax, xmax

    # ---- Figure setup ----
    total_height = abs(y) + 1.0
    fig_height = max(10, total_height * 0.38)
    fig, ax = plt.subplots(figsize=(9.5, fig_height))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y_ticks: list[float] = []
    y_labels: list[str]  = []

    # ---- Layer 1: smooth red→white→green gradient background ----
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

    # ---- Layer 2: alternating white overlay per category (striped effect) ----
    # Use separator_y positions as stripe bounds so stripes and lines share the same edge.
    cat_items = list(cat_y_map.items())  # ordered by CATEGORY_ORDER via insertion order
    n_cats = len(cat_items)
    for i, (cat, ys) in enumerate(cat_items):
        y_stripe_top = separator_y[i - 1] if i > 0           else max(ys) + Y_STEP / 2
        y_stripe_bot = separator_y[i]     if i < n_cats - 1  else min(ys) - Y_STEP / 2
        if i % 2 == 0:
            ax.axhspan(y_stripe_bot, y_stripe_top, facecolor="white", alpha=0.38, zorder=1)

    # ---- Layer 3: bars + scenario-spread whiskers ----
    for yp, (cat, var) in zip(y_positions, ordered_vars):
        for vals, color, offset in [
            (female_vals, FEMALE_COLOR, +BAR_H * 0.55),
            (male_vals,   MALE_COLOR,   -BAR_H * 0.55),
        ]:
            mean_val, std_val = vals.get(var, (0.0, 0.0))
            ax.barh(
                yp + offset,
                mean_val,
                height=BAR_H,
                color=color,
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
            )
            # Whisker: ± std of per-scenario mean Δ, centred on the bar tip.
            # A wide whisker means scenarios diverge (some push +, others −).
            if std_val > 0:
                ax.errorbar(
                    mean_val, yp + offset,
                    xerr=std_val,
                    fmt="none",
                    ecolor=color,
                    elinewidth=1.4,
                    capsize=2.8,
                    capthick=1.4,
                    alpha=0.65,
                    zorder=4,
                )

        y_ticks.append(yp)
        y_labels.append(_var_display(var))

    # ---- Category separator lines ----
    for sy in separator_y:
        ax.axhline(sy, color="#999999", linewidth=0.9, linestyle="-", zorder=5)

    # ---- Category labels on the right margin ----
    for cat, ys in cat_y_map.items():
        center_y = (ys[0] + ys[-1]) / 2
        ax.text(
            1.002, center_y,
            CAT_DISPLAY.get(cat, cat),
            transform=ax.get_yaxis_transform(),
            va="center", ha="left",
            fontsize=7.5,
            color="#555555",
            style="italic",
        )

    ax.set_xlim(xlim_lo, xlim_hi)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=8.5)

    # White grid lines on top of background, below bars
    for xval in ax.get_xticks():
        ax.axvline(xval, color="white", linewidth=0.8, zorder=2)

    ax.axvline(0, color="#555555", linewidth=0.9, zorder=6)
    ax.set_xlabel("Mean Δ (averaged across scenarios)", fontsize=10)
    ax.set_title(MODEL_DISPLAY.get(model_name, model_name), fontsize=14, fontweight="bold", pad=12)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(axis="both", length=0)

    # Legend
    from matplotlib.lines import Line2D
    female_patch  = mpatches.Patch(facecolor=FEMALE_COLOR, edgecolor="white", linewidth=0.4, label="Female")
    male_patch    = mpatches.Patch(facecolor=MALE_COLOR,   edgecolor="white", linewidth=0.4, label="Male")
    whisker_entry = Line2D([0], [0], color="#888888", linewidth=1.4,
                           marker="|", markersize=6, markeredgewidth=1.4,
                           label="± std across scenarios")
    ax.legend(handles=[female_patch, male_patch, whisker_entry], loc="lower right",
              fontsize=9, frameon=True, framealpha=0.9, edgecolor="#cccccc")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Gender-split variation impact charts per model.")
    parser.add_argument("--evaluation-root", type=Path, default=Path("output/evaluation"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/evaluation/eval_charts"))
    args = parser.parse_args()

    model_dirs = _discover_model_dirs(args.evaluation_root)
    if not model_dirs:
        raise FileNotFoundError(f"No model dirs found under {args.evaluation_root}")

    print(f"Found models: {[d.name for d in model_dirs]}")

    all_model_data: dict[str, dict[str, dict[str, float]]] = {
        d.name: _load_model_data(d) for d in model_dirs
    }

    # Global color normalization across all models and genders
    all_vals = [
        mean
        for gender_data in all_model_data.values()
        for gender_vals in gender_data.values()
        for mean, _std in gender_vals.values()
    ]
    global_vmax = max(abs(v) for v in all_vals) if all_vals else 1.0

    ordered_vars = _build_ordered_variations(all_model_data)
    print(f"Total variations: {len(ordered_vars)}")

    for model_dir in model_dirs:
        name = model_dir.name
        output_path = args.output_dir / f"{name}_gender_split.png"
        _plot_model(name, all_model_data[name], ordered_vars, output_path)


if __name__ == "__main__":
    main()
