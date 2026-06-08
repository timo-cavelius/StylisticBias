#!/usr/bin/env python3
"""Radar chart: per-category variation sensitivity profiles across models.

Each axis = one variation category (mean |Δ| averaged over all variations,
genders, and scenarios within that category). One filled polygon per model.

Output: output/evaluation/eval_charts/variation_category_radar.png

Usage:
  python3 src/variation_category_radar.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EVALUATION_ROOT = Path("output/evaluation")
OUTPUT_DIR      = Path("output/evaluation/eval_charts")

# Clockwise from top, matching the reference image
CATEGORY_ORDER = [
    "hair_style",
    "hair_length",
    "hair_color",
    "skin_irregularities",
    "accessories",
    "piercings",
    "eyewear",
    "fashion_style",
    "tattoos",
    "lip_makeup_female",
    "makeup_female",
    "facial_hair_male",
]

CAT_DISPLAY = {
    "hair_style":          "Hair style",
    "hair_length":         "Hair length",
    "hair_color":          "Hair color",
    "skin_irregularities": "Skin\nirregularities",
    "accessories":         "Accessories",
    "piercings":           "Piercings",
    "eyewear":             "Eyewear",
    "fashion_style":       "Fashion\nstyle",
    "tattoos":             "Tattoos",
    "lip_makeup_female":   "Lip makeup\n(F)",
    "makeup_female":       "Makeup\n(F)",
    "facial_hair_male":    "Facial hair\n(M)",
}

MODEL_DISPLAY = {
    "gemma3":     "Gemma-3",
    "gemma4":     "Gemma-4",
    "internvl":   "InternVL3",
    "llava_next": "LLaVA-v1.6",
    "pixtral":    "Pixtral",
    "qwen3":      "Qwen3-VL",
}

MODEL_COLORS = {
    "gemma3":     "#3A80C0",
    "gemma4":     "#4CAF50",
    "internvl":   "#F5A020",
    "llava_next": "#8B4F28",
    "pixtral":    "#909090",
    "qwen3":      "#00B4CC",
}

# Legend order matching reference image (row-major, 3 per row)
LEGEND_ORDER = ["gemma3", "llava_next", "qwen3", "gemma4", "pixtral", "internvl"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _discover_model_dirs(root: Path) -> list[Path]:
    return [
        e for e in sorted(root.iterdir())
        if e.is_dir() and (e / "variation_impact_summary.csv").exists()
    ]


def _load_category_strength(model_dir: Path) -> dict[str, float]:
    """Mean |Δ| per category for one model, averaged across all rows."""
    cat_vals: dict[str, list[float]] = defaultdict(list)
    with (model_dir / "variation_impact_summary.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            cat = row["variation_name"].partition(":")[0]
            cat_vals[cat].append(abs(float(row["mean_delta"])))
    return {cat: float(np.mean(vals)) for cat, vals in cat_vals.items()}


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _label_alignment(angle_from_top_cw: float) -> tuple[str, str]:
    """(ha, va) for a label at angle measured clockwise from top."""
    a = angle_from_top_cw % (2 * np.pi)
    eps = 0.15
    if a < eps or a > 2 * np.pi - eps:
        return "center", "bottom"
    if abs(a - np.pi) < eps:
        return "center", "top"
    ha = "left"  if np.sin(a) > 0 else "right"
    va = "bottom" if np.cos(a) > 0 else "top"
    return ha, va


def _draw_radar(
    models: list[str],
    strength: dict[str, dict[str, float]],
    categories: list[str],
    output_path: Path,
) -> None:
    N = len(categories)

    # Polar angles: top = π/2, going clockwise → subtract
    angles = [np.pi / 2 - 2 * np.pi * i / N for i in range(N)]
    angles_closed = angles + [angles[0]]

    # Scale
    all_vals = [strength[m].get(c, 0.0) for m in models for c in categories]
    raw_max  = max(all_vals)
    max_val  = np.ceil(raw_max / 0.05) * 0.05
    ring_vals = np.arange(0.05, max_val + 0.001, 0.05)

    # ---- Figure ----
    fig = plt.figure(figsize=(8, 8), facecolor="white")
    ax  = fig.add_axes([0.1, 0.12, 0.8, 0.78], polar=True)
    ax.set_facecolor("white")

    # ---- Concentric rings ----
    theta_full = np.linspace(0, 2 * np.pi, 360)
    for rv in ring_vals:
        is_outer = abs(rv - ring_vals[-1]) < 1e-9
        ax.plot(theta_full, [rv] * 360,
                color="#BBBBBB" if is_outer else "#DDDDDD",
                linewidth=1.0 if is_outer else 0.6,
                zorder=1)

    # ---- Radial spokes ----
    for angle in angles:
        ax.plot([angle, angle], [0, max_val],
                color="#CCCCCC", linewidth=0.7, zorder=1)

    # ---- Ring value labels (along the top-right spoke) ----
    label_angle = angles[0]
    for rv in ring_vals[:-1]:
        ax.text(label_angle, rv, f"{rv:.2f}",
                ha="center", va="bottom",
                fontsize=12, color="#AAAAAA", zorder=2)

    # ---- Polygons: fill then line ----
    for model in models:
        vals        = [strength[model].get(c, 0.0) for c in categories]
        vals_closed = vals + [vals[0]]
        color       = MODEL_COLORS.get(model, "#888888")

        ax.fill(angles_closed, vals_closed,
                color=color, alpha=0.12, zorder=3)
        ax.plot(angles_closed, vals_closed,
                color=color, linewidth=1.8, zorder=4,
                solid_capstyle="round")
        ax.scatter(angles, vals,
                   color=color, s=28, zorder=5,
                   edgecolors="white", linewidths=0.8)

    # ---- Category axis labels ----
    label_r = max_val * 1.04
    for i, (angle, cat) in enumerate(zip(angles, categories)):
        cw_angle = np.pi / 2 - angle   # convert back to clockwise-from-top
        ha, va   = _label_alignment(cw_angle)
        ax.text(angle, label_r,
                CAT_DISPLAY.get(cat, cat),
                ha=ha, va=va,
                fontsize=11.5, fontweight="bold",
                color="#333333", zorder=6,
                multialignment="center")

    # ---- Axes cleanup ----
    ax.set_ylim(0, max_val * 1.05)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines["polar"].set_visible(False)
    ax.grid(False)
OUT_DIR = Path("output/evaluation/eval_charts")


def _read_all() -> Dict[str, Dict[str, float]]:
    out = {}
    for m in sorted(EVAL_ROOT.iterdir()):
        if not m.is_dir() or not (m / "variation_impact_summary.csv").exists():
            continue
        d = {}
        with (m / "variation_impact_summary.csv").open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                var = r["variation_name"].split(":", 1)[0].strip()
                d[var] = d.get(var, 0.0) + abs(float(r["mean_delta"]))
        out[m.name] = d
    return out


def _plot_cat(cat: str, values: List[float], labels: List[str], out_png: Path) -> None:
    N = len(labels)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    vals = values + values[:1]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, vals, linewidth=1.5)
    ax.fill(angles, vals, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    data = _read_all()
    if not data:
        raise FileNotFoundError("No evaluation data")
    # select first model and category keys as representative
    first = next(iter(data))
    cats = list(data[first].keys())[:6]
    labels = cats
    values = [np.mean([data[m].get(c, 0.0) for m in data]) for c in cats]
    _plot_cat("all_models", values, labels, OUT_DIR / "variation_category_radar.png")


if __name__ == "__main__":
    main()
