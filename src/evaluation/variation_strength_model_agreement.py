#!/usr/bin/env python3
"""Two-panel chart: variation strength per category & model agreement.

Chart 1 — Average variation strength per category (mean |Δ| across all
           models, genders, scenarios), sorted descending.
           Color encodes strength tier: high=orange, mid=steel-blue, low=gray.

Chart 2 — Model agreement on category strength: std of per-model category
           strength across the 6 models (same category order as chart 1).
           Color encodes disagreement tier: high=orchid, mid=steel-blue, low=gray.

Output:
    output/evaluation/eval_charts/variation_strength.png
    output/evaluation/eval_charts/model_agreement.png

Usage:
  python3 src/variation_strength_model_agreement.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EVALUATION_ROOT = Path("output/evaluation")
OUTPUT_DIR      = Path("output/evaluation/eval_charts")

CAT_DISPLAY = {
    "fashion_style":       "Fashion style",
    "tattoos":             "Tattoos",
    "eyewear":             "Eyewear",
    "lip_makeup_female":   "Lip makeup",
    "makeup_female":       "Makeup",
    "facial_hair_male":    "Facial hair",
    "hair_style":          "Hair style",
    "piercings":           "Piercings",
    "accessories":         "Accessories",
    "skin_irregularities": "Skin irreg.",
    "hair_length":         "Hair length",
    "hair_color":          "Hair color",
}

# Strength chart: 3-tier palette
COLOR_HIGH = "#E07B54"   # warm orange — strongest signal
COLOR_MID  = "#5B8DB8"   # steel blue  — moderate signal
COLOR_LOW  = "#B0B0B0"   # gray        — weakest signal

# Agreement chart: 3-tier palette (disagreement)
AGREE_HIGH = "#C76DAE"   # orchid  — models most disagree
AGREE_MID  = "#5B8DB8"   # steel blue
AGREE_LOW  = "#B0B0B0"   # gray


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _discover_model_dirs(root: Path) -> list[Path]:
    return [
        e for e in sorted(root.iterdir())
        if e.is_dir() and (e / "variation_impact_summary.csv").exists()
    ]


def _load_category_strength(model_dir: Path) -> dict[str, float]:
    """Mean |Δ| per category for one model."""
    cat_vals: dict[str, list[float]] = defaultdict(list)
    with (model_dir / "variation_impact_summary.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            cat = row["variation_name"].partition(":")[0]
            cat_vals[cat].append(abs(float(row["mean_delta"])))
    return {cat: float(np.mean(vals)) for cat, vals in cat_vals.items()}


def _build_category_tables(model_dirs: list[Path]):
    """
    Returns
    -------
    strength : dict[str, float]   — mean |Δ| averaged across all models
    agreement_std : dict[str, float] — std of per-model strength across models
    categories : list[str]        — sorted by strength descending
    """
    per_model: list[dict[str, float]] = [_load_category_strength(d) for d in model_dirs]
    all_cats = sorted({cat for d in per_model for cat in d})

    strength: dict[str, float] = {}
    agreement_std: dict[str, float] = {}

    for cat in all_cats:
        vals = [d.get(cat, 0.0) for d in per_model]
        strength[cat]      = float(np.mean(vals))
        agreement_std[cat] = float(np.std(vals, ddof=1))

    categories = sorted(all_cats, key=lambda c: strength[c], reverse=True)
    return strength, agreement_std, categories


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def _tier_colors(values: list[float], c_high: str, c_mid: str, c_low: str) -> list[str]:
    """Assign colors by tertile rank."""
    n = len(values)
    sorted_idx = np.argsort(values)[::-1]   # highest first
    colors = [c_low] * n
    for rank, idx in enumerate(sorted_idx):
        if rank < n / 3:
            colors[idx] = c_high
        elif rank < 2 * n / 3:
            colors[idx] = c_mid
    return colors


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _horizontal_bar_chart(
    categories: list[str],
    values: list[float],
    colors: list[str],
    title: str,
    subtitle: str,
    xlabel: str,
    legend_patches: list[mpatches.Patch],
    output_path: Path,
):
    n = len(categories)
    fig, ax = plt.subplots(figsize=(6.5, 0.42 * n + 1.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F7F7F8")

    y = np.arange(n)
    bars = ax.barh(y, values, color=colors, height=0.6, zorder=2)

    # Value labels
    x_max = max(values) * 1.01
    for bar, val in zip(bars, values):
        ax.text(
            val + x_max * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center", ha="left", fontsize=8.5, color="#444444",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(
        [CAT_DISPLAY.get(c, c) for c in categories],
        fontsize=9.5,
    )
    ax.invert_yaxis()
    ax.set_xlim(0, x_max * 1.18)
    ax.set_xlabel(xlabel, fontsize=9, color="#555555")

    ax.text(
        0, 1.11, title,
        transform=ax.transAxes, fontsize=11.5, fontweight="bold",
        va="bottom", clip_on=False,
    )
    ax.text(
        0, 1.03, subtitle,
        transform=ax.transAxes, fontsize=8, color="#777777",
        va="bottom", clip_on=False,
    )

    for spine in ["top", "right", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.tick_params(axis="both", length=0)
    ax.xaxis.grid(False)

    # Alternating row backgrounds
    for i in range(n):
        if i % 2 == 1:
            ax.axhspan(i - 0.5, i + 0.5, facecolor="white", alpha=0.6, zorder=1)

    if legend_patches:
        ax.legend(
            handles=legend_patches,
            loc="lower right",
            frameon=False,
            fontsize=8,
            handlelength=1.0,
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    model_dirs = _discover_model_dirs(EVALUATION_ROOT)
    if not model_dirs:
        raise FileNotFoundError(f"No model dirs found under {EVALUATION_ROOT}")
    print(f"Models: {[d.name for d in model_dirs]}")

    strength, agreement_std, categories = _build_category_tables(model_dirs)

    strength_vals = [strength[c] for c in categories]
    agree_vals    = [agreement_std[c] for c in categories]

    strength_colors = _tier_colors(strength_vals, COLOR_HIGH, COLOR_MID, COLOR_LOW)
    agree_colors    = _tier_colors(agree_vals,    AGREE_HIGH, AGREE_MID, AGREE_LOW)

    strength_legend = [
        mpatches.Patch(color=COLOR_HIGH, label="High strength"),
        mpatches.Patch(color=COLOR_MID,  label="Mid strength"),
        mpatches.Patch(color=COLOR_LOW,  label="Low strength"),
    ]
    agree_legend = [
        mpatches.Patch(color=AGREE_HIGH, label="High disagreement"),
        mpatches.Patch(color=AGREE_MID,  label="Mid disagreement"),
        mpatches.Patch(color=AGREE_LOW,  label="Low disagreement"),
    ]

    _horizontal_bar_chart(
        categories, strength_vals, strength_colors,
        title="Average variation strength per category",
        subtitle="mean |Δ| across all models, genders, and scenarios",
        xlabel="Mean |Δ|",
        legend_patches=strength_legend,
        output_path=OUTPUT_DIR / "variation_strength.png",
    )

    _horizontal_bar_chart(
        categories, agree_vals, agree_colors,
        title="Model agreement on category strength",
        subtitle="std of per-model category strength across models  (higher = models disagree more)",
        xlabel="Std of mean |Δ| across models",
        legend_patches=agree_legend,
        output_path=OUTPUT_DIR / "model_agreement.png",
    )


if __name__ == "__main__":
    main()
