#!/usr/bin/env python3
"""Scenario sensitivity chart: top-N scenarios by mean |Δ| across models.

Horizontal bar chart with min–max whisker and mean marker per scenario,
coloured by scenario category. Top 10 scenarios shown by default.

Output: output/evaluation/eval_charts/scen_sens.png

Usage:
  python3 src/scen_sens.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EVALUATION_ROOT = Path("output/evaluation")
OUTPUT_DIR      = Path("output/evaluation/eval_charts")
TOP_N           = 10

SCENARIO_CATEGORIES: dict[str, str] = {
    "Competent | Incompetent":        "Personality & Social",
    "Likeable | Unlikeable":          "Personality & Social",
    "Intelligent | Unintelligent":    "Personality & Social",
    "Responsible | Irresponsible":    "Personality & Social",
    "Open-minded | Closed-minded":    "Personality & Social",
    "Conscientious | Careless":       "Personality & Social",
    "Extraverted | Introverted":      "Personality & Social",
    "Emotionally stable | Anxious":   "Personality & Social",
    "Confident | Insecure":           "Personality & Social",
    "Curious | Indifferent":          "Personality & Social",
    "Loving | Cold":                  "Interpersonal",
    "Trustworthy | Untrustworthy":    "Interpersonal",
    "Friendly | Unfriendly":          "Interpersonal",
    "Loyal | Disloyal":               "Interpersonal",
    "Polite | Rude":                  "Interpersonal",
    "Honest | Fraudulent":            "Interpersonal",
    "Obedient | Unruly":              "Behavioral",
    "Peaceful | Controversial":       "Behavioral",
    "Rational | Emotional":           "Behavioral",
    "Independent | Dependent":        "Behavioral",
    "Home owner | Renter":            "Socioeconomic & App.",
    "Educated | Uneducated":          "Socioeconomic & App.",
    "Attractive | Unattractive":      "Socioeconomic & App.",
    "Stylish | Unstylish":            "Socioeconomic & App.",
    "Wealthy | Poor":                 "Socioeconomic & App.",
}

CATEGORY_COLORS = {
    "Personality & Social": "#6aaed6",
    "Interpersonal":        "#74c476",
    "Behavioral":           "#fd8d3c",
    "Socioeconomic & App.": "#e8435a",
}

CATEGORY_BG = {
    "Personality & Social": "#E8F3FA",
    "Interpersonal":        "#E9F5E9",
    "Behavioral":           "#FEF0E6",
    "Socioeconomic & App.": "#FAE9EB",
}

CATEGORY_ORDER = [
    "Personality & Social",
    "Interpersonal",
    "Behavioral",
    "Socioeconomic & App.",
]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _latest_comparison_dir(root: Path) -> Path:
    dirs = sorted(d for d in root.iterdir()
                  if d.is_dir() and d.name.startswith("model_comparison"))
    if not dirs:
        raise FileNotFoundError(f"No model_comparison folder in {root}")
    return dirs[-1]


def _load(csv_path: Path) -> list[dict]:
    with csv_path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _compute(rows: list[dict], top_n: int) -> list[dict]:
    result = []
    for row in rows:
        label = row["scenario_label"].strip()
        vals = []
        for k, v in row.items():
            if k.endswith("_mean_delta"):
                try:
                    vals.append(abs(float(v)))
                except ValueError:
                    pass
        if not vals:
            continue
        result.append({
            "label":    label,
            "category": SCENARIO_CATEGORIES.get(label, "Personality & Social"),
            "mean":     float(np.mean(vals)),
            "min":      float(min(vals)),
            "max":      float(max(vals)),
        })
    result.sort(key=lambda d: d["mean"], reverse=True)
    return result[:top_n]


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(data: list[dict], output_path: Path) -> None:
    # Plot bottom-to-top so highest bar is at top
    data_plot = list(reversed(data))
    n         = len(data_plot)
    ys        = list(range(n))

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # ---- Row background bands ----
    for i, d in enumerate(data_plot):
        ax.axhspan(i - 0.45, i + 0.45,
                   facecolor=CATEGORY_BG[d["category"]],
                   alpha=0.7, zorder=0)

    # ---- Bars ----
    bar_h = 0.55
    for i, d in enumerate(data_plot):
        color = CATEGORY_COLORS[d["category"]]
        ax.barh(i, d["mean"],
                height=bar_h,
                color=color,
                alpha=0.85,
                zorder=2)

    # ---- Min–max whiskers ----
    cap_h = 0.22
    for i, d in enumerate(data_plot):
        ax.plot([d["min"], d["max"]], [i, i],
                color="#555555", linewidth=1.8,
                solid_capstyle="butt", zorder=3)
        ax.plot([d["min"], d["min"]], [i - cap_h, i + cap_h],
                color="#555555", linewidth=1.8,
                solid_capstyle="butt", zorder=3)
        ax.plot([d["max"], d["max"]], [i - cap_h, i + cap_h],
                color="#555555", linewidth=1.8,
                solid_capstyle="butt", zorder=3)

    # ---- Mean markers (open circle) ----
    for i, d in enumerate(data_plot):
        ax.plot(d["mean"], i,
                marker="o", markersize=9,
                color="white",
                markeredgecolor="#333333",
                markeredgewidth=1.8,
                zorder=4)
        ax.text(d["mean"], i - 0.32,
                f"{d['mean']:.3f}",
                va="top", ha="center",
                fontsize=13, color="#222222",
                zorder=5)

    # ---- Y-axis labels ----
    ax.set_yticks(ys)
    ax.set_yticklabels([d["label"] for d in data_plot], fontsize=13.5)

    # ---- X-axis ----
    x_max = max(d["max"] for d in data) * 1.08
    ax.set_xlim(0, x_max)
    ax.set_xlabel("Scenario sensitivity, mean |Δ| across models", fontsize=14.5, labelpad=8)
    ax.tick_params(axis="x", labelsize=13, length=0)
    ax.tick_params(axis="y", length=0)

    ax.xaxis.grid(True, color="#DDDDDD", linewidth=0.7, linestyle=":", zorder=1)
    ax.set_axisbelow(True)

    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#cccccc")

    # ---- Legend ----
    cat_handles = [
        mpatches.Patch(facecolor=CATEGORY_COLORS[c], edgecolor="none", label=c)
        for c in CATEGORY_ORDER
    ]
    range_handle = Line2D([0], [0], color="#555555", linewidth=1.8,
                          solid_capstyle="butt",
                          label="Model range")
    mean_handle  = Line2D([0], [0], marker="o", linestyle="None",
                          markersize=9, color="white",
                          markeredgecolor="#333333", markeredgewidth=1.8,
                          label="Mean")
    legend_handles = cat_handles + [range_handle, mean_handle]

    ax.legend(
        handles=legend_handles,
        loc="lower right",
        ncol=2,
        fontsize=12,
        frameon=True,
        framealpha=0.95,
        edgecolor="#cccccc",
        columnspacing=1.2,
        handlelength=1.6,
    )

    # ---- Subtitle ----
    ax.set_title(
        "A shared set of scenarios dominates visual sensitivity",
        fontsize=16, fontweight="bold", pad=14,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    comp_dir = _latest_comparison_dir(EVALUATION_ROOT)
    print(f"Using: {comp_dir.name}")
    rows = _load(comp_dir / "scenario_comparison.csv")
    data = _compute(rows, TOP_N)
    print(f"Top {TOP_N} scenarios:")
    for d in data:
        print(f"  {d['label'][:40]:40s}  mean={d['mean']:.3f}  [{d['min']:.3f}, {d['max']:.3f}]")
    plot(data, OUTPUT_DIR / "scen_sens.png")


if __name__ == "__main__":
    main()
