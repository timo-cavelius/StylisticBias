#!/usr/bin/env python3
"""Per-variation prediction shift and cross-model spread.

For each variation the chart shows:
  - Gray line   : range (min → max) of per-model mean Δ
  - Shaded band : ±1 std around the cross-model mean
  - Dot         : cross-model mean Δ  (green = positive, red = negative)
  - ↔ symbol    : models disagree on direction (sign flip)

Sorted by |cross-model mean Δ| descending.

Output: output/evaluation/eval_charts/variation_shift_model_spread.png

Usage:
  python3 src/variation_shift_model_spread.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
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

VAL_DISPLAY = {
    "worn / distressed clothing": "Worn / Distressed clothing",
}

COLOR_POS  = "#2A8A4A"   # green  — mean Δ > 0
COLOR_NEG  = "#C04040"   # red    — mean Δ < 0
COLOR_BAND_POS = "#A8D8B0"
COLOR_BAND_NEG = "#F0B0A8"
COLOR_RANGE    = "#AAAAAA"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _discover_model_dirs(root: Path) -> list[Path]:
    return [
        e for e in sorted(root.iterdir())
        if e.is_dir() and (e / "variation_impact_summary.csv").exists()
    ]


def _load_per_model_means(model_dir: Path) -> dict[str, float]:
    """Mean Δ per variation for one model, averaged across genders and scenarios."""
    bucket: dict[str, list[float]] = defaultdict(list)
    with (model_dir / "variation_impact_summary.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            bucket[row["variation_name"]].append(float(row["mean_delta"]))
    return {var: float(np.mean(vals)) for var, vals in bucket.items()}


def _var_label(variation_name: str) -> str:
    _, _, val = variation_name.partition(":")
    val = val.strip()
    return VAL_DISPLAY.get(val.lower(), val)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(
    per_model: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    all_vars = sorted({v for d in per_model.values() for v in d})

    # ---- build cross-model stats per variation ----
    rows: list[dict] = []
    for var in all_vars:
        vals = [per_model[m].get(var, np.nan) for m in per_model]
        vals = [v for v in vals if np.isfinite(v)]
        if not vals:
            continue
        arr = np.array(vals)
        mean = float(np.mean(arr))
        std  = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        rows.append({
            "var":       var,
            "label":     _var_label(var),
            "mean":      mean,
            "std":       std,
            "min":       float(arr.min()),
            "max":       float(arr.max()),
            "sign_flip": bool(arr.min() < 0 < arr.max()),
        })

    # sort by |mean Δ| descending → most impactful at top (inverted y-axis)
    rows.sort(key=lambda r: abs(r["mean"]), reverse=True)

    n = len(rows)
    y = np.arange(n)

    all_x = [r["min"] for r in rows] + [r["max"] for r in rows]
    xabs  = max(abs(v) for v in all_x) * 1.25

    fig_h = max(10, n * 0.29)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F7F7F8")

    # ---- alternating row backgrounds ----
    for i in range(n):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, facecolor="white", alpha=0.55, zorder=0)

    # Signed background: left = very light red, right = very light green
    ax.axvspan(-xabs, 0,    facecolor="#FAE8E8", alpha=0.55, zorder=0)
    ax.axvspan(0,     xabs, facecolor="#E8F5EA", alpha=0.55, zorder=0)

    ax.axvline(0, color="#555555", linewidth=1.0, zorder=3)
    ax.xaxis.grid(True, color="white", linewidth=0.8, zorder=0)

    for i, row in enumerate(rows):
        mean      = row["mean"]
        std       = row["std"]
        lo        = row["min"]
        hi        = row["max"]
        sign_flip = row["sign_flip"]
        dot_color = COLOR_POS if mean >= 0 else COLOR_NEG
        band_col  = COLOR_BAND_POS if mean >= 0 else COLOR_BAND_NEG

        # ±1 std shaded band
        ax.fill_betweenx(
            [i - 0.18, i + 0.18],
            mean - std, mean + std,
            color=band_col, alpha=0.55, zorder=1,
        )

        # min–max range line
        ax.plot(
            [lo, hi], [i, i],
            color=COLOR_RANGE, linewidth=1.6,
            solid_capstyle="round", zorder=2,
        )

        # mean dot
        ax.plot(
            mean, i,
            marker="o", markersize=7,
            color=dot_color,
            markeredgecolor="white", markeredgewidth=0.7,
            zorder=4,
        )

        # sign-flip symbol
        if sign_flip:
            ax.text(
                hi + xabs * 0.025, i,
                "↔",
                va="center", ha="left",
                fontsize=8, color=COLOR_NEG,
                zorder=5,
            )

    # ---- axes ----
    ax.set_xlim(-xabs, xabs)
    ax.set_ylim(-0.5, n - 0.5)
    ax.invert_yaxis()
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=8.5)
    ax.set_xlabel("Mean Δ  (range = min/max across all models)", fontsize=10.5)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(axis="both", length=0)

    # ---- title ----
    ax.text(
        0.5, 1.04,
        "Per-Variation Prediction Shift and Cross-Model Spread",
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=13, fontweight="bold",
    )
    ax.text(
        0.5, 1.015,
        "↔ = sign flip (models disagree on direction)",
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=9.5, color="#555555", style="italic",
    )

    # ---- legend ----
    legend_handles = [
        Line2D([0], [0], color=COLOR_RANGE, linewidth=1.6, label="Range (min–max)"),
        Line2D([0], [0], marker="o", linestyle="None", markersize=7,
               color=COLOR_POS, markeredgecolor="white", markeredgewidth=0.7,
               label="Mean Δ > 0 (favorable)"),
        Line2D([0], [0], marker="o", linestyle="None", markersize=7,
               color=COLOR_NEG, markeredgecolor="white", markeredgewidth=0.7,
               label="Mean Δ < 0 (unfavorable)"),
        mpatches.Patch(facecolor=COLOR_BAND_POS, alpha=0.55, label="±1 std band"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        frameon=True, framealpha=0.92,
        edgecolor="#cccccc",
        fontsize=8.5,
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
    model_dirs = _discover_model_dirs(EVALUATION_ROOT)
    if not model_dirs:
        raise FileNotFoundError(f"No model dirs under {EVALUATION_ROOT}")
    print(f"Models: {[d.name for d in model_dirs]}")

    per_model = {d.name: _load_per_model_means(d) for d in model_dirs}
    plot(per_model, OUTPUT_DIR / "variation_shift_model_spread.png")


if __name__ == "__main__":
    main()
