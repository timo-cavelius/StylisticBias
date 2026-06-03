#!/usr/bin/env python3
"""Cumulative impact chart: cross-model average — few attributes dominate.

For each visual attribute, computes |mean Δ| averaged across models, genders,
and scenarios, then plots the cumulative share of total |Δ| as attributes are
added in descending order of impact.

Output: output/evaluation/eval_charts/cross_model_average.png

Usage:
  python3 src/cross_model_average.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EVALUATION_ROOT = Path("output/evaluation")
OUTPUT_DIR      = Path("output/evaluation/eval_charts")

LINE_COLOR  = "#1A55A8"
DOT_COLOR   = "#1A55A8"
ANNOT_COLOR = "#1A55A8"
THRESH      = 0.80          # cumulative share threshold to annotate


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _discover_model_dirs(root: Path) -> list[Path]:
    return [e for e in sorted(root.iterdir())
            if e.is_dir() and (e / "variation_impact_summary.csv").exists()]


def _compute_global_abs(model_dirs: list[Path]) -> dict[str, float]:
    """Per-variation |mean Δ| averaged across models.

    For each model: average signed Δ across all genders and scenarios first,
    then take the absolute value. Finally average across models.
    """
    collector: dict[str, list[float]] = defaultdict(list)
    for d in model_dirs:
        within: dict[str, list[float]] = defaultdict(list)
        with (d / "variation_impact_summary.csv").open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                within[row["variation_name"]].append(float(row["mean_delta"]))
        for var, vals in within.items():
            collector[var].append(abs(float(np.mean(vals))))
    return {var: float(np.mean(vals)) for var, vals in collector.items()}


LABEL_ABBREV: dict[str, str] = {
    "Worn / Distressed clothing":    "Distressed clothing",
    "Formal / Evening wear":         "Formal / Evening",
    "Professional / Business formal":"Prof. / Business",
    "Functional / outdoor wear":     "Functional / Outdoor",
    "Sporty / Athletic wear":        "Sporty / Athletic",
}


def _var_label(var: str) -> str:
    _, _, val = var.partition(":")
    val = val.strip()
    return LABEL_ABBREV.get(val, val)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(global_abs: dict[str, float], output_path: Path) -> None:
    sorted_vars = sorted(global_abs, key=global_abs.get, reverse=True)
    values_all  = [global_abs[v] for v in sorted_vars]
    total       = sum(values_all)
    cumsum_all  = list(np.cumsum(values_all) / total)

    # Index where cumulative first reaches THRESH
    idx_thresh = next(i for i, c in enumerate(cumsum_all) if c >= THRESH)
    n_thresh   = idx_thresh + 1

    # Truncate to threshold + 3 extra attributes
    keep        = idx_thresh + 1 + 3
    sorted_vars = sorted_vars[:keep]
    labels      = [_var_label(v) for v in sorted_vars]
    cumsum      = cumsum_all[:keep]
    x           = list(range(keep))

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # ---- Dashed grid ----
    ax.yaxis.grid(True, color="#CCCCCC", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_axisbelow(True)

    # ---- Main line ----
    ax.plot(x, cumsum,
            color=LINE_COLOR, linewidth=2.2,
            solid_capstyle="round", zorder=2)

    # ---- Filled dots ----
    ax.plot(x, cumsum,
            marker="o", markersize=10,
            color=DOT_COLOR, markeredgecolor="white",
            markeredgewidth=1.0, linestyle="None",
            zorder=3)

    # ---- Highlight dot at threshold (open circle) ----
    ax.plot(idx_thresh, cumsum[idx_thresh],
            marker="o", markersize=11,
            color="white",
            markeredgecolor=DOT_COLOR, markeredgewidth=2.2,
            linestyle="None", zorder=4)

    # ---- Horizontal dashed line at THRESH ----
    ax.axhline(THRESH, color=ANNOT_COLOR, linewidth=1.4,
               linestyle="--", zorder=1)

    # ---- Vertical dashed line at threshold index ----
    ax.axvline(idx_thresh, color=ANNOT_COLOR, linewidth=1.4,
               linestyle="--", zorder=1)

    # ---- Annotation ----
    ax.text(
        idx_thresh + 0.35,
        THRESH - 0.025,
        f"{n_thresh} attributes\n≈ {int(THRESH * 100)}% of total |Δ|",
        ha="left", va="top",
        fontsize=14, color=ANNOT_COLOR,
        style="italic",
    )

    # ---- Axes styling ----
    ax.set_xlim(-0.5, len(x) - 0.5)
    ax.set_ylim(0, 1.04)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=13)
    ax.set_ylabel("Cumulative share of total |Δ|", fontsize=15, fontweight="bold", labelpad=10)
    ax.set_xlabel("Visual attributes ranked by impact", fontsize=15, fontweight="bold", labelpad=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.tick_params(axis="y", labelsize=13)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(axis="both", length=0)

    # ---- Title ----
    ax.set_title(
        "A. Cross-model average: few attributes dominate",
        fontsize=18, fontweight="bold", pad=16,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")
    print(f"  {n_thresh} attributes account for ≥{int(THRESH*100)}% of total |Δ|"
          f"  (cumulative = {cumsum[idx_thresh]:.3f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    model_dirs = _discover_model_dirs(EVALUATION_ROOT)
    if not model_dirs:
        raise FileNotFoundError(f"No model dirs under {EVALUATION_ROOT}")
    print(f"Models: {[d.name for d in model_dirs]}")

    global_abs = _compute_global_abs(model_dirs)
    print(f"Variations: {len(global_abs)}")

    plot(global_abs, OUTPUT_DIR / "cross_model_average.png")


if __name__ == "__main__":
    main()
