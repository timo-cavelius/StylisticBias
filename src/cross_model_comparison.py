#!/usr/bin/env python3
"""Cross-model comparison: Mean Δ for high-impact variations.

Grouped bar chart — one group per top variation, one bar per model.
Vertical red→white→green gradient background (white at y=0).

Output: output/evaluation/eval_charts/cross_model_comparison.png

Usage:
  python3 src/cross_model_comparison.py
  python3 src/cross_model_comparison.py --top-n 12
"""

from __future__ import annotations

import argparse
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

MODEL_DISPLAY = {
    "gemma3":    "Gemma-3",
    "gemma4":    "Gemma-4",
    "internvl":  "InternVL3",
    "llava_next": "LLaVA-1.6",
    "pixtral":   "Pixtral",
    "qwen3":     "Qwen3-VL",
}

MODEL_COLORS = {
    "gemma3":    "#6399C8",
    "gemma4":    "#74BC74",
    "internvl":  "#F5A623",
    "llava_next": "#E06B5A",
    "pixtral":   "#7B6FBF",
    "qwen3":     "#5BC4D8",
}

CAT_PREFIX_DISPLAY = {
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

VAL_DISPLAY = {
    "worn / distressed clothing": "distressed clothing",
}

CMAP_BG = LinearSegmentedColormap.from_list("rg_bg", ["#F1C9C9", "#FCFCFD", "#CDEBD3"])


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _discover_model_dirs(evaluation_root: Path) -> list[Path]:
    return [
        e for e in sorted(evaluation_root.iterdir())
        if e.is_dir() and (e / "variation_impact_summary.csv").exists()
    ]


def _load_variation_data(model_dir: Path) -> dict[str, float]:
    """Mean delta per variation, averaged across all genders and scenarios."""
    sums: dict[str, list[float]] = defaultdict(list)
    with (model_dir / "variation_impact_summary.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            sums[row["variation_name"]].append(float(row["mean_delta"]))
    return {var: float(np.mean(vals)) for var, vals in sums.items()}


def _select_top_variations(all_model_data: dict[str, dict[str, float]], top_n: int) -> list[str]:
    all_vars = {v for d in all_model_data.values() for v in d}
    global_abs = {
        var: float(np.mean([abs(d.get(var, 0.0)) for d in all_model_data.values()]))
        for var in all_vars
    }
    return sorted(all_vars, key=lambda v: global_abs[v], reverse=True)[:top_n]


def _var_label(variation_name: str) -> str:
    cat, _, val = variation_name.partition(":")
    prefix = CAT_PREFIX_DISPLAY.get(cat, cat)
    val = VAL_DISPLAY.get(val, val)
    return f"{prefix}:\n{val}"


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _plot(
    all_model_data: dict[str, dict[str, float]],
    models: list[str],
    top_vars: list[str],
    output_path: Path,
):
    n_vars   = len(top_vars)
    n_models = len(models)

    group_width = 0.72
    bar_w = group_width / n_models
    x_centers = np.arange(n_vars, dtype=float)

    # ---- Compute y limits from data ----
    all_vals = [all_model_data[m].get(v, 0.0) for m in models for v in top_vars]
    yabs = max(abs(v) for v in all_vals) * 1.25
    ylim_lo, ylim_hi = -yabs, yabs

    fig, ax = plt.subplots(figsize=(13, 6.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # ---- Layer 1: vertical red→white→green gradient ----
    n_grad = 800
    y_vals = np.linspace(ylim_lo, ylim_hi, n_grad)
    # Map: y<0 → [-1,0], y>0 → [0,1], ensuring white exactly at y=0
    grad_norm = np.where(y_vals < 0, y_vals / abs(ylim_lo), y_vals / ylim_hi)
    gradient_data = grad_norm.reshape(-1, 1)

    xlim_lo = x_centers[0]  - group_width / 2 - 0.2
    xlim_hi = x_centers[-1] + group_width / 2 + 0.2
    ax.imshow(
        gradient_data,
        aspect="auto",
        cmap=CMAP_BG,
        extent=[xlim_lo, xlim_hi, ylim_lo, ylim_hi],
        origin="lower",
        vmin=-1, vmax=1,
        zorder=0,
    )

    # ---- Layer 2: alternating white vertical band per variation group ----
    for i in range(n_vars):
        if i % 2 == 0:
            ax.axvspan(
                x_centers[i] - 0.5,
                x_centers[i] + 0.5,
                facecolor="white", alpha=0.35, zorder=1,
            )

    # ---- Layer 3: horizontal grid lines ----
    ax.yaxis.grid(True, color="white", linewidth=0.8, zorder=2)
    ax.set_axisbelow(False)

    # ---- Layer 4: bars ----
    for mi, model in enumerate(models):
        offset = (mi - (n_models - 1) / 2) * bar_w
        vals = [all_model_data[model].get(v, 0.0) for v in top_vars]
        ax.bar(
            x_centers + offset,
            vals,
            width=bar_w * 0.9,
            color=MODEL_COLORS.get(model, "#888888"),
            label=MODEL_DISPLAY.get(model, model),
            zorder=3,
            edgecolor="white",
            linewidth=0.4,
        )

    # ---- Axes decoration ----
    ax.axhline(0, color="#555555", linewidth=0.9, zorder=4)
    ax.set_xlim(xlim_lo, xlim_hi)
    ax.set_ylim(ylim_lo, ylim_hi)
    ax.set_xticks(x_centers)
    ax.set_xticklabels(
        [_var_label(v) for v in top_vars],
        fontsize=9, ha="right", rotation=35, rotation_mode="anchor",
    )
    ax.set_ylabel("Mean Δ", fontsize=11)
    ax.set_title(
        "Cross-Model Comparison: Mean Δ for High-Impact Variations",
        fontsize=13, fontweight="bold", pad=14,
    )
    ax.text(
        0.0, 1.02,
        "Signed mean delta for top variations, grouped by variation. "
        "Shows where models agree or diverge in direction and magnitude.",
        transform=ax.transAxes, fontsize=8.5, color="#555555",
    )

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(axis="both", length=0)

    # Legend
    handles = [
        mpatches.Patch(facecolor=MODEL_COLORS.get(m, "#888"), label=MODEL_DISPLAY.get(m, m))
        for m in models
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.13),
        ncol=n_models,
        frameon=False,
        fontsize=9,
        handlelength=1.2,
        columnspacing=1.2,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, default=Path("output/evaluation"))
    parser.add_argument("--output-dir",      type=Path, default=Path("output/evaluation/eval_charts"))
    parser.add_argument("--top-n",           type=int,  default=15)
    args = parser.parse_args()

    model_dirs = _discover_model_dirs(args.evaluation_root)
    if not model_dirs:
        raise FileNotFoundError(f"No model dirs found under {args.evaluation_root}")

    models = [d.name for d in model_dirs]
    print(f"Models: {models}")

    all_model_data = {d.name: _load_variation_data(d) for d in model_dirs}
    top_vars = _select_top_variations(all_model_data, args.top_n)
    print(f"Top {args.top_n} variations selected.")

    _plot(all_model_data, models, top_vars, args.output_dir / "cross_model_comparison.png")


if __name__ == "__main__":
    main()
