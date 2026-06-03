#!/usr/bin/env python3
"""Create two scenario-level charts matching the paper style.

Plot 1: Mean Absolute Prediction Shift by Scenario Category and Model
  - Grouped bar chart; one group per model, four bars per group (one per category)

Plot 2: All 25 Scenarios: Average Prediction Shift (sorted)
  - Horizontal dot-and-range chart sorted by cross-model average mean Δ

Usage:
  python3 src/create_scenario_charts.py
  python3 src/create_scenario_charts.py --comparison-dir output/evaluation/model_comparison_20260425_214738
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np


# ---------------------------------------------------------------------------
# Scenario → category mapping
# ---------------------------------------------------------------------------
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

CATEGORIES = ["Personality & Social", "Interpersonal", "Behavioral", "Socioeconomic & App."]

CATEGORY_COLORS = {
    "Personality & Social": "#6aaed6",
    "Interpersonal":        "#74c476",
    "Behavioral":           "#fd8d3c",
    "Socioeconomic & App.": "#e8435a",
}

CATEGORY_BG_COLORS = {
    "Personality & Social": "#E8F3FA",
    "Interpersonal":        "#E9F5E9",
    "Behavioral":           "#FEF0E6",
    "Socioeconomic & App.": "#FAE9EB",
}

MODEL_DISPLAY = {
    "gemma3":    "Gemma-3",
    "gemma4":    "Gemma-4",
    "internvl":  "InternVL3",
    "llava_next": "LLaVA-v1.6",
    "pixtral":   "Pixtral",
    "qwen3":     "Qwen3-VL",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _discover_latest_comparison(evaluation_root: Path) -> Path:
    candidates = sorted(
        [d for d in evaluation_root.iterdir() if d.is_dir() and d.name.startswith("model_comparison")],
        key=lambda p: p.name,
    )
    if not candidates:
        raise FileNotFoundError(f"No model_comparison folder found in {evaluation_root}")
    return candidates[-1]


def _load_scenario_data(comparison_dir: Path) -> list[dict]:
    csv_path = comparison_dir / "scenario_comparison.csv"
    rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _get_model_columns(rows: list[dict]) -> list[str]:
    if not rows:
        return []
    keys = list(rows[0].keys())
    return [k.replace("_mean_delta", "") for k in keys if k.endswith("_mean_delta")]


# ---------------------------------------------------------------------------
# Plot 1: Grouped bar chart by category and model
# ---------------------------------------------------------------------------

def plot_category_bars(rows: list[dict], models: list[str], output_path: Path):
    # Compute mean |Δ| per model per category
    cat_model_values: dict[str, dict[str, list[float]]] = {
        cat: {m: [] for m in models} for cat in CATEGORIES
    }

    for row in rows:
        label = (row.get("scenario_label") or "").strip()
        cat = SCENARIO_CATEGORIES.get(label)
        if cat is None:
            continue
        for m in models:
            val_str = row.get(f"{m}_mean_delta", "")
            try:
                cat_model_values[cat][m].append(abs(float(val_str)))
            except (ValueError, TypeError):
                pass

    means: dict[str, dict[str, float]] = {}
    for cat in CATEGORIES:
        means[cat] = {}
        for m in models:
            vals = cat_model_values[cat][m]
            means[cat][m] = float(np.mean(vals)) if vals else 0.0

    n_models = len(models)
    n_cats = len(CATEGORIES)
    group_width   = 0.96
    bar_width     = group_width / n_cats
    model_spacing = group_width + bar_width * 0.92   # gap = one bar width
    x = np.arange(n_models) * model_spacing

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F7F7F7")
    ax.grid(axis="y", color="white", linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)

    for ci, cat in enumerate(CATEGORIES):
        offsets = (ci - (n_cats - 1) / 2) * bar_width
        vals = [means[cat][m] for m in models]
        bars = ax.bar(
            x + offsets,
            vals,
            width=bar_width * 0.92,
            color=CATEGORY_COLORS[cat],
            label=cat,
            zorder=3,
            linewidth=0.4,
            edgecolor="#888888",
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{v:.3f}",
                ha="center", va="bottom",
                fontsize=10,
                color="#333333",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in models], fontsize=14)
    ax.set_xlim(x[0] - 0.60, x[-1] + 0.60)
    ax.set_ylabel("Mean |Δ| (absolute shift)", fontsize=14, labelpad=8)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.14)
    ax.set_title("Mean Absolute Prediction Shift by Scenario Category and Model", fontsize=15, fontweight="bold", pad=16)

    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", length=0)

    legend_handles = [
        mpatches.Patch(facecolor=CATEGORY_COLORS[cat], edgecolor="#888888", linewidth=0.4, label=cat)
        for cat in CATEGORIES
    ]
    ax.legend(
        handles=legend_handles,
        ncol=2,
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        fontsize=12,
        edgecolor="#cccccc",
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot 2: Horizontal sorted dot-and-range chart across all scenarios
# ---------------------------------------------------------------------------

def plot_scenario_shifts(rows: list[dict], models: list[str], output_path: Path):
    # Build per-scenario average mean Δ and per-model values
    scenario_data: list[dict] = []
    for row in rows:
        label = (row.get("scenario_label") or "").strip()
        cat = SCENARIO_CATEGORIES.get(label, "Personality & Social")
        model_vals: list[float] = []
        for m in models:
            val_str = row.get(f"{m}_mean_delta", "")
            try:
                model_vals.append(float(val_str))
            except (ValueError, TypeError):
                pass
        if not model_vals:
            continue
        avg = float(np.mean(model_vals))
        scenario_data.append({
            "label": label,
            "category": cat,
            "avg": avg,
            "min": min(model_vals),
            "max": max(model_vals),
        })

    scenario_data.sort(key=lambda d: d["avg"])

    n = len(scenario_data)
    fig_height = max(8, n * 0.42)
    fig, ax = plt.subplots(figsize=(13, fig_height))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y_positions = np.arange(n)

    # Draw alternating background bands per category block
    prev_cat = None
    band_start = 0
    bands: list[tuple[int, int, str]] = []
    for i, d in enumerate(scenario_data):
        if d["category"] != prev_cat:
            if prev_cat is not None:
                bands.append((band_start, i - 1, prev_cat))
            band_start = i
            prev_cat = d["category"]
    if prev_cat is not None:
        bands.append((band_start, n - 1, prev_cat))

    for start, end, cat in bands:
        ax.axhspan(start - 0.5, end + 0.5, facecolor=CATEGORY_BG_COLORS[cat], alpha=0.55, zorder=0)

    ax.axvline(0, color="#888888", linewidth=1.0, linestyle="--", zorder=1)
    ax.grid(axis="x", color="#dddddd", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    for i, d in enumerate(scenario_data):
        color = CATEGORY_COLORS[d["category"]]
        ax.plot(
            [d["min"], d["max"]],
            [i, i],
            color=color,
            linewidth=2.8,
            solid_capstyle="round",
            zorder=2,
            alpha=0.6,
        )
        ax.plot(
            d["avg"], i,
            marker="D",
            markersize=12,
            color=color,
            markeredgecolor="white",
            markeredgewidth=1.5,
            zorder=3,
        )
        sign = "+" if d["avg"] >= 0 else ""
        x_offset = 0.012 if d["avg"] >= 0 else -0.012
        ha = "left" if d["avg"] >= 0 else "right"
        ax.text(
            d["avg"] + x_offset, i - 0.28,
            f"{sign}{d['avg']:.3f}",
            va="top", ha=ha,
            fontsize=11,
            color=color,
            fontweight="bold",
            zorder=4,
        )

    # Left axis: negative pole (text after "|") — dark red
    neg_labels = [d["label"].split("|")[1].strip() for d in scenario_data]
    pos_labels = [d["label"].split("|")[0].strip() for d in scenario_data]

    ax.set_yticks(y_positions)
    ax.set_yticklabels(neg_labels, fontsize=13, color="#111111")

    # Right axis: positive pole (text before "|") — black
    ax_r = ax.twinx()
    ax_r.set_ylim(ax.get_ylim())
    ax_r.set_yticks(y_positions)
    ax_r.set_yticklabels(pos_labels, fontsize=13, color="#111111")
    ax_r.tick_params(axis="y", length=0)
    for sp in ax_r.spines.values():
        sp.set_visible(False)

    # Circled symbols just above the topmost label on each axis
    # get_yaxis_transform: x in axes fraction, y in data coords
    ax.text(0, n - 0.35, "⊖",
            transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=24, color="#7A1010", fontweight="bold", clip_on=False)
    ax_r.text(1, n - 0.35, "⊕",
              transform=ax_r.get_yaxis_transform(), ha="left", va="bottom",
              fontsize=24, color="#1B6B1B", fontweight="bold", clip_on=False)

    ax.set_xlabel("Average Mean Δ across 6 models", fontsize=13, labelpad=8)
    ax.tick_params(axis="x", labelsize=12)
    ax.set_title("All 25 Scenarios: Average Prediction Shift (sorted)", fontsize=15, fontweight="bold", pad=16)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(axis="both", length=0)

    legend_handles = [
        mpatches.Patch(facecolor=CATEGORY_COLORS[cat], edgecolor="#888888", linewidth=0.4, label=cat)
        for cat in CATEGORIES
    ]
    legend_handles += [
        Line2D([0], [0], color="#888888", linewidth=1.8, alpha=0.6,
               solid_capstyle="round", label="Min–max range across models"),
        Line2D([0], [0], marker="D", linestyle="None", markersize=7,
               color="#888888", label="Cross-model average"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        frameon=True,
        framealpha=0.95,
        fontsize=12,
        edgecolor="#cccccc",
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
    parser = argparse.ArgumentParser(description="Create two scenario-level paper charts.")
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=Path("output/evaluation"),
    )
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=None,
        help="Explicit model_comparison directory. If omitted, the latest one is used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/evaluation/eval_charts"),
    )
    args = parser.parse_args()

    comparison_dir = args.comparison_dir or _discover_latest_comparison(args.evaluation_root)
    print(f"Using comparison dir: {comparison_dir}")

    rows = _load_scenario_data(comparison_dir)
    models = _get_model_columns(rows)
    print(f"Models found: {models}")

    plot_category_bars(rows, models, args.output_dir / "mean_abs_shift_by_category.png")
    plot_scenario_shifts(rows, models, args.output_dir / "scenario_shift_sorted.png")


if __name__ == "__main__":
    main()
