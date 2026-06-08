#!/usr/bin/env python3
"""Scenario sensitivity scatter plot.

Moved from root `src/scenario_sensitivity.py`.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EVALUATION_ROOT = Path("output/evaluation")
OUTPUT_DIR      = Path("output/evaluation/eval_charts")

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

CATEGORY_ORDER = [
    "Personality & Social",
    "Interpersonal",
    "Behavioral",
    "Socioeconomic & App.",
]

CATEGORY_COLORS = {
    "Personality & Social": "#6aaed6",
    "Interpersonal":        "#74c476",
    "Behavioral":           "#fd8d3c",
    "Socioeconomic & App.": "#e8435a",
}


def _latest_comparison_dir(root: Path) -> Path:
    dirs = sorted(d for d in root.iterdir()
                  if d.is_dir() and d.name.startswith("model_comparison"))
    if not dirs:
        raise FileNotFoundError(f"No model_comparison folder in {root}")
    return dirs[-1]


def _load(csv_path: Path) -> list[dict]:
    with csv_path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _compute_points(rows: list[dict]) -> list[dict]:
    """For each scenario compute signed mean Δ and mean |Δ| across models."""
    points = []
    for row in rows:
        label = row["scenario_label"].strip()
        model_vals = []
        for k, v in row.items():
            if k.endswith("_mean_delta"):
                try:
                    model_vals.append(float(v))
                except ValueError:
                    pass
        if not model_vals:
            continue
        signed_mean = float(np.mean(model_vals))
        abs_mean    = float(np.mean([abs(v) for v in model_vals]))
        points.append({
            "label":       label,
            "category":    SCENARIO_CATEGORIES.get(label, "Personality & Social"),
            "signed_mean": signed_mean,
            "abs_mean":    abs_mean,
        })
    return points


def _should_label(pt: dict, all_pts: list[dict], top_n: int = 8) -> bool:
    """Label the top-N outliers by distance from origin."""
    dist = lambda p: p["signed_mean"] ** 2 + p["abs_mean"] ** 2
    ranked = sorted(all_pts, key=dist, reverse=True)
    return pt in ranked[:top_n]


def plot(points: list[dict], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # ---- Grid ----
    ax.grid(color="#E0E0E0", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    # ---- Vertical dashed line at x=0 ----
    ax.axvline(0, color="#888888", linewidth=1.0, linestyle="--", zorder=1)

    # ---- Scatter dots ----
    for cat in CATEGORY_ORDER:
        cat_pts = [p for p in points if p["category"] == cat]
        ax.scatter(
            [p["signed_mean"] for p in cat_pts],
            [p["abs_mean"]    for p in cat_pts],
            color=CATEGORY_COLORS[cat],
            s=150, alpha=0.82,
            edgecolors="white", linewidths=0.8,
            zorder=3, label=cat,
        )

    # ---- Labels for outliers (auto-adjusted to avoid overlap) ----
    texts = []
    for pt in points:
        if not _should_label(pt, points):
            continue
        short = pt["label"].replace(" | ", "\n")

        # Place "Responsible | Irresponsible" manually to the left
        if pt["label"] == "Responsible | Irresponsible":
            ax.annotate(
                short,
                xy=(pt["signed_mean"], pt["abs_mean"]),
                xytext=(pt["signed_mean"] - 0.03, pt["abs_mean"] - 0.018),
                fontsize=11.5, color="#333333",
                ha="right", va="center", zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.85),
                arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=0.8),
            )
            continue

        t = ax.text(
            pt["signed_mean"],
            pt["abs_mean"],
            short,
            fontsize=11.5, color="#333333",
            va="bottom", ha="left",
            zorder=5,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor="none", alpha=0.85),
        )
        texts.append(t)

    adjust_text(
        texts,
        x=[pt["signed_mean"] for pt in points if _should_label(pt, points)],
        y=[pt["abs_mean"]    for pt in points if _should_label(pt, points)],
        ax=ax,
        arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=0.8),
        expand=(1.8, 2.2),
        force_text=(1.0, 1.2),
        force_points=(0.8, 1.0),
    )

    # ---- Axes ----
    all_x = [p["signed_mean"] for p in points]
    all_y = [p["abs_mean"]    for p in points]
    xpad  = (max(all_x) - min(all_x)) * 0.12
    ypad  = (max(all_y) - min(all_y)) * 0.12
    ax.set_xlim(min(all_x) - xpad, 0.25)
    ax.set_ylim(max(0, min(all_y) - ypad), 0.25)

    ax.set_xlabel("Average signed shift across models, Δ", fontsize=13.5, labelpad=14)
    ax.set_ylabel("Mean absolute shift across models, |Δ|", fontsize=13.5, labelpad=14)
    ax.tick_params(axis="both", labelsize=12, length=0)

    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")

    # ---- Title ----
    ax.set_title(
        "Scenario sensitivity is concentrated in status-related judgments",
        fontsize=15, fontweight="bold", pad=28,
    )

    # ---- Legend ----
    ax.legend(
        loc="upper left",
        fontsize=11,
        frameon=True, framealpha=0.92,
        edgecolor="#cccccc",
        markerscale=1.1,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    comp_dir = _latest_comparison_dir(EVALUATION_ROOT)
    print(f"Using: {comp_dir.name}")
    rows   = _load(comp_dir / "scenario_comparison.csv")
    points = _compute_points(rows)
    print(f"Scenarios: {len(points)}")
    plot(points, OUTPUT_DIR / "scenario_sensitivity.png")


if __name__ == "__main__":
    main()
