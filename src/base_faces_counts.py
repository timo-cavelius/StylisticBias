#!/usr/bin/env python3
"""Base face demographic distribution — 4 vertical bar charts.

Counts all 500 base face metadata files directly from output/images/.
One chart per demographic attribute (Gender, Ethnicity, Body Type, Age),
stacked vertically in a single figure.

Output: output/evaluation/eval_charts/base_faces_counts.png

Usage:
  python3 src/base_faces_counts.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGES_DIR = Path("output/images")
OUTPUT_DIR = Path("output/evaluation/eval_charts")

CATEGORIES = [
    {
        "key":    "gender",
        "title":  "Base Face Gender Count",
        "order":  ["female", "male"],
        "labels": {"female": "female", "male": "male"},
    },
    {
        "key":    "body_type",
        "title":  "Base Face Body Type Count",
        "order":  ["normal", "obese", "thin"],
        "labels": {"normal": "normal", "obese": "obese", "thin": "thin"},
    },
    {
        "key":    "ethnicity",
        "title":  "Base Face Ethnicity Count",
        "order":  ["African", "Asian", "European", "Latino", "Middle Eastern"],
        "labels": {
            "African":        "African",
            "Asian":          "Asian",
            "European":       "European",
            "Latino":         "Latino",
            "Middle Eastern": "Middle Eastern",
        },
    },
    {
        "key":    "age",
        "title":  "Base Face Age Count",
        "order":  ["elderly", "middle-aged adult", "young adult"],
        "labels": {
            "elderly":            "elderly",
            "middle-aged adult":  "middle-aged adult",
            "young adult":        "young adult",
        },
    },
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_counts() -> dict[str, Counter]:
    counters: dict[str, Counter] = {cfg["key"]: Counter() for cfg in CATEGORIES}
    total = 0
    for path in IMAGES_DIR.glob("*_metadata.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        c = data.get("characteristics", {})
        for key in counters:
            val = c.get(key, "").strip()
            if not val and key == "body_type":
                val = "normal"
            if val:
                counters[key][val] += 1
        total += 1
    print(f"Loaded {total} base face metadata files.")
    return counters


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(counters: dict[str, Counter], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    for ax, cfg in zip(axes.flat, CATEGORIES):
        key    = cfg["key"]
        order  = cfg["order"]
        labels = cfg["labels"]
        data   = counters[key]

        vals      = [data.get(v, 0) for v in order]
        tick_lbls = [labels[v] for v in order]
        xs        = np.arange(len(order))

        bars = ax.bar(xs, vals, color="#6aaed6")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() / 2,
                    str(v),
                    ha="center", va="center",
                    fontsize=14, color="white", fontweight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels(tick_lbls, fontsize=13)
        ax.set_ylabel("count", fontsize=13)
        ax.tick_params(axis="y", labelsize=12)
        ax.set_title(cfg["title"], fontweight="bold", fontsize=15)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    counters = load_counts()
    for cfg in CATEGORIES:
        key = cfg["key"]
        print(f"  {cfg['title']}: {dict(counters[key])}")
    plot(counters, OUTPUT_DIR / "base_faces_counts.png")


if __name__ == "__main__":
    main()
