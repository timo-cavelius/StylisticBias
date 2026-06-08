#!/usr/bin/env python3
"""Per-model mean Δ across scenarios, grouped by category.

Moved from root `src/model_mean_across_scenarios.py`.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EVALUATION_ROOT = Path("output/evaluation")
OUTPUT_DIR      = Path("output/evaluation/eval_charts")

# Scenario → category (same mapping as create_scenario_charts.py)
SCENARIO_CATEGORIES: dict[str, str] = {
    "Competent | Incompetent":        "Personality",
    "Likeable | Unlikeable":          "Personality",
    "Intelligent | Unintelligent":    "Personality",
    "Responsible | Irresponsible":    "Personality",
    "Open-minded | Closed-minded":    "Personality",
    "Conscientious | Careless":       "Personality",
    "Extraverted | Introverted":      "Personality",
    "Emotionally stable | Anxious":   "Personality",
    "Confident | Insecure":           "Personality",
    "Curious | Indifferent":          "Personality",
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
    "Home owner | Renter":            "Socioeconomic",
    "Educated | Uneducated":          "Socioeconomic",
    "Attractive | Unattractive":      "Socioeconomic",
    "Stylish | Unstylish":            "Socioeconomic",
    "Wealthy | Poor":                 "Socioeconomic",
}

CATEGORY_ORDER = ["Personality", "Interpersonal", "Behavioral", "Socioeconomic"]

CATEGORY_BG = {
    "Personality":  "#E8F3FA",
    "Interpersonal":"#E9F5E9",
    "Behavioral":   "#FEF0E6",
    "Socioeconomic":"#FAE9EB",
}

CATEGORY_LABEL_COLOR = {
    "Personality":  "#6aaed6",
    "Interpersonal":"#74c476",
    "Behavioral":   "#fd8d3c",
    "Socioeconomic":"#e8435a",
}

MODEL_DISPLAY = {
    "gemma3":    "Gemma-3",
    "gemma4":    "Gemma-4",
    "internvl":  "InternVL3",
    "llava_next": "LLaVA-1.6",
    "pixtral":   "Pixtral",
    "qwen3":     "Qwen3-VL",
}

# Each model: color + linestyle + marker matching the reference figure
MODEL_STYLES: dict[str, dict] = {
    "gemma3":    {"color": "#3A74B0", "ls": "-",   "marker": "o", "ms": 6.5, "lw": 1.8},
    "gemma4":    {"color": "#C84B4B", "ls": "--",  "marker": "s", "ms": 6.0, "lw": 1.8},
    "internvl":  {"color": "#3A9A3A", "ls": "--",  "marker": "^", "ms": 6.5, "lw": 1.8},
    "llava_next":{"color": "#8B55B5", "ls": ":",   "marker": "D", "ms": 6.0, "lw": 1.8},
    "pixtral":   {"color": "#D97820", "ls": "-",   "marker": "o", "ms": 6.5, "lw": 1.8},
    "qwen3":     {"color": "#28A8C0", "ls": "-.",  "marker": "o", "ms": 6.0, "lw": 1.8},
}


def _latest_comparison_dir(root: Path) -> Path:
    dirs = sorted(d for d in root.iterdir()
                  if d.is_dir() and d.name.startswith("model_comparison"))
    if not dirs:
        raise FileNotFoundError(f"No model_comparison folder found in {root}")
    return dirs[-1]


def _load(csv_path: Path) -> tuple[list[str], list[str], dict[str, list[float]]]:
    """Return (scenario_labels, short_labels, {model: [mean_delta, ...]})."""
    rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"Empty CSV: {csv_path}")

    models = [k.replace("_mean_delta", "")
              for k in rows[0] if k.endswith("_mean_delta")]

    labels: list[str] = []
    short:  list[str] = []
    data:   dict[str, list[float]] = {m: [] for m in models}

    for row in rows:
        lbl = row["scenario_label"].strip()
        labels.append(lbl)
        short.append(lbl.split("|")[0].strip())
        for m in models:
            try:
                data[m].append(float(row[f"{m}_mean_delta"]))
            except (KeyError, ValueError):
                data[m].append(0.0)

    return labels, short, data


def _category_spans(labels: list[str]) -> list[tuple[str, int, int]]:
    """Return [(category, x_start, x_end), ...] based on label order."""
    spans: list[tuple[str, int, int]] = []
    cur_cat = SCENARIO_CATEGORIES.get(labels[0], "")
    start   = 0
    for i, lbl in enumerate(labels[1:], 1):
        cat = SCENARIO_CATEGORIES.get(lbl, "")
        if cat != cur_cat:
            spans.append((cur_cat, start, i - 1))
            cur_cat, start = cat, i
    spans.append((cur_cat, start, len(labels) - 1))
    return spans


def plot(
    labels: list[str],
    short:  list[str],
    data:   dict[str, list[float]],
    output_path: Path,
) -> None:
    n       = len(labels)
    x       = np.arange(n)
    models  = [m for m in MODEL_STYLES if m in data]   # keep defined order
    spans   = _category_spans(labels)

    all_vals = [v for vals in data.values() for v in vals]
    yabs     = max(abs(v) for v in all_vals) * 1.25
    y_lo, y_hi = -yabs, yabs

    fig, ax = plt.subplots(figsize=(16, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # ---- category background bands ----
    for cat, xs, xe in spans:
        ax.axvspan(xs - 0.5, xe + 0.5,
                   facecolor=CATEGORY_BG.get(cat, "#F5F5F5"),
                   alpha=1.0, zorder=0)
        # italic category label near the bottom of the band
        ax.text(
            (xs + xe) / 2,
            y_lo + (y_hi - y_lo) * 0.04,
            cat,
            ha="center", va="bottom",
            fontsize=11, style="italic",
            color=CATEGORY_LABEL_COLOR.get(cat, "#888888"),
            zorder=1,
        )

    # ---- zero line ----
    ax.axhline(0, color="#888888", linewidth=0.9, linestyle="-", zorder=2)

    # ---- one line per model ----
    for model in models:
        st = MODEL_STYLES[model]
        vals = data[model]
        ax.plot(
            x, vals,
            color=st["color"],
            linestyle=st["ls"],
            linewidth=st["lw"],
            marker=st["marker"],
            markersize=st["ms"],
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=MODEL_DISPLAY.get(model, model),
            zorder=3,
        )

    # ---- category boundary lines ----
    for _, xs, xe in spans[:-1]:
        ax.axvline(xe + 0.5, color="#cccccc", linewidth=0.8, zorder=2)

    # ---- axes styling ----
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=9.5)
    ax.set_ylabel("Mean Δ", fontsize=12, fontweight="bold")
    ax.yaxis.grid(True, color="white", linewidth=0.8, zorder=1)
    ax.set_axisbelow(False)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(axis="both", length=0)

    # ---- category labels above the x-axis (top-level grouping) ----
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks([(xs + xe) / 2 for _, xs, xe in spans])
    ax2.set_xticklabels(
        [cat for cat, _, _ in spans],
        fontsize=11, fontweight="bold",
    )
    for spine in ("top", "right", "left", "bottom"):
        ax2.spines[spine].set_visible(False)
    ax2.tick_params(axis="x", length=0)
    for lbl, (cat, _, _) in zip(ax2.get_xticklabels(), spans):
        lbl.set_color(CATEGORY_LABEL_COLOR.get(cat, "#888888"))

    # ---- title ----
    ax.set_title(
        "Per-model mean Δ across scenarios (grouped by category)",
        fontsize=14, fontweight="bold", pad=28,
    )

    # ---- legend ----
    handles = [
        plt.Line2D(
            [0], [0],
            color=MODEL_STYLES[m]["color"],
            linestyle=MODEL_STYLES[m]["ls"],
            linewidth=1.8,
            marker=MODEL_STYLES[m]["marker"],
            markersize=MODEL_STYLES[m]["ms"],
            markeredgecolor="white", markeredgewidth=0.6,
            label=MODEL_DISPLAY.get(m, m),
        )
        for m in models
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.32),
        ncol=len(models),
        frameon=False,
        fontsize=10,
        handlelength=2.0,
        columnspacing=1.4,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    comp_dir = _latest_comparison_dir(EVALUATION_ROOT)
    print(f"Using: {comp_dir}")
    csv_path = comp_dir / "scenario_comparison.csv"
    labels, short, data = _load(csv_path)

    # Re-order scenarios so all scenarios of the same category are adjacent,
    # following CATEGORY_ORDER. Within each category the CSV order is preserved.
    def _cat_rank(lbl: str) -> int:
        cat = SCENARIO_CATEGORIES.get(lbl, "")
        try:
            return CATEGORY_ORDER.index(cat)
        except ValueError:
            return len(CATEGORY_ORDER)

    order = sorted(range(len(labels)), key=lambda i: _cat_rank(labels[i]))
    labels = [labels[i] for i in order]
    short  = [short[i]  for i in order]
    data   = {m: [vals[i] for i in order] for m, vals in data.items()}

    print(f"Scenarios: {len(labels)}  |  Models: {list(data)}")
    plot(labels, short, data, OUTPUT_DIR / "model_mean_across_scenarios.png")


if __name__ == "__main__":
    main()
