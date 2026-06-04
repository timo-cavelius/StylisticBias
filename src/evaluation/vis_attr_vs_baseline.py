#!/usr/bin/env python3
"""Visual attributes vs. demographic baseline — two-panel dot plot.

Left panel:  high-impact attributes  (|Δ| ≥ demographic baseline)
Right panel: below-baseline attributes (|Δ| < demographic baseline)

Demographic baseline = mean |Δ| from base face demographic categories
(age, gender, ethnicity, body_type) averaged across all models.

Output: output/evaluation/eval_charts/vis_attr_vs_baseline.png

Usage:
  python3 src/vis_attr_vs_baseline.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EVALUATION_ROOT = Path("output/evaluation")
OUTPUT_DIR      = Path("output/evaluation/eval_charts")

CAT_GROUP = {
    "fashion_style":       "FASHION",
    "lip_makeup_female":   "MAKEUP",
    "makeup_female":       "MAKEUP",
    "facial_hair_male":    "FACIAL HAIR",
    "hair_style":          "HAIR STYLE",
    "skin_irregularities": "SKIN",
    "eyewear":             "EYEWEAR",
    "tattoos":             "TATTOOS",
    "hair_length":         "HAIR LENGTH",
    "accessories":         "ACCESSORIES",
    "piercings":           "PIERCINGS",
    "hair_color":          "HAIR COLOR",
}

GROUP_ORDER = [
    "FASHION", "MAKEUP", "FACIAL HAIR", "HAIR STYLE", "SKIN",
    "EYEWEAR", "TATTOOS", "HAIR LENGTH", "ACCESSORIES", "PIERCINGS", "HAIR COLOR",
]

FEMALE_COLOR = "#C4134E"
MALE_COLOR   = "#1040A8"
LINE_COLOR   = "#AAAAAA"

# Vertical layout (data-coordinate units)
Y_STEP     = 0.75  # distance between variation rows
CAT_HEADER = 0.80  # vertical space consumed by a category header row
CAT_GAP    = 0.18  # extra gap between category blocks

# Physical scale: inches per data-coordinate unit, used only for figure height
INCHES_PER_UNIT = 0.30

BAND_COLORS = [
    "#FFF0F3", "#FFF7EE", "#F0FFF4", "#F0F4FF", "#FFFFF0",
    "#FFF0FF", "#F0FFFF", "#FFF5F0", "#F5F0FF", "#F0F5FF", "#FFF0F5",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _latest_comparison_dir(root: Path) -> Path:
    dirs = sorted(d for d in root.iterdir()
                  if d.is_dir() and d.name.startswith("model_comparison"))
    if not dirs:
        raise FileNotFoundError(f"No model_comparison folder in {root}")
    return dirs[-1]


def _discover_model_dirs(root: Path) -> list[Path]:
    return [e for e in sorted(root.iterdir())
            if e.is_dir() and (e / "variation_impact_summary.csv").exists()]


def _compute_baseline(comp_dir: Path) -> float:
    """Mean variation_strength of base-face demographic categories across all models."""
    path = comp_dir / "base_face_category_variation_strength.csv"
    if not path.exists():
        return 0.058
    vals: list[float] = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                vals.append(float(row["variation_strength"]))
            except (KeyError, ValueError):
                pass
    return float(np.mean(vals)) if vals else 0.058


def _load_variation_data(model_dirs: list[Path]) -> dict[str, dict[str, float]]:
    """Cross-model average mean Δ per gender per variation."""
    collector: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for d in model_dirs:
        within: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        with (d / "variation_impact_summary.csv").open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                within[row["gender"]][row["variation_name"]].append(float(row["mean_delta"]))
        for gender, vdata in within.items():
            for var, vals in vdata.items():
                collector[gender][var].append(float(np.mean(vals)))
    return {
        gender: {var: float(np.mean(vals)) for var, vals in vdata.items()}
        for gender, vdata in collector.items()
    }


# ---------------------------------------------------------------------------
# Build variation rows
# ---------------------------------------------------------------------------

def _build_rows(avg: dict[str, dict[str, float]]) -> list[dict]:
    female_data = avg.get("female", {})
    male_data   = avg.get("male",   {})
    all_vars    = set(female_data) | set(male_data)

    rows: list[dict] = []
    for var in all_vars:
        raw_cat, _, label = var.partition(":")
        group = CAT_GROUP.get(raw_cat, raw_cat.upper())
        fval  = female_data.get(var, np.nan)
        mval  = male_data.get(var,   np.nan)
        vals  = [v for v in (fval, mval) if np.isfinite(v)]
        abs_mean = float(np.mean([abs(v) for v in vals])) if vals else 0.0
        rows.append({
            "var": var, "group": group, "label": label.strip(),
            "female": fval, "male": mval, "abs_mean": abs_mean,
        })
    return rows


def _ordered(rows: list[dict]) -> list[dict]:
    def _rank(r: dict) -> tuple:
        try:
            gi = GROUP_ORDER.index(r["group"])
        except ValueError:
            gi = len(GROUP_ORDER)
        return (gi, -r["abs_mean"])
    return sorted(rows, key=_rank)


# ---------------------------------------------------------------------------
# Y-position layout
# ---------------------------------------------------------------------------

def _build_layout(rows: list[dict]) -> tuple[list[float], list[tuple], list[tuple]]:
    """
    Returns:
      var_y      : y position per variation row (parallel to rows)
      tick_items : [(y, label, is_header), ...] — for set_yticks
      bands      : [(y_top, y_bot, color_index), ...] — category background bands
    """
    var_y:      list[float]  = []
    tick_items: list[tuple]  = []
    bands:      list[tuple]  = []

    y        = 0.0
    prev_grp = None
    grp_idx  = -1
    grp_start: float | None = None

    def _close_band(grp_start, y_last, ci):
        if grp_start is not None:
            bands.append((grp_start + CAT_HEADER * 0.5, y_last - Y_STEP * 0.45, ci))

    for i, row in enumerate(rows):
        grp = row["group"]
        if grp != prev_grp:
            if prev_grp is not None:
                _close_band(grp_start, var_y[-1], grp_idx)
                y -= CAT_GAP
            grp_idx += 1
            # Header occupies CAT_HEADER units
            hdr_y = y - CAT_HEADER * 0.5
            tick_items.append((hdr_y, grp, True))
            y -= CAT_HEADER
            grp_start = hdr_y
            prev_grp  = grp

        tick_items.append((y, row["label"], False))
        var_y.append(y)
        y -= Y_STEP

    # Close last band
    if var_y:
        _close_band(grp_start, var_y[-1], grp_idx)

    return var_y, tick_items, bands


def _total_data_height(rows: list[dict]) -> float:
    n_rows   = len(rows)
    n_groups = len({r["group"] for r in rows})
    return n_rows * Y_STEP + n_groups * CAT_HEADER + max(0, n_groups - 1) * CAT_GAP


# ---------------------------------------------------------------------------
# Panel drawing
# ---------------------------------------------------------------------------

def _draw_panel(
    ax: plt.Axes,
    rows: list[dict],
    baseline: float,
    xlim: tuple[float, float],
    title: str,
    title_color: str,
) -> None:
    if not rows:
        ax.set_visible(False)
        return

    var_y, tick_items, bands = _build_layout(rows)

    y_max = max(t[0] for t in tick_items) + CAT_HEADER * 0.6
    y_min = min(var_y) - Y_STEP * 0.55

    # Standard orientation: y=0 at top, y=negative at bottom
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(*xlim)
    ax.set_facecolor("white")

    # ---- Category background bands ----
    for ci, (bt, bb, idx) in enumerate(bands):
        color = BAND_COLORS[idx % len(BAND_COLORS)]
        ax.axhspan(bb, bt, facecolor=color, alpha=0.90, zorder=0)

    # ---- Grid + reference lines ----
    ax.xaxis.grid(True, color="#EEEEEE", linewidth=0.7, zorder=1)
    ax.axvline(0,         color="#666666", linewidth=0.8,            zorder=2)
    ax.axvline( baseline, color="#999999", linewidth=1.0, linestyle="--", zorder=2)
    ax.axvline(-baseline, color="#999999", linewidth=1.0, linestyle="--", zorder=2)

    # ---- Dots + connecting line ----
    for row, yp in zip(rows, var_y):
        fval = row["female"]
        mval = row["male"]
        # Lines from each diamond to the zero axis
        if np.isfinite(fval):
            ax.plot([0, fval], [yp - 0.17, yp - 0.17],
                    color=FEMALE_COLOR, linewidth=1.3, zorder=3,
                    solid_capstyle="round")
        if np.isfinite(mval):
            ax.plot([0, mval], [yp + 0.17, yp + 0.17],
                    color=MALE_COLOR, linewidth=1.3, zorder=3,
                    solid_capstyle="round")
        if np.isfinite(fval):
            ax.scatter([fval], [yp - 0.17], s=46, marker="D",
                       color=FEMALE_COLOR, edgecolors="white", linewidths=0.55, zorder=4)
        if np.isfinite(mval):
            ax.scatter([mval], [yp + 0.17], s=46, marker="D",
                       color=MALE_COLOR, edgecolors="white", linewidths=0.55, zorder=4)

    # ---- Y ticks ----
    yticks = [y for y, _, _ in tick_items]
    ylabels = [lbl for _, lbl, _ in tick_items]
    is_hdr  = [is_h for _, _, is_h in tick_items]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8.5)
    for lbl, hdr in zip(ax.get_yticklabels(), is_hdr):
        if hdr:
            lbl.set_fontweight("bold")
            lbl.set_color(title_color)

    # ---- Text labels over dots ----
    for row, yp in zip(rows, var_y):
        fval = row["female"]
        mval = row["male"]
        if np.isfinite(fval):
            ax.text(fval, yp - 0.42, f"{fval:+.3f}", ha="center", va="top",
                    fontsize=7.4, color=FEMALE_COLOR, fontweight="bold")
        if np.isfinite(mval):
            ax.text(mval, yp + 0.42, f"{mval:+.3f}", ha="center", va="bottom",
                    fontsize=7.4, color=MALE_COLOR, fontweight="bold")

    # ---- Axis decorations ----
    ax.set_xlabel("Mean Δ", fontsize=9.5)
    ax.tick_params(axis="x", labelsize=8.5, length=0)
    ax.tick_params(axis="y", length=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")

    ax.text(0.0, 1.04, title, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=11.2, fontweight="bold", color=title_color)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    comp_dir = _latest_comparison_dir(EVALUATION_ROOT)
    model_dirs = _discover_model_dirs(EVALUATION_ROOT)
    baseline = _compute_baseline(comp_dir)
    print(f"Using comparison dir: {comp_dir.name}")
    print(f"Baseline |Δ| = {baseline:.4f}")

    avg = _load_variation_data(model_dirs)
    rows = _ordered(_build_rows(avg))

    # Split into above / below baseline by the larger of gender means
    above: list[dict] = []
    below: list[dict] = []
    for r in rows:
        score = r["abs_mean"]
        (above if score >= baseline else below).append(r)

    fig, axes = plt.subplots(1, 2, figsize=(16, max(8.5, 0.34 * len(rows) + 2.8)), sharey=False)
    fig.patch.set_facecolor("white")

    _draw_panel(
        axes[0],
        above,
        baseline,
        xlim=(min([min(r["female"], r["male"]) for r in above] + [0]) * 1.15, max([max(r["female"], r["male"]) for r in above] + [baseline]) * 1.25),
        title="High-impact attributes",
        title_color="#C4134E",
    )
    _draw_panel(
        axes[1],
        below,
        baseline,
        xlim=(min([min(r["female"], r["male"]) for r in below] + [0]) * 1.15, max([max(r["female"], r["male"]) for r in below] + [baseline]) * 1.25),
        title="Below-baseline attributes",
        title_color="#1040A8",
    )

    axes[0].set_ylabel("Variation", fontsize=9.5)
    axes[1].set_ylabel("")

    fig.suptitle(
        "Visual attributes compared against demographic baseline",
        fontsize=14.5,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "vis_attr_vs_baseline.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
