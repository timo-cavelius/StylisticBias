#!/usr/bin/env python3
"""Gemma family comparison: scaling changes magnitude more than semantic structure.

Left panel : radar chart — mean |Δ| per scenario category (Gemma-3 vs Gemma-4).
Right panel: scatter plot — Gemma-3 Δ vs Gemma-4 Δ per scenario, coloured by
             scenario category, with regression line + stats box.

Output: output/evaluation/eval_charts/gemma_family.png

Usage:
  python3 src/gemma_family.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats
from scipy.stats import kendalltau, pearsonr, spearmanr
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

# Clockwise from top for radar
RADAR_CATEGORIES = [
    "Personality & Social",
    "Socioeconomic & App.",
    "Behavioral",
    "Interpersonal",
]

RADAR_DISPLAY = {
    "Personality & Social": "Personality\n& Social",
    "Interpersonal":        "Interpersonal",
    "Behavioral":           "Behavioral",
    "Socioeconomic & App.": "Socioeconomic\n& App.",
}

CATEGORY_COLORS = {
    "Personality & Social": "#6aaed6",
    "Interpersonal":        "#74c476",
    "Behavioral":           "#fd8d3c",
    "Socioeconomic & App.": "#e8435a",
}

GEMMA3_COLOR = "#3A80C0"
GEMMA4_COLOR = "#4CAF50"

# Scenarios to label as outliers in scatter
OUTLIER_LABELS = {
    "Stylish | Unstylish",
    "Wealthy | Poor",
    "Confident | Insecure",
    "Attractive | Unattractive",
    "Educated | Uneducated",
}


# ---------------------------------------------------------------------------
# Data loading
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


def _parse_rows(rows: list[dict]) -> list[dict]:
    result = []
    for row in rows:
        label = row["scenario_label"].strip()
        try:
            g3 = float(row["gemma3_mean_delta"])
            g4 = float(row["gemma4_mean_delta"])
        except (ValueError, KeyError):
            continue
        result.append({
            "label":    label,
            "category": SCENARIO_CATEGORIES.get(label, "Personality & Social"),
            "gemma3":   g3,
            "gemma4":   g4,
        })
    return result


# ---------------------------------------------------------------------------
# Radar helpers
# ---------------------------------------------------------------------------

def _compute_radar_values(data: list[dict]) -> dict[str, dict[str, float]]:
    """Mean |Δ| per scenario category for gemma3 and gemma4."""
    from collections import defaultdict
    g3: dict[str, list[float]] = defaultdict(list)
    g4: dict[str, list[float]] = defaultdict(list)
    for d in data:
        g3[d["category"]].append(abs(d["gemma3"]))
        g4[d["category"]].append(abs(d["gemma4"]))
    return {
        "gemma3": {cat: float(np.mean(g3[cat])) if g3[cat] else 0.0 for cat in RADAR_CATEGORIES},
        "gemma4": {cat: float(np.mean(g4[cat])) if g4[cat] else 0.0 for cat in RADAR_CATEGORIES},
    }


def _label_alignment(angle_from_top_cw: float) -> tuple[str, str]:
    a = angle_from_top_cw % (2 * np.pi)
    eps = 0.25
    if a < eps or a > 2 * np.pi - eps:
        return "center", "bottom"
    if abs(a - np.pi) < eps:
        return "center", "top"
    ha = "left"   if np.sin(a) > 0 else "right"
    va = "bottom" if np.cos(a) > 0 else "top"
    return ha, va


def _draw_radar(ax: plt.Axes, radar_vals: dict[str, dict[str, float]]) -> None:
    N = len(RADAR_CATEGORIES)
    angles = [np.pi / 2 - 2 * np.pi * i / N for i in range(N)]
    angles_closed = angles + [angles[0]]

    all_vals = [v for m in radar_vals.values() for v in m.values()]
    raw_max  = max(all_vals)
    max_val  = np.ceil(raw_max / 0.01) * 0.01
    ring_step = max_val / 4
    ring_vals = np.arange(ring_step, max_val + 1e-9, ring_step)

    theta_full = np.linspace(0, 2 * np.pi, 360)

    # Rings
    for rv in ring_vals:
        is_outer = abs(rv - ring_vals[-1]) < 1e-9
        ax.plot(theta_full, [rv] * 360,
                color="#BBBBBB" if is_outer else "#DDDDDD",
                linewidth=1.0 if is_outer else 0.6,
                zorder=1)

    # Spokes
    for angle in angles:
        ax.plot([angle, angle], [0, max_val],
                color="#CCCCCC", linewidth=0.7, zorder=1)

    # Ring labels along top-right spoke
    label_angle = angles[0]
    for rv in ring_vals[:-1]:
        ax.text(label_angle, rv, f"{rv:.3f}",
                ha="center", va="bottom",
                fontsize=14, color="#AAAAAA", zorder=2)

    # Polygons
    models = [("gemma3", GEMMA3_COLOR, "Gemma-3"),
              ("gemma4", GEMMA4_COLOR, "Gemma-4")]
    for model_key, color, _ in models:
        vals        = [radar_vals[model_key].get(c, 0.0) for c in RADAR_CATEGORIES]
        vals_closed = vals + [vals[0]]
        ax.fill(angles_closed, vals_closed, color=color, alpha=0.15, zorder=3)
        ax.plot(angles_closed, vals_closed, color=color, linewidth=2.4, zorder=4,
                solid_capstyle="round")
        ax.scatter(angles, vals, color=color, s=55, zorder=5,
                   edgecolors="white", linewidths=0.8)

    # Category labels
    label_r = max_val * 1.05
    for angle, cat in zip(angles, RADAR_CATEGORIES):
        cw_angle = np.pi / 2 - angle
        ha, va   = _label_alignment(cw_angle)
        ax.text(angle, label_r,
                RADAR_DISPLAY.get(cat, cat),
                ha=ha, va=va,
                fontsize=16, fontweight="bold",
                color="#333333", zorder=6,
                multialignment="center")

    ax.set_ylim(0, max_val * 1.12)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines["polar"].set_visible(False)
    ax.grid(False)


# ---------------------------------------------------------------------------
# Scatter helpers
# ---------------------------------------------------------------------------

def _draw_scatter(ax: plt.Axes, data: list[dict]) -> None:
    x_all = [d["gemma3"] for d in data]
    y_all = [d["gemma4"]  for d in data]

    # Identity line
    lim_min = min(min(x_all), min(y_all)) * 1.1
    lim_max = max(max(x_all), max(y_all)) * 1.1
    ax.plot([lim_min, lim_max], [lim_min, lim_max],
            color="#AAAAAA", linewidth=1.2, linestyle="--", zorder=1,
            label="y = x (identity)")

    # Scatter dots per category
    for cat in ["Personality & Social", "Interpersonal", "Behavioral", "Socioeconomic & App."]:
        pts = [d for d in data if d["category"] == cat]
        ax.scatter([d["gemma3"] for d in pts],
                   [d["gemma4"]  for d in pts],
                   color=CATEGORY_COLORS[cat],
                   s=170, alpha=0.88,
                   edgecolors="white", linewidths=0.9,
                   zorder=3, label=cat)

    # Regression line
    x_arr = np.array(x_all)
    y_arr = np.array(y_all)
    slope, intercept, r_val, _, _ = stats.linregress(x_arr, y_arr)
    x_reg = np.linspace(min(x_arr), max(x_arr), 200)
    ax.plot(x_reg, slope * x_reg + intercept,
            color="#444444", linewidth=2.0, linestyle="-", zorder=2)

    r_sp, _ = stats.spearmanr(x_arr, y_arr)
    print(f"\nCaption stats: Pearson r={r_val:.2f}, Spearman ρ={r_sp:.2f}, slope={slope:.2f}")

    # Outlier labels — collected then auto-adjusted to avoid dot overlap
    outlier_pts = [d for d in data if d["label"] in OUTLIER_LABELS]
    texts = []
    for d in outlier_pts:
        short = d["label"].replace(" | ", "\n")

        # Place "Educated | Uneducated" manually to avoid bubble overlap
        if d["label"] == "Educated | Uneducated":
            ax.annotate(
                short,
                xy=(d["gemma3"], d["gemma4"]),
                xytext=(d["gemma3"] - 0.04, d["gemma4"] + 0.03),
                fontsize=13.5, color="#333333",
                ha="right", va="bottom", zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.9),
                arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=0.9),
            )
            continue

        t = ax.text(d["gemma3"], d["gemma4"],
                    short,
                    fontsize=13.5, color="#333333",
                    va="bottom", ha="left", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                              edgecolor="none", alpha=0.9))
        texts.append(t)

    adjust_text(
        texts,
        x=[d["gemma3"] for d in outlier_pts if d["label"] != "Educated | Uneducated"],
        y=[d["gemma4"]  for d in outlier_pts if d["label"] != "Educated | Uneducated"],
        ax=ax,
        arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=0.9),
        expand=(1.5, 1.8),
        force_text=(0.6, 0.9),
    )

    ax.axvline(0, color="#CCCCCC", linewidth=0.8, zorder=0)
    ax.axhline(0, color="#CCCCCC", linewidth=0.8, zorder=0)
    ax.set_xlabel("Gemma-3 mean Δ per scenario", fontsize=16)
    ax.set_ylabel("Gemma-4 mean Δ per scenario", fontsize=16)
    ax.tick_params(axis="both", labelsize=14, length=0)
    ax.grid(color="#EEEEEE", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")


# ---------------------------------------------------------------------------
# Correlation statistics + CSV export
# ---------------------------------------------------------------------------

def _fisher_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """95% CI for a correlation via Fisher's z-transform."""
    if n <= 3 or abs(r) >= 1.0:
        return float("nan"), float("nan")
    z    = np.arctanh(r)
    se   = 1.0 / np.sqrt(n - 3)
    z_c  = stats.norm.ppf(1 - alpha / 2)
    return float(np.tanh(z - z_c * se)), float(np.tanh(z + z_c * se))


def _bootstrap_corr(x: np.ndarray, y: np.ndarray,
                    n_boot: int = 2000, seed: int = 42
                    ) -> tuple[tuple[float, float], tuple[float, float]]:
    """Bootstrap 95% CI for Pearson r and Spearman ρ (percentile method)."""
    rng = np.random.default_rng(seed)
    n   = len(x)
    pr_boot, sp_boot = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb, yb = x[idx], y[idx]
        if np.std(xb) == 0 or np.std(yb) == 0:
            continue
        pr_boot.append(float(pearsonr(xb, yb)[0]))
        sp_boot.append(float(spearmanr(xb, yb)[0]))
    return (
        (float(np.percentile(pr_boot, 2.5)), float(np.percentile(pr_boot, 97.5))),
        (float(np.percentile(sp_boot, 2.5)), float(np.percentile(sp_boot, 97.5))),
    )


def _corr_block(x: np.ndarray, y: np.ndarray, label: str) -> list[list]:
    """Return CSV rows for one group (overall or per-category)."""
    n = len(x)
    if n < 4:
        return [[label, "n<4 — skipped", "", "", "", "", "", "", ""]]

    r_p,  p_p  = pearsonr(x, y)
    r_sp, p_sp = spearmanr(x, y)
    tau,  p_tau = kendalltau(x, y)

    slope, intercept, _, _, slope_se = stats.linregress(x, y)

    ci_p_fisher  = _fisher_ci(r_p,  n)
    ci_sp_fisher = _fisher_ci(r_sp, n)
    (ci_p_boot, ci_sp_boot) = _bootstrap_corr(x, y)

    rows = [
        [label, "n", str(n), "", "", "", "", "", ""],
        # Pearson
        [label, "Pearson_r",      f"{r_p:.4f}",
         f"{ci_p_fisher[0]:.4f}", f"{ci_p_fisher[1]:.4f}", "Fisher z",
         f"{ci_p_boot[0]:.4f}",   f"{ci_p_boot[1]:.4f}",  f"{p_p:.4f}"],
        # Spearman
        [label, "Spearman_rho",   f"{r_sp:.4f}",
         f"{ci_sp_fisher[0]:.4f}", f"{ci_sp_fisher[1]:.4f}", "Fisher z",
         f"{ci_sp_boot[0]:.4f}",   f"{ci_sp_boot[1]:.4f}",  f"{p_sp:.4f}"],
        # Kendall
        [label, "Kendall_tau",    f"{tau:.4f}",
         "", "", "",
         "", "", f"{p_tau:.4f}"],
        # Regression slope
        [label, "OLS_slope",      f"{slope:.4f}",
         f"{slope - 1.96*slope_se:.4f}", f"{slope + 1.96*slope_se:.4f}", "±1.96·SE",
         "", "", ""],
        [label, "OLS_intercept",  f"{intercept:.4f}", "", "", "", "", "", ""],
    ]
    return rows


def save_correlation_csv(data: list[dict], output_path: Path) -> None:
    header = ["group", "statistic", "value",
              "ci_lower", "ci_upper", "ci_method",
              "boot_ci_lower", "boot_ci_upper", "p_value"]

    x_all = np.array([d["gemma3"] for d in data])
    y_all = np.array([d["gemma4"] for d in data])

    all_rows = [header]
    all_rows += _corr_block(x_all, y_all, "Overall (n=25)")

    for cat in ["Personality & Social", "Interpersonal", "Behavioral", "Socioeconomic & App."]:
        pts = [d for d in data if d["category"] == cat]
        xc  = np.array([d["gemma3"] for d in pts])
        yc  = np.array([d["gemma4"] for d in pts])
        all_rows += _corr_block(xc, yc, cat)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(all_rows)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def main() -> None:
    comp_dir = _latest_comparison_dir(EVALUATION_ROOT)
    print(f"Using: {comp_dir.name}")

    rows = _load(comp_dir / "scenario_comparison.csv")
    data = _parse_rows(rows)
    print(f"Scenarios: {len(data)}")

    radar_vals = _compute_radar_values(data)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Figure 1: radar ---
    fig_r = plt.figure(figsize=(10, 8), facecolor="white")
    ax_radar = fig_r.add_subplot(111, polar=True)
    ax_radar.set_facecolor("white")
    _draw_radar(ax_radar, radar_vals)

    fig_r.text(
        0.5, 0.96,
        "same semantic shape",
        ha="center", va="bottom",
        fontsize=18, color="#666666", style="italic",
    )

    radar_handles = [
        Line2D([0], [0], color=GEMMA3_COLOR, linewidth=2.4,
               marker="o", markersize=10,
               markeredgecolor="white", markeredgewidth=0.8,
               label="Gemma-3"),
        Line2D([0], [0], color=GEMMA4_COLOR, linewidth=2.4,
               marker="o", markersize=10,
               markeredgecolor="white", markeredgewidth=0.8,
               label="Gemma-4"),
    ]
    ax_radar.legend(
        handles=radar_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False,
        fontsize=14.5,
        handlelength=2.4,
        columnspacing=1.8,
    )


    out_r = OUTPUT_DIR / "gemma_family_radar.png"
    fig_r.savefig(out_r, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig_r)
    print(f"Saved: {out_r}")

    # --- Figure 2: scatter ---
    fig_s, ax_scatter = plt.subplots(figsize=(10, 8))
    fig_s.patch.set_facecolor("white")
    ax_scatter.set_facecolor("white")
    _draw_scatter(ax_scatter, data)

    fig_s.text(
        0.5, 0.99,
        "magnitude attenuated",
        ha="center", va="top",
        fontsize=18, color="#666666", style="italic",
    )

    cat_handles = [
        mpatches.Patch(facecolor=CATEGORY_COLORS[c], edgecolor="white",
                       linewidth=0.5, label=c)
        for c in ["Personality & Social", "Interpersonal", "Behavioral", "Socioeconomic & App."]
    ]
    cat_handles.append(
        Line2D([0], [0], color="#AAAAAA", linewidth=1.4, linestyle="--",
               label="Identity (y = x)")
    )
    ax_scatter.legend(
        handles=cat_handles,
        loc="lower right",
        fontsize=13.5,
        frameon=True,
        framealpha=0.95,
        edgecolor="#cccccc",
    )

    fig_s.tight_layout()
    out_s = OUTPUT_DIR / "gemma_family_scatter.png"
    fig_s.savefig(out_s, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig_s)
    print(f"Saved: {out_s}")

    save_correlation_csv(data, OUTPUT_DIR / "gemma_family_scatter_correlations.csv")


if __name__ == "__main__":
    main()
