#!/usr/bin/env python3
"""scenario_sensitivity scatter with bootstrap CI error bars and significance markers.

Changes vs. original scenario_sensitivity.png:
  - Cross-shaped error bars: x = 95% bootstrap CI of signed mean Δ,
                              y = 95% bootstrap CI of mean |Δ|
  - Significance markers (*** ** * ns) inside each dot's label (BH-corrected)
  - Saves as scenario_sensitivity_ci.png (does NOT overwrite original)

Bootstrap unit: base face (one mean Δ per face per scenario, averaged across
all variations and models). n ≈ 130–334 faces per scenario.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats
from adjustText import adjust_text

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "output" / "evaluation"
OUT_DIR  = ROOT / "output" / "evaluation" / "eval_charts"
MODELS   = ["gemma3", "gemma4", "internvl", "llava_next", "pixtral", "qwen3"]
OFFICIAL_SCENARIOS = set(range(1, 26))
N_BOOT   = 1000
SEED     = 42

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
    "Personality & Social", "Interpersonal",
    "Behavioral", "Socioeconomic & App.",
]
CATEGORY_COLORS = {
    "Personality & Social": "#6aaed6",
    "Interpersonal":        "#74c476",
    "Behavioral":           "#fd8d3c",
    "Socioeconomic & App.": "#e8435a",
}

# ── Utilities ─────────────────────────────────────────────────────────────────

def _bh_correct(p_values: list[float]) -> list[float]:
    n = len(p_values)
    if n == 0:
        return []
    ordered = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [1.0] * n
    prev = 1.0
    for rank, (idx, p) in enumerate(reversed(ordered), 1):
        adj = min(p * n / (n - rank + 1), prev)
        prev = adj
        adjusted[idx] = adj
    return adjusted


def _sig_marker(p: float) -> str:
    if math.isnan(p): return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


# ── Data ──────────────────────────────────────────────────────────────────────

def load_scenario_labels(comparison_dir: Path) -> dict[int, str]:
    path = comparison_dir / "scenario_comparison.csv"
    mapping = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                mapping[int(row["scenario"])] = row["scenario_label"].strip()
            except Exception:
                pass
    return mapping


def load_face_means_per_scenario() -> dict[int, list[float]]:
    """scenario_id → list of per-face mean Δ (averaged across all variations & models)."""
    face_scen_deltas: dict[tuple[int, str], list[float]] = defaultdict(list)
    for model in MODELS:
        path = EVAL_DIR / model / "paired_deltas.csv"
        if not path.exists():
            continue
        with path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    scen  = int(row["scenario"])
                    delta = float(row["delta"])
                except (ValueError, TypeError):
                    continue
                if scen not in OFFICIAL_SCENARIOS:
                    continue
                face = (row.get("face_folder") or "").strip()
                if not face:
                    continue
                face_scen_deltas[(scen, face)].append(delta)

    result: dict[int, list[float]] = defaultdict(list)
    for (scen, _face), deltas in face_scen_deltas.items():
        result[scen].append(float(np.mean(deltas)))
    return dict(result)


# ── Bootstrap + stats ─────────────────────────────────────────────────────────

def compute_point(face_means: list[float], rng: np.random.Generator) -> dict:
    """Compute signed mean, |mean|, and bootstrap CIs for both axes."""
    arr = np.asarray(face_means, dtype=float)
    n   = len(arr)

    signed_mean = float(np.mean(arr))
    abs_mean    = float(np.mean(np.abs(arr)))

    # Bootstrap: resample face means, compute both statistics each time
    boot_signed = []
    boot_abs    = []
    for _ in range(N_BOOT):
        sample = rng.choice(arr, size=n, replace=True)
        boot_signed.append(float(np.mean(sample)))
        boot_abs.append(float(np.mean(np.abs(sample))))

    ci_signed_lo = float(np.percentile(boot_signed, 2.5))
    ci_signed_hi = float(np.percentile(boot_signed, 97.5))
    ci_abs_lo    = float(np.percentile(boot_abs,    2.5))
    ci_abs_hi    = float(np.percentile(boot_abs,    97.5))

    # Wilcoxon signed-rank vs zero (two-sided)
    try:
        _, p_raw = scipy_stats.wilcoxon(arr, zero_method="wilcox", alternative="two-sided")
    except Exception:
        p_raw = float("nan")

    return {
        "signed_mean":    signed_mean,
        "abs_mean":       abs_mean,
        "ci_signed_lo":   ci_signed_lo,
        "ci_signed_hi":   ci_signed_hi,
        "ci_abs_lo":      ci_abs_lo,
        "ci_abs_hi":      ci_abs_hi,
        "p_raw":          p_raw,
        "n":              n,
    }


# ── Plot ──────────────────────────────────────────────────────────────────────

def _should_label(pt: dict, all_pts: list[dict], top_n: int = 8) -> bool:
    dist = lambda p: p["signed_mean"] ** 2 + p["abs_mean"] ** 2
    ranked = sorted(all_pts, key=dist, reverse=True)
    return pt in ranked[:top_n]


def plot(points: list[dict], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.grid(color="#E0E0E0", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.axvline(0, color="#888888", linewidth=1.0, linestyle="--", zorder=1)

    for cat in CATEGORY_ORDER:
        cat_pts = [p for p in points if p["category"] == cat]
        color   = CATEGORY_COLORS[cat]

        for pt in cat_pts:
            # Error bars: x = signed-mean CI, y = |mean| CI
            x_err = np.array([[pt["signed_mean"] - pt["ci_signed_lo"]],
                               [pt["ci_signed_hi"] - pt["signed_mean"]]])
            y_err = np.array([[pt["abs_mean"] - pt["ci_abs_lo"]],
                               [pt["ci_abs_hi"] - pt["abs_mean"]]])
            ax.errorbar(
                pt["signed_mean"], pt["abs_mean"],
                xerr=x_err, yerr=y_err,
                fmt="none",
                ecolor=color, elinewidth=1.4, capsize=3.5, capthick=1.4,
                alpha=0.7, zorder=2,
            )

        # Scatter dots (on top of error bars)
        xs = [p["signed_mean"] for p in cat_pts]
        ys = [p["abs_mean"]    for p in cat_pts]
        ax.scatter(xs, ys,
                   color=color, s=140, alpha=0.85,
                   edgecolors="white", linewidths=0.9,
                   zorder=3, label=cat)

    # ---- Labels for outliers ------------------------------------------------
    texts = []
    labeled_pts = [p for p in points if _should_label(p, points)]

    for pt in labeled_pts:
        sig   = pt.get("sig", "")
        short = pt["label"].replace(" | ", "\n")
        if sig and sig != "ns":
            short = short + f"\n{sig}"

        if pt["label"] == "Responsible | Irresponsible":
            ax.annotate(
                short,
                xy=(pt["signed_mean"], pt["abs_mean"]),
                xytext=(pt["signed_mean"] - 0.03, pt["abs_mean"] - 0.018),
                fontsize=10.5, color="#333333",
                ha="right", va="center", zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.85),
                arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=0.8),
            )
            continue

        if pt["label"] == "Stylish | Unstylish":
            ax.annotate(
                short,
                xy=(pt["signed_mean"], pt["abs_mean"]),
                xytext=(pt["signed_mean"] - 0.06, pt["abs_mean"] - 0.03),
                fontsize=10.5, color="#333333",
                ha="right", va="top", zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.85),
                arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=0.8),
            )
            continue

        if pt["label"] == "Conscientious | Careless":
            ax.annotate(
                short,
                xy=(pt["signed_mean"], pt["abs_mean"]),
                xytext=(pt["signed_mean"] - 0.02, pt["abs_mean"] - 0.015),
                fontsize=10.5, color="#333333",
                ha="right", va="top", zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.85),
                arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=0.8),
            )
            continue

        if pt["label"] == "Open-minded | Closed-minded":
            ax.annotate(
                short,
                xy=(pt["signed_mean"], pt["abs_mean"]),
                xytext=(pt["signed_mean"] + 0.01, pt["abs_mean"] - 0.025),
                fontsize=10.5, color="#333333",
                ha="left", va="top", zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.85),
                arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=0.8),
            )
            continue

        if pt["label"] == "Confident | Insecure":
            ax.annotate(
                short,
                xy=(pt["signed_mean"], pt["abs_mean"]),
                xytext=(pt["signed_mean"] - 0.045, pt["abs_mean"] + 0.008),
                fontsize=10.5, color="#333333",
                ha="right", va="bottom", zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.85),
                arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=0.8),
            )
            continue

        t = ax.text(
            pt["signed_mean"], pt["abs_mean"], short,
            fontsize=10.5, color="#333333",
            va="bottom", ha="left", zorder=5,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor="none", alpha=0.85),
        )
        texts.append(t)

    MANUAL_LABELS = {"Responsible | Irresponsible", "Stylish | Unstylish", "Conscientious | Careless", "Confident | Insecure", "Open-minded | Closed-minded"}
    auto_pts = [p for p in labeled_pts if p["label"] not in MANUAL_LABELS]

    adjust_text(
        texts,
        x=[p["signed_mean"] for p in auto_pts],
        y=[p["abs_mean"]    for p in auto_pts],
        ax=ax,
        arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=0.8),
        expand=(1.8, 2.2),
        force_text=(1.0, 1.2),
        force_points=(0.8, 1.0),
    )

    # ---- Significance markers for non-labeled points -------------------------
    for pt in points:
        if _should_label(pt, points):
            continue
        sig = pt.get("sig", "")
        if sig in ("***", "**", "*"):
            ax.text(pt["signed_mean"], pt["abs_mean"] + 0.003, sig,
                    ha="center", va="bottom", fontsize=8,
                    color="#1A7A4A", fontweight="bold", zorder=4)

    # ---- Axes ----------------------------------------------------------------
    all_x = [p["signed_mean"] for p in points]
    all_y = [p["abs_mean"]    for p in points]
    xpad  = (max(all_x) - min(all_x)) * 0.14
    ypad  = (max(all_y) - min(all_y)) * 0.14
    ax.set_xlim(min(all_x) - xpad, max(all_x) + xpad)
    ax.set_ylim(max(0, min(all_y) - ypad), max(all_y) + ypad)

    ax.set_xlabel("Average signed shift across models, Δ  (face-level bootstrap mean)",
                  fontsize=13, labelpad=12)
    ax.set_ylabel("Mean absolute shift across models, |Δ|  (face-level bootstrap mean)",
                  fontsize=13, labelpad=12)
    ax.tick_params(axis="both", labelsize=11.5, length=0)

    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")

    ax.set_title(
        "Scenario sensitivity — with 95% bootstrap CIs and significance markers",
        fontsize=14, fontweight="bold", pad=20,
    )

    # ---- Legend --------------------------------------------------------------
    # Category handles already added via scatter label=cat
    from matplotlib.lines import Line2D
    extra = [
        Line2D([0], [0], color="#888888", linewidth=1.4, alpha=0.7,
               marker="|", markersize=6, label="Bootstrap 95% CI (face-level)"),
    ]
    handles, lbls = ax.get_legend_handles_labels()
    ax.legend(handles=handles + extra,
              labels=lbls + ["Bootstrap 95% CI (face-level)"],
              loc="upper left", fontsize=10.5,
              frameon=True, framealpha=0.92,
              edgecolor="#cccccc", markerscale=1.1)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    comp_dirs = sorted(d for d in EVAL_DIR.iterdir()
                       if d.is_dir() and d.name.startswith("model_comparison"))
    if not comp_dirs:
        raise FileNotFoundError("No model_comparison directory found.")
    comp_dir = comp_dirs[-1]
    print(f"Using: {comp_dir.name}")
    labels = load_scenario_labels(comp_dir)

    print("Loading face-level data from paired_deltas.csv …")
    face_means_per_scen = load_face_means_per_scenario()
    print(f"  Scenarios: {len(face_means_per_scen)}")

    rng = np.random.default_rng(SEED)

    # Compute per-scenario stats
    raw_ps:    list[float] = []
    scen_ids:  list[int]   = []
    pts_draft: list[dict]  = []
    for scen_id, face_means in sorted(face_means_per_scen.items()):
        pt = compute_point(face_means, rng)
        pt["label"]    = labels.get(scen_id, str(scen_id))
        pt["category"] = SCENARIO_CATEGORIES.get(pt["label"], "Personality & Social")
        raw_ps.append(pt["p_raw"] if not math.isnan(pt["p_raw"]) else 1.0)
        scen_ids.append(scen_id)
        pts_draft.append(pt)

    adj_ps = _bh_correct(raw_ps)
    for pt, p_adj in zip(pts_draft, adj_ps):
        pt["p_adj"] = p_adj
        pt["sig"]   = _sig_marker(p_adj)

    # Print
    print(f"\n{'Scenario':<36} {'n':>5} {'signed':>8} {'|mean|':>8} "
          f"{'x_ci_lo':>8} {'x_ci_hi':>8} {'p_adj':>10} {'sig':>5}")
    print("-" * 90)
    for pt in sorted(pts_draft, key=lambda p: p["signed_mean"]):
        print(f"{pt['label']:<36} {pt['n']:>5} {pt['signed_mean']:>+8.4f} "
              f"{pt['abs_mean']:>8.4f} {pt['ci_signed_lo']:>+8.4f} "
              f"{pt['ci_signed_hi']:>+8.4f} {pt['p_adj']:>10.3e} {pt['sig']:>5}")

    n_sig = sum(1 for p in pts_draft if p["sig"] not in ("ns", ""))
    print(f"\nSignificant (BH-corrected): {n_sig}/{len(pts_draft)}")

    output_path = OUT_DIR / "scenario_sensitivity_ci.png"
    plot(pts_draft, output_path)


if __name__ == "__main__":
    main()
