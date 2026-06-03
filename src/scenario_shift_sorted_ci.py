#!/usr/bin/env python3
"""scenario_shift_sorted with bootstrap CIs and Wilcoxon significance markers.

Changes vs. original scenario_shift_sorted.png:
  - Error bars = 95% bootstrap CI from face-level means (n ≈ 130–334 faces per scenario)
  - Thin grey whisker retained as model-range reference
  - Wilcoxon signed-rank vs zero per scenario, BH-corrected across 25
  - Significance markers (*** ** * ns) next to each point
  - Saves as scenario_shift_sorted_ci.png (does NOT overwrite original)

Bootstrap unit: base face (one mean Δ per face per scenario, averaged across
all variations and models). This gives proper statistical uncertainty rather
than the model-disagreement range shown in the original.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
from scipy import stats as scipy_stats

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
EVAL_DIR    = ROOT / "output" / "evaluation"
OUT_DIR     = ROOT / "output" / "evaluation" / "eval_charts"
MODELS      = ["gemma3", "gemma4", "internvl", "llava_next", "pixtral", "qwen3"]
OFFICIAL_SCENARIOS = set(range(1, 26))
N_BOOT      = 1000
SEED        = 42

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
    "gemma3": "Gemma-3", "gemma4": "Gemma-4",
    "internvl": "InternVL3", "llava_next": "LLaVA-v1.6",
    "pixtral": "Pixtral", "qwen3": "Qwen3-VL",
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
    """scenario_id → label string."""
    path = comparison_dir / "scenario_comparison.csv"
    mapping = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                mapping[int(row["scenario"])] = row["scenario_label"].strip()
            except Exception:
                pass
    return mapping


def load_model_means(comparison_dir: Path) -> dict[int, dict[str, float]]:
    """scenario_id → {model: mean_delta} from scenario_comparison.csv."""
    path = comparison_dir / "scenario_comparison.csv"
    result: dict[int, dict[str, float]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            scen = int(row["scenario"])
            vals = {}
            for k, v in row.items():
                if k.endswith("_mean_delta"):
                    model = k.replace("_mean_delta", "")
                    try:
                        vals[model] = float(v)
                    except Exception:
                        pass
            result[scen] = vals
    return result


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
                    scen = int(row["scenario"])
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


# ── Bootstrap + significance ──────────────────────────────────────────────────

def compute_stats(face_means: list[float], rng: np.random.Generator) -> dict:
    arr = np.asarray(face_means, dtype=float)
    mean = float(np.mean(arr))
    n = len(arr)

    # Bootstrap CI
    boot = np.array([np.mean(rng.choice(arr, size=n, replace=True)) for _ in range(N_BOOT)])
    ci_lo = float(np.percentile(boot, 2.5))
    ci_hi = float(np.percentile(boot, 97.5))

    # Wilcoxon signed-rank vs zero
    try:
        _, p_raw = scipy_stats.wilcoxon(arr, zero_method="wilcox", alternative="two-sided")
    except Exception:
        p_raw = float("nan")

    return {"mean": mean, "ci_lo": ci_lo, "ci_hi": ci_hi, "p_raw": p_raw, "n": n}


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot(scenario_data: list[dict], output_path: Path) -> None:
    n = len(scenario_data)
    fig_height = max(8, n * 0.44)
    fig, ax = plt.subplots(figsize=(13.5, fig_height))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y_positions = np.arange(n)

    # Alternating category background bands
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
        ax.axhspan(start - 0.5, end + 0.5,
                   facecolor=CATEGORY_BG_COLORS[cat], alpha=0.55, zorder=0)

    ax.axvline(0, color="#888888", linewidth=1.0, linestyle="--", zorder=1)
    ax.grid(axis="x", color="#dddddd", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    for i, d in enumerate(scenario_data):
        color = CATEGORY_COLORS[d["category"]]
        sig   = d["sig"]

        # Thin grey whisker: model min–max range (reference)
        if d.get("model_min") is not None and d.get("model_max") is not None:
            ax.plot([d["model_min"], d["model_max"]], [i, i],
                    color="#cccccc", linewidth=1.8,
                    solid_capstyle="round", zorder=2, alpha=0.7)

        # Bootstrap 95% CI bar (thicker, coloured)
        ax.plot([d["ci_lo"], d["ci_hi"]], [i, i],
                color=color, linewidth=5.0,
                solid_capstyle="round", zorder=3, alpha=0.55)

        # Central diamond
        ax.plot(d["mean"], i,
                marker="D", markersize=10,
                color=color,
                markeredgecolor="white", markeredgewidth=1.4,
                zorder=4)

        # Mean value label
        sign = "+" if d["mean"] >= 0 else ""
        x_offset = 0.011 if d["mean"] >= 0 else -0.011
        ha_lbl   = "left"  if d["mean"] >= 0 else "right"
        ax.text(d["mean"] + x_offset, i - 0.26,
                f"{sign}{d['mean']:.3f}",
                va="top", ha=ha_lbl, fontsize=10.5,
                color=color, fontweight="bold", zorder=5)

        # Significance marker to the right of the CI bar
        sig_x = max(d["ci_hi"], d.get("model_max", d["ci_hi"])) + 0.005
        sig_color = "#1A7A4A" if sig in ("***", "**", "*") else "#C0392B"
        ax.text(sig_x, i, sig,
                va="center", ha="left", fontsize=9,
                color=sig_color, fontweight="bold", zorder=5)

    # Y-axis labels: negative pole left, positive pole right
    neg_labels = [d["label"].split("|")[1].strip() for d in scenario_data]
    pos_labels = [d["label"].split("|")[0].strip() for d in scenario_data]

    ax.set_yticks(y_positions)
    ax.set_yticklabels(neg_labels, fontsize=12.5, color="#111111")

    ax_r = ax.twinx()
    ax_r.set_ylim(ax.get_ylim())
    ax_r.set_yticks(y_positions)
    ax_r.set_yticklabels(pos_labels, fontsize=12.5, color="#111111")
    ax_r.tick_params(axis="y", length=0)
    for sp in ax_r.spines.values():
        sp.set_visible(False)

    # Pole symbols
    ax.text(0, n - 0.35, "⊖",
            transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=24, color="#7A1010", fontweight="bold", clip_on=False)
    ax_r.text(1, n - 0.35, "⊕",
              transform=ax_r.get_yaxis_transform(), ha="left", va="bottom",
              fontsize=24, color="#1B6B1B", fontweight="bold", clip_on=False)

    ax.set_xlabel("Average Mean Δ across all models and variations (face-level)", fontsize=12.5, labelpad=8)
    ax.tick_params(axis="x", labelsize=11.5)
    ax.set_title("All 25 Scenarios: Average Prediction Shift with Bootstrap 95% CI",
                 fontsize=14.5, fontweight="bold", pad=16)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(axis="both", length=0)

    legend_handles = [
        mpatches.Patch(facecolor=CATEGORY_COLORS[cat],
                       edgecolor="#888888", linewidth=0.4, label=cat)
        for cat in CATEGORIES
    ] + [
        Line2D([0], [0], color="#aaaaaa", linewidth=1.8, alpha=0.7,
               solid_capstyle="round", label="Model min–max range"),
        Line2D([0], [0], color="#888888", linewidth=5.0, alpha=0.55,
               solid_capstyle="round", label="Bootstrap 95% CI (face-level)"),
        Line2D([0], [0], marker="D", linestyle="None", markersize=7,
               color="#888888", label="Cross-model face-level mean"),
    ]
    ax.legend(handles=legend_handles, loc="lower right",
              frameon=True, framealpha=0.95, fontsize=11, edgecolor="#cccccc")

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

    labels      = load_scenario_labels(comp_dir)
    model_means = load_model_means(comp_dir)

    print("Loading face-level data from paired_deltas.csv …")
    face_means_per_scen = load_face_means_per_scenario()
    print(f"  Scenarios with face data: {len(face_means_per_scen)}")

    rng = np.random.default_rng(SEED)

    # Compute stats per scenario
    scenario_stats: dict[int, dict] = {}
    raw_ps: list[float] = []
    scen_ids: list[int] = []
    for scen_id, face_means in sorted(face_means_per_scen.items()):
        stats = compute_stats(face_means, rng)
        scenario_stats[scen_id] = stats
        raw_ps.append(stats["p_raw"] if not math.isnan(stats["p_raw"]) else 1.0)
        scen_ids.append(scen_id)

    adj_ps = _bh_correct(raw_ps)
    for scen_id, p_adj in zip(scen_ids, adj_ps):
        scenario_stats[scen_id]["p_adj"] = p_adj
        scenario_stats[scen_id]["sig"]   = _sig_marker(p_adj)

    # Print results
    print(f"\n{'Scenario':<36} {'n':>5} {'mean':>8} {'ci_lo':>8} {'ci_hi':>8} {'p_adj':>10} {'sig':>5}")
    print("-" * 80)
    for scen_id in sorted(scen_ids):
        lbl = labels.get(scen_id, str(scen_id))
        st  = scenario_stats[scen_id]
        print(f"{lbl:<36} {st['n']:>5} {st['mean']:>+8.4f} "
              f"{st['ci_lo']:>+8.4f} {st['ci_hi']:>+8.4f} "
              f"{st['p_adj']:>10.3e} {st['sig']:>5}")

    # Build scenario_data list for plot (sorted by mean)
    scenario_data = []
    for scen_id, st in scenario_stats.items():
        lbl = labels.get(scen_id, str(scen_id))
        mm  = model_means.get(scen_id, {})
        model_vals = list(mm.values())
        scenario_data.append({
            "label":    lbl,
            "category": SCENARIO_CATEGORIES.get(lbl, "Personality & Social"),
            "mean":     st["mean"],
            "ci_lo":    st["ci_lo"],
            "ci_hi":    st["ci_hi"],
            "sig":      st["sig"],
            "model_min": min(model_vals) if model_vals else None,
            "model_max": max(model_vals) if model_vals else None,
        })
    scenario_data.sort(key=lambda d: d["mean"])

    n_sig = sum(1 for d in scenario_data if d["sig"] not in ("ns", ""))
    print(f"\nSignificant scenarios (BH-corrected): {n_sig}/{len(scenario_data)}")

    output_path = OUT_DIR / "scenario_shift_sorted_ci.png"
    plot(scenario_data, output_path)


if __name__ == "__main__":
    main()
