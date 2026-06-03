#!/usr/bin/env python3
"""Mean absolute shift by category — with cross-model agreement metrics.

Adds to the original mean_abs_shift_by_category.png:
  - Kendall's W (concordance of model rankings of the 4 categories)
  - ICC(2,1) computed on all 25 scenarios (two-way mixed, consistency)
  - Pairwise Spearman correlation heatmap between models (bottom panel)
  - 95% bootstrap CI for ICC

Saved as mean_abs_shift_by_category_agreement.png (originals NOT overwritten).
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy import stats as scipy_stats

ROOT     = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "output" / "evaluation"
OUT_DIR  = ROOT / "output" / "evaluation" / "eval_charts"

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

MODEL_DISPLAY = {
    "gemma3":    "Gemma-3",
    "gemma4":    "Gemma-4",
    "internvl":  "InternVL3",
    "llava_next": "LLaVA-v1.6",
    "pixtral":   "Pixtral",
    "qwen3":     "Qwen3-VL",
}


# ── Data ──────────────────────────────────────────────────────────────────────

def load_data(evaluation_root: Path) -> tuple[list[dict], list[str]]:
    comp_dirs = sorted(d for d in evaluation_root.iterdir()
                       if d.is_dir() and d.name.startswith("model_comparison"))
    if not comp_dirs:
        raise FileNotFoundError("No model_comparison directory found.")
    comp_dir = comp_dirs[-1]
    print(f"Using: {comp_dir.name}")
    csv_path = comp_dir / "scenario_comparison.csv"
    rows: list[dict] = []
    with csv_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    models = [k.replace("_mean_delta", "") for k in rows[0] if k.endswith("_mean_delta")]
    return rows, models


# ── Agreement metrics ─────────────────────────────────────────────────────────

def kendalls_w(matrix: np.ndarray) -> tuple[float, float, float]:
    """Kendall's W for k raters (rows) × n items (cols). Returns W, chi2, p."""
    k, n = matrix.shape
    ranks = np.array([scipy_stats.rankdata(row) for row in matrix])
    R = ranks.sum(axis=0)
    R_mean = k * (n + 1) / 2.0
    S = float(np.sum((R - R_mean) ** 2))
    W = 12.0 * S / (k ** 2 * (n ** 3 - n))
    chi2 = k * (n - 1) * W
    p = float(1.0 - scipy_stats.chi2.cdf(chi2, df=n - 1))
    return W, chi2, p


def icc_21(data: np.ndarray) -> tuple[float, float, float]:
    """ICC(2,1) — two-way mixed, consistency — for n items × k raters matrix.

    Returns icc, F_stat, p_value.
    ICC = (MS_r - MS_e) / (MS_r + (k-1)*MS_e)
    """
    n, k = data.shape
    grand_mean = data.mean()
    # MS between subjects (items = scenarios)
    row_means = data.mean(axis=1)
    SS_r = k * np.sum((row_means - grand_mean) ** 2)
    MS_r = SS_r / (n - 1)
    # MS between raters (columns = models)
    col_means = data.mean(axis=0)
    SS_c = n * np.sum((col_means - grand_mean) ** 2)
    MS_c = SS_c / (k - 1)
    # MS error
    SS_t = np.sum((data - grand_mean) ** 2)
    SS_e = SS_t - SS_r - SS_c
    MS_e = SS_e / ((n - 1) * (k - 1))
    # ICC consistency
    icc = (MS_r - MS_e) / (MS_r + (k - 1) * MS_e)
    # F-test for items
    F_stat = MS_r / MS_e
    p_val = float(1.0 - scipy_stats.f.cdf(F_stat, n - 1, (n - 1) * (k - 1)))
    return float(icc), float(F_stat), p_val


def bootstrap_icc_ci(data: np.ndarray, n_boot: int = 1000, seed: int = 42
                     ) -> tuple[float, float]:
    """95% bootstrap CI for ICC(2,1) by resampling rows (scenarios)."""
    rng = np.random.default_rng(seed)
    n = data.shape[0]
    boot_iccs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        icc_b, _, _ = icc_21(data[idx])
        boot_iccs.append(icc_b)
    return float(np.percentile(boot_iccs, 2.5)), float(np.percentile(boot_iccs, 97.5))


def pairwise_spearman(data: np.ndarray, models: list[str]
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise Spearman r matrix (k × k) and p-value matrix."""
    k = len(models)
    r_mat = np.ones((k, k))
    p_mat = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            r, p = scipy_stats.spearmanr(data[:, i], data[:, j])
            r_mat[i, j] = r_mat[j, i] = r
            p_mat[i, j] = p_mat[j, i] = p
    return r_mat, p_mat


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot(rows: list[dict], models: list[str], output_path: Path) -> None:

    # ── Build category × model matrix (mean |Δ|) ───────────────────────────
    cat_model: dict[str, dict[str, list[float]]] = {
        cat: {m: [] for m in models} for cat in CATEGORIES
    }
    for row in rows:
        label = (row.get("scenario_label") or "").strip()
        cat = SCENARIO_CATEGORIES.get(label)
        if cat is None:
            continue
        for m in models:
            try:
                cat_model[cat][m].append(abs(float(row[f"{m}_mean_delta"])))
            except (ValueError, TypeError, KeyError):
                pass

    means: dict[str, dict[str, float]] = {}
    for cat in CATEGORIES:
        means[cat] = {m: float(np.mean(v)) if v else 0.0
                      for m, v in cat_model[cat].items()}

    # matrix: models (rows) × categories (cols)
    cat_matrix = np.array([[means[cat][m] for cat in CATEGORIES] for m in models])
    W, chi2_w, p_w = kendalls_w(cat_matrix)

    # ── Scenario-level matrix for ICC (25 scenarios × 6 models, abs |Δ|) ──
    scen_vals: list[list[float]] = []
    for row in rows:
        label = (row.get("scenario_label") or "").strip()
        if label not in SCENARIO_CATEGORIES:
            continue
        vals = []
        for m in models:
            try:
                vals.append(abs(float(row[f"{m}_mean_delta"])))
            except (ValueError, TypeError, KeyError):
                break
        if len(vals) == len(models):
            scen_vals.append(vals)
    scen_matrix = np.array(scen_vals)    # n_scenarios × n_models

    icc, F_icc, p_icc = icc_21(scen_matrix)
    ci_lo, ci_hi = bootstrap_icc_ci(scen_matrix)

    r_mat, p_mat = pairwise_spearman(scen_matrix, models)

    print(f"\nKendall's W = {W:.3f}, χ²({len(CATEGORIES)-1}) = {chi2_w:.2f}, p = {p_w:.4f}")
    print(f"ICC(2,1)   = {icc:.3f} [{ci_lo:.3f}, {ci_hi:.3f}], "
          f"F = {F_icc:.2f}, p = {p_icc:.4f}")
    print("\nPairwise Spearman r matrix:")
    display = [MODEL_DISPLAY.get(m, m) for m in models]
    print("         " + "  ".join(f"{d:>10}" for d in display))
    for i, mi in enumerate(models):
        row_str = "  ".join(f"{r_mat[i,j]:>10.3f}" for j in range(len(models)))
        print(f"{MODEL_DISPLAY.get(mi, mi):>10} {row_str}")

    # ── Figure layout: bar chart (top) + Spearman heatmap (bottom) ─────────
    fig = plt.figure(figsize=(13, 10))
    gs  = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.0], hspace=0.38)
    ax_bar  = fig.add_subplot(gs[0])
    ax_heat = fig.add_subplot(gs[1])

    fig.patch.set_facecolor("white")

    # ── Bar chart ────────────────────────────────────────────────────────────
    ax_bar.set_facecolor("#F7F7F7")
    ax_bar.grid(axis="y", color="white", linewidth=1.0, zorder=0)
    ax_bar.set_axisbelow(True)

    n_models = len(models)
    n_cats   = len(CATEGORIES)
    group_width   = 0.96
    bar_width     = group_width / n_cats
    model_spacing = group_width + bar_width * 0.92
    x = np.arange(n_models) * model_spacing

    for ci, cat in enumerate(CATEGORIES):
        offsets = (ci - (n_cats - 1) / 2) * bar_width
        vals = [means[cat][m] for m in models]
        bars = ax_bar.bar(
            x + offsets, vals, width=bar_width * 0.92,
            color=CATEGORY_COLORS[cat], label=cat,
            zorder=3, linewidth=0.4, edgecolor="#888888",
        )
        for bar, v in zip(bars, vals):
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{v:.3f}", ha="center", va="bottom",
                fontsize=10, color="#333333",
            )

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in models], fontsize=14)
    ax_bar.set_xlim(x[0] - 0.60, x[-1] + 0.60)
    ax_bar.set_ylabel("Mean |Δ| (absolute shift)", fontsize=14, labelpad=8)
    ax_bar.tick_params(axis="y", labelsize=13)
    ax_bar.set_ylim(0, ax_bar.get_ylim()[1] * 1.22)
    ax_bar.set_title(
        "Mean Absolute Prediction Shift by Scenario Category and Model",
        fontsize=15, fontweight="bold", pad=16,
    )

    for spine in ["top", "right", "left", "bottom"]:
        ax_bar.spines[spine].set_visible(False)
    ax_bar.tick_params(axis="both", length=0)

    legend_handles = [
        mpatches.Patch(facecolor=CATEGORY_COLORS[cat], edgecolor="#888888",
                       linewidth=0.4, label=cat)
        for cat in CATEGORIES
    ]
    ax_bar.legend(handles=legend_handles, ncol=2, loc="upper right",
                  frameon=True, framealpha=0.95, fontsize=12, edgecolor="#cccccc")

    # ── Agreement annotation box ──────────────────────────────────────────────
    def _fmt_p(p: float) -> str:
        if p < 0.001: return "< 0.001"
        if p < 0.01:  return f"= {p:.3f}"
        return f"= {p:.3f}"

    w_stars  = "***" if p_w  < 0.001 else ("**" if p_w  < 0.01 else ("*" if p_w  < 0.05 else "ns"))
    icc_stars = "***" if p_icc < 0.001 else ("**" if p_icc < 0.01 else ("*" if p_icc < 0.05 else "ns"))

    box_txt = (
        f"Cross-model agreement\n"
        f"Kendall's W = {W:.3f}{w_stars}  (χ²({len(CATEGORIES)-1}) = {chi2_w:.2f}, p {_fmt_p(p_w)})\n"
        f"ICC(2,1)  = {icc:.3f}{icc_stars}  [95% CI: {ci_lo:.3f}, {ci_hi:.3f}]"
    )
    ax_bar.text(
        0.01, 0.99, box_txt,
        transform=ax_bar.transAxes,
        fontsize=11, va="top", ha="left",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#cccccc", alpha=0.95),
        zorder=6,
    )

    # ── Spearman heatmap ──────────────────────────────────────────────────────
    pale_cmap = LinearSegmentedColormap.from_list(
        "pale_rg", ["#E07070", "#F5DC60", "#5BBD5B"]   # muted rose → muted yellow → muted green
    )
    ax_heat.set_facecolor("white")
    im = ax_heat.imshow(r_mat, cmap=pale_cmap, vmin=0.2, vmax=1.0, aspect="auto")

    disp_labels = [MODEL_DISPLAY.get(m, m) for m in models]
    ax_heat.set_xticks(range(len(models)))
    ax_heat.set_yticks(range(len(models)))
    ax_heat.set_xticklabels(disp_labels, fontsize=11, rotation=25, ha="right")
    ax_heat.set_yticklabels(disp_labels, fontsize=11)

    for i in range(len(models)):
        for j in range(len(models)):
            val = r_mat[i, j]
            txt_col = "#111111"
            sig = ""
            if i != j:
                p_ij = p_mat[i, j]
                sig = "***" if p_ij < 0.001 else ("**" if p_ij < 0.01
                      else ("*" if p_ij < 0.05 else ""))
            cell_txt = f"{val:.2f}{sig}" if i != j else "—"
            ax_heat.text(j, i, cell_txt, ha="center", va="center",
                         fontsize=10, color=txt_col, fontweight="bold")

    ax_heat.set_title(
        "Pairwise Spearman r between models (scenario-level |Δ|,  * p<.05  ** p<.01  *** p<.001)",
        fontsize=12, pad=8,
    )
    for sp in ax_heat.spines.values():
        sp.set_visible(False)
    ax_heat.tick_params(length=0)

    cbar = fig.colorbar(im, ax=ax_heat, orientation="vertical",
                        fraction=0.025, pad=0.02)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label("Spearman r", fontsize=10)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {output_path}")
    return r_mat, p_mat


# ── Main ──────────────────────────────────────────────────────────────────────

def save_spearman_csv(r_mat: np.ndarray, p_mat: np.ndarray,
                      models: list[str], output_path: Path) -> None:
    disp = [MODEL_DISPLAY.get(m, m) for m in models]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model"] + disp)
        for i, mi in enumerate(models):
            row = [MODEL_DISPLAY.get(mi, mi)]
            for j in range(len(models)):
                if i == j:
                    row.append("1.000")
                else:
                    row.append(f"{r_mat[i, j]:.4f}")
            writer.writerow(row)
    print(f"Saved: {output_path}")

    # also write p-values
    p_path = output_path.with_name(output_path.stem + "_pvalues.csv")
    with p_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model"] + disp)
        for i, mi in enumerate(models):
            row = [MODEL_DISPLAY.get(mi, mi)]
            for j in range(len(models)):
                if i == j:
                    row.append("—")
                else:
                    row.append(f"{p_mat[i, j]:.4f}")
            writer.writerow(row)
    print(f"Saved: {p_path}")


def main() -> None:
    rows, models = load_data(EVAL_DIR)
    print(f"Models: {models}")
    out = OUT_DIR / "mean_abs_shift_by_category_agreement.png"
    r_mat, p_mat = plot(rows, models, out)
    save_spearman_csv(r_mat, p_mat, models,
                      OUT_DIR / "spearman_cross_model.csv")


if __name__ == "__main__":
    main()
