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
            ax.plot(fval, yp - 0.17, marker="D", markersize=9,
                    color=FEMALE_COLOR, markeredgecolor="white",
                    markeredgewidth=0.6, zorder=4)
            sign = "+" if fval >= 0 else ""
            off  = (xlim[1] - xlim[0]) * 0.014
            ax.text(fval + (off if fval >= 0 else -off), yp - 0.17,
                    f"{sign}{fval:.2f}",
                    ha="left" if fval >= 0 else "right",
                    va="center", fontsize=8.5,
                    color=FEMALE_COLOR, fontweight="bold", zorder=5)
        if np.isfinite(mval):
            ax.plot(mval, yp + 0.17, marker="D", markersize=9,
                    color=MALE_COLOR, markeredgecolor="white",
                    markeredgewidth=0.6, zorder=4)
            sign = "+" if mval >= 0 else ""
            off  = (xlim[1] - xlim[0]) * 0.014
            ax.text(mval + (off if mval >= 0 else -off), yp + 0.17,
                    f"{sign}{mval:.2f}",
                    ha="left" if mval >= 0 else "right",
                    va="center", fontsize=8.5,
                    color=MALE_COLOR, fontweight="bold", zorder=5)

    # ---- Y-axis tick labels ----
    ys     = [t[0] for t in tick_items]
    labels = [t[1] for t in tick_items]
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=8.8)
    for lbl, (_, _, is_hdr) in zip(ax.get_yticklabels(), tick_items):
        if is_hdr:
            lbl.set_fontweight("bold")
            lbl.set_fontsize(9.2)

    # ---- X-axis ----
    ax.set_xlabel("Average Mean Δ across models", fontsize=10)

    # ---- Panel title ----
    ax.set_title(title, fontsize=11, fontweight="bold", color=title_color, pad=8)

    # ---- Spine styling ----
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(axis="both", length=0)


# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------

def plot(avg: dict[str, dict[str, float]], baseline: float, output_path: Path) -> None:
    all_rows  = _ordered(_build_rows(avg))
    high_rows = [r for r in all_rows if r["abs_mean"] >= baseline]
    low_rows  = [r for r in all_rows if r["abs_mean"] <  baseline]

    # Shared x range
    all_vals = [r["female"] for r in all_rows if np.isfinite(r["female"])] + \
               [r["male"]   for r in all_rows if np.isfinite(r["male"])]
    xabs = max(abs(v) for v in all_vals) * 1.30
    xlim = (-xabs, xabs)

    # Figure height: driven by the taller panel, scaled to physical inches
    fig_h = max(
        _total_data_height(high_rows),
        _total_data_height(low_rows),
    ) * INCHES_PER_UNIT + 2.5   # +2.5 for title, subtitle, legend

    fig_h = max(8.0, min(18.0, fig_h))
    fig_w = 22.0

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2,
        figsize=(fig_w, fig_h),
    )
    fig.patch.set_facecolor("white")

    # Leave room at top for title+subtitle, bottom for legend
    fig.subplots_adjust(
        left=0.16, right=0.97,
        top=0.91,  bottom=0.13,
        wspace=0.55,
    )

    _draw_panel(ax_l, high_rows, baseline, xlim,
                f"High-impact attributes  (|Δ| ≥ {baseline:.3f})", "#C84040")
    _draw_panel(ax_r, low_rows,  baseline, xlim,
                f"Below-baseline attributes  (|Δ| < {baseline:.3f})", "#3A74B0")

    # ---- Main title + subtitle (tight to axes) ----
    fig.text(0.5, 0.975,
             "Fine-grained visual attributes vs. demographic baseline",
             ha="center", va="top", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.955,
             (f"Mean signed shift, Δ (variation – base), averaged across models. "
              f"Dashed lines mark the demographic baseline (|Δ| = {baseline:.3f})."),
             ha="center", va="top", fontsize=8.5, color="#555555", style="italic")

    # ---- Shared legend at the bottom ----
    legend_handles = [
        Line2D([0], [0], marker="D", linestyle="None", markersize=7,
               color=FEMALE_COLOR, markeredgecolor="white", markeredgewidth=0.5,
               label="Female"),
        Line2D([0], [0], marker="D", linestyle="None", markersize=7,
               color=MALE_COLOR, markeredgecolor="white", markeredgewidth=0.5,
               label="Male"),
        Line2D([0], [0], color="#999999", linewidth=1.2, linestyle="--",
               label=f"Demographic baseline (|Δ| = {baseline:.3f})"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.04),
        ncol=3,
        frameon=True, framealpha=0.95,
        edgecolor="#cccccc", fontsize=10,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")
    print(f"Baseline: {baseline:.4f}  |  High: {len(high_rows)}  |  Low: {len(low_rows)}")


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

# Model display order and names matching the PDF table
MODEL_ORDER   = ["gemma3", "gemma4", "llava_next", "pixtral", "qwen3", "internvl"]
MODEL_DISPLAY = {
    "gemma3":     "Gemma 3",
    "gemma4":     "Gemma 4",
    "llava_next": "LLaVA-NeXT",
    "pixtral":    "Pixtral",
    "qwen3":      "Qwen 3",
    "internvl":   "InternVL",
}

# Categories in PDF order, with abbreviated display labels for variations
TABLE_GROUPS = [
    ("Skin",           "skin_irregularities", [
        ("Acne",         "Acne"),
        ("Freckles",     "Freckles"),
        ("Moles",        "Moles"),
    ]),
    ("Hair\\\\Color",  "hair_color", [
        ("Black",        "Black"),
        ("Blonde",       "Blonde"),
        ("Brown",        "Brown"),
        ("Gray",         "Gray"),
    ]),
    ("Hair\\\\Length", "hair_length", [
        ("Bald",         "Bald"),
        ("Long",         "Long"),
        ("Short",        "Short"),
    ]),
    ("Hair\\\\Style",  "hair_style", [
        ("Messy",        "Messy"),
        ("Mohawk",       "Mohawk"),
        ("Slicked back", "Slicked back"),
    ]),
    ("Facial\\\\Hair", "facial_hair_male", [
        ("Clean-shaven", "Clean-shaven"),
        ("Full beard",   "Full beard"),
    ]),
    ("Makeup",         "makeup_female", [
        ("Heavy",        "Heavy"),
        ("Light",        "Light"),
    ]),
    ("Lip\\\\Makeup",  "lip_makeup_female", [
        ("Red lipstick", "Red lipstick"),
    ]),
    ("Tattoos",        "tattoos", [
        ("Facial tattoo","Facial tattoo"),
    ]),
    ("Fashion",        "fashion_style", [
        ("Casual",                         "Casual"),
        ("Formal / Evening wear",          "Formal/Evening"),
        ("Functional / outdoor wear",      "Functional/outdoor"),
        ("Professional / Business formal", "Prof./Business"),
        ("Smart casual",                   "Smart casual"),
        ("Sporty / Athletic wear",         "Sporty/Athletic"),
        ("Streetwear",                     "Streetwear"),
        ("Vintage / Retro",                "Vintage / Retro"),
        ("Worn / Distressed clothing",     "Worn/Distressed"),
    ]),
    ("Eyewear",        "eyewear", [
        ("Sunglasses",   "Sunglasses"),
        ("Thick-rimmed", "Thick-rimmed"),
    ]),
    ("Piercing",       "piercings", [
        ("Multiple",     "Multiple"),
        ("Single nose",  "Single nose"),
    ]),
    ("Access.",        "accessories", [
        ("Beanie",       "Beanie"),
        ("Cap",          "Cap"),
    ]),
]

# Categories only applicable to one gender
FEMALE_ONLY = {"makeup_female", "lip_makeup_female"}
MALE_ONLY   = {"facial_hair_male"}


def _load_per_model_data(
    model_dirs: list[Path],
) -> dict[str, dict[str, dict[str, float]]]:
    """Return {model: {gender: {variation_name: mean_delta_across_scenarios}}}."""
    result: dict[str, dict[str, dict[str, float]]] = {}
    for d in model_dirs:
        acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        with (d / "variation_impact_summary.csv").open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                acc[row["gender"]][row["variation_name"]].append(float(row["mean_delta"]))
        result[d.name] = {
            gender: {var: float(np.mean(vals)) for var, vals in vdata.items()}
            for gender, vdata in acc.items()
        }
    return result


def _cell_color(avg: float) -> str:
    """Return LaTeX \\cellcolor{...} command based on average Δ value."""
    if avg >= 0.10:
        return r"\cellcolor{strongpos}"
    if avg >= 0.04:
        return r"\cellcolor{modpos}"
    if avg <= -0.10:
        return r"\cellcolor{strongneg}"
    if avg <= -0.04:
        return r"\cellcolor{modneg}"
    return ""


def _fmt_val(v: float | None, apply_color: bool = True) -> str:
    """Format a single delta value with sign."""
    if v is None:
        return r"\textemdash{}"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.3f}"


def _cell(fval: float | None, mval: float | None, cat: str) -> str:
    """Build one table cell: color + 'F_val/M_val'."""
    f_applicable = cat not in MALE_ONLY
    m_applicable = cat not in FEMALE_ONLY

    f_str = _fmt_val(fval) if (f_applicable and fval is not None) else r"\textemdash{}"
    m_str = _fmt_val(mval) if (m_applicable and mval is not None) else r"\textemdash{}"

    # Average of applicable non-None values for shading
    vals = []
    if f_applicable and fval is not None:
        vals.append(fval)
    if m_applicable and mval is not None:
        vals.append(mval)
    avg = float(np.mean(vals)) if vals else 0.0

    color = _cell_color(avg)
    f_col = r"{\color{fcol}" + f_str + "}"
    m_col = r"{\color{mcol}" + m_str + "}"
    return f"{color}{f_col}/{m_col}"


def _write_latex_table(
    per_model: dict[str, dict[str, dict[str, float]]],
    output_path: Path,
) -> None:
    models = [m for m in MODEL_ORDER if m in per_model]

    lines: list[str] = []

    # ---- Preamble snippet (paste into document header) ----
    lines += [
        r"% Required packages: booktabs, colortbl, multirow, xcolor",
        r"% Add to preamble:",
        r"%   \usepackage{booktabs,colortbl,multirow,xcolor}",
        r"%   \definecolor{strongpos}{RGB}{160,215,160}",
        r"%   \definecolor{modpos}{RGB}{210,235,210}",
        r"%   \definecolor{modneg}{RGB}{248,215,175}",
        r"%   \definecolor{strongneg}{RGB}{235,175,155}",
        r"%   \definecolor{fcol}{RGB}{180,30,50}",
        r"%   \definecolor{mcol}{RGB}{20,60,160}",
        "",
    ]

    n_models = len(models)
    col_spec = "@{}p{1.0cm}p{2.2cm}" + "c" * n_models + "@{}"

    lines += [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.10}",
        (r"\caption{Mean $\Delta$ per appearance variation across six VLMs. "
         r"Values: {\color{fcol}F} (female) / {\color{mcol}M} (male). "
         r"Shading: {\color{strongpos!60!black}$\blacksquare$}~strong pos.\ ($\Delta \geq +0.10$), "
         r"{\color{modpos!60!black}$\blacksquare$}~moderate pos., "
         r"{\color{modneg!60!black}$\blacksquare$}~moderate neg., "
         r"{\color{strongneg!60!black}$\blacksquare$}~strong neg.\ ($\Delta \leq -0.10$). "
         r"Dashes: variation not applicable for that gender.}"),
        r"\label{tab:variation_impact}",
        r"\begin{tabular}{" + col_spec + r"}",
        r"\toprule",
    ]

    # Header row 1: model names (each spans 1 column)
    header1 = " & ".join(
        ["", ""] + [r"\textbf{" + MODEL_DISPLAY[m] + "}" for m in models]
    )
    lines.append(header1 + r" \\")

    # Header row 2: F/M sub-labels
    fm_label = r"{\color{fcol}F}/{\color{mcol}M}"
    header2 = " & ".join(["", ""] + [fm_label] * n_models)
    lines.append(header2 + r" \\")
    lines.append(r"\midrule")

    # ---- Data rows ----
    for grp_display, cat_key, var_list in TABLE_GROUPS:
        n_vars = len(var_list)
        first_in_group = True

        for var_val, var_display in var_list:
            var_name = f"{cat_key}:{var_val}"
            cells = []
            for m in models:
                fdata = per_model[m].get("female", {})
                mdata = per_model[m].get("male",   {})
                fv = fdata.get(var_name)
                mv = mdata.get(var_name)
                cells.append(_cell(fv, mv, cat_key))

            # Category label: multirow on first variation of the group
            if first_in_group:
                cat_col = (r"\multirow{" + str(n_vars) + r"}{*}{\textbf{"
                           + grp_display.replace("\\\\", r"\\")
                           + r"}}")
                first_in_group = False
            else:
                cat_col = ""

            row_parts = [cat_col, var_display] + cells
            lines.append(" & ".join(row_parts) + r" \\")

        lines.append(r"\midrule")

    # Remove last \midrule, replace with \bottomrule
    lines[-1] = r"\bottomrule"

    lines += [
        r"\end{tabular}",
        r"\end{table}",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved LaTeX table: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    comp_dir   = _latest_comparison_dir(EVALUATION_ROOT)
    baseline   = _compute_baseline(comp_dir)
    model_dirs = _discover_model_dirs(EVALUATION_ROOT)

    print(f"Comparison dir : {comp_dir.name}")
    print(f"Models         : {[d.name for d in model_dirs]}")
    print(f"Baseline       : {baseline:.4f}")

    avg = _load_variation_data(model_dirs)
    plot(avg, baseline, OUTPUT_DIR / "vis_attr_vs_baseline.png")

    per_model = _load_per_model_data(model_dirs)
    _write_latex_table(per_model, OUTPUT_DIR / "vis_attr_vs_baseline_table.tex")


if __name__ == "__main__":
    main()
