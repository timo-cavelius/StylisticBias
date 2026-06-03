#!/usr/bin/env python3
"""Demographic × variation delta table.

For each appearance variation (rows) and each demographic group value (columns),
compute mean Δ averaged across all 6 models and all scenarios.

Outputs:
    output/evaluation/eval_charts/main_table_data.csv   — computed values
    output/evaluation/eval_charts/main_table.png         — rendered table image

Usage:
  python3 src/create_main_table.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EVALUATION_ROOT = Path("output/evaluation")
OUTPUT_DIR      = Path("output/evaluation/eval_charts")
MODELS          = ["gemma3", "gemma4", "internvl", "llava_next", "pixtral", "qwen3"]

# Demographic columns in display order
DEMO_GROUPS = [
    ("Age",       "age",        ["young adult", "middle-aged adult", "elderly"]),
    ("Gender",    "gender",     ["male", "female"]),
    ("Ethnicity", "ethnicity",  ["Asian", "African", "European", "Middle Eastern", "Latino"]),
    ("Body Type", "body_index", ["thin", "normal", "obese"]),
]

ALL_DEMO_COLS: list[tuple[str, str]] = [
    (col, val) for _, col, vals in DEMO_GROUPS for val in vals
]

# Variation row structure — same order as reference table
TABLE_GROUPS = [
    ("Skin",           "skin_irregularities", [
        ("Acne",                           "Acne"),
        ("Freckles",                       "Freckles"),
        ("Moles",                          "Moles"),
    ]),
    ("Hair\nColor",    "hair_color", [
        ("Black",                          "Black"),
        ("Blonde",                         "Blonde"),
        ("Brown",                          "Brown"),
        ("Gray",                           "Gray"),
    ]),
    ("Hair\nLength",   "hair_length", [
        ("Bald",                           "Bald"),
        ("Long",                           "Long"),
        ("Short",                          "Short"),
    ]),
    ("Hair\nStyle",    "hair_style", [
        ("Messy",                          "Messy"),
        ("Mohawk",                         "Mohawk"),
        ("Slicked back",                   "Slicked back"),
    ]),
    ("Facial\nHair",   "facial_hair_male", [
        ("Clean-shaven",                   "Clean-shaven"),
        ("Full beard",                     "Full beard"),
    ]),
    ("Makeup",         "makeup_female", [
        ("Heavy",                          "Heavy"),
        ("Light",                          "Light"),
    ]),
    ("Lip\nMakeup",    "lip_makeup_female", [
        ("Red lipstick",                   "Red lipstick"),
    ]),
    ("Tattoos",        "tattoos", [
        ("Facial tattoo",                  "Facial tattoo"),
    ]),
    ("Fashion",        "fashion_style", [
        ("Casual",                         "Casual"),
        ("Formal / Evening wear",          "Formal/Evening"),
        ("Functional / outdoor wear",      "Functional/outdoor"),
        ("Professional / Business formal", "Prof./Business"),
        ("Smart casual",                   "Smart casual"),
        ("Sporty / Athletic wear",         "Sporty/Athletic"),
        ("Streetwear",                     "Streetwear"),
        ("Vintage / Retro",                "Vintage/Retro"),
        ("Worn / Distressed clothing",     "Worn/Distressed"),
    ]),
    ("Eyewear",        "eyewear", [
        ("Sunglasses",                     "Sunglasses"),
        ("Thick-rimmed",                   "Thick-rimmed"),
    ]),
    ("Piercing",       "piercings", [
        ("Multiple",                       "Multiple"),
        ("Single nose",                    "Single nose"),
    ]),
    ("Access.",        "accessories", [
        ("Beanie",                         "Beanie"),
        ("Cap",                            "Cap"),
    ]),
]

FEMALE_ONLY = {"makeup_female", "lip_makeup_female"}
MALE_ONLY   = {"facial_hair_male"}


# ---------------------------------------------------------------------------
# Step 1 — Compute delta means
# ---------------------------------------------------------------------------

def compute_data() -> dict[tuple[str, str, str], float]:
    """Return {(var_name, demo_col, demo_val): mean_delta_across_models}."""
    # Per-model accumulation: {model: {(var_name, demo_col, demo_val): [deltas]}}
    per_model: dict[str, dict[tuple, list[float]]] = {}

    for model in MODELS:
        path = EVALUATION_ROOT / model / "paired_deltas.csv"
        if not path.exists():
            print(f"  Missing: {path}")
            continue
        print(f"  Loading {model} …")
        acc: dict[tuple, list[float]] = defaultdict(list)
        with path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                var_name = row["variation_name"]
                delta    = float(row["delta"])
                for _, demo_col, _ in DEMO_GROUPS:
                    demo_val = row[demo_col]
                    key = (var_name, demo_col, demo_val)
                    acc[key].append(delta)
        per_model[model] = {k: float(np.mean(v)) for k, v in acc.items()}

    # Average across models (equal weight per model)
    collector: dict[tuple, list[float]] = defaultdict(list)
    for model_data in per_model.values():
        for key, val in model_data.items():
            collector[key].append(val)
    return {k: float(np.mean(v)) for k, v in collector.items()}


def save_csv(data: dict, output_path: Path) -> None:
    col_headers = [f"{col}:{val}" for col, val in ALL_DEMO_COLS]
    fieldnames  = ["variation_name"] + col_headers

    with output_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for grp_display, cat_key, var_list in TABLE_GROUPS:
            for var_val, _ in var_list:
                var_name = f"{cat_key}:{var_val}"
                row: dict = {"variation_name": var_name}
                for demo_col, demo_val in ALL_DEMO_COLS:
                    col_key = f"{demo_col}:{demo_val}"
                    if cat_key in MALE_ONLY and demo_val == "female":
                        row[col_key] = ""
                    elif cat_key in FEMALE_ONLY and demo_val == "male":
                        row[col_key] = ""
                    else:
                        v = data.get((var_name, demo_col, demo_val))
                        row[col_key] = f"{v:.4f}" if v is not None else ""
                w.writerow(row)
    print(f"Saved CSV: {output_path}")


# ---------------------------------------------------------------------------
# Step 2 — Render table
# ---------------------------------------------------------------------------

def _all_neutral(cells: list) -> bool:
    """True if every non-N/A cell is neutral (|Δ| < 0.04)."""
    non_na = [v for v in cells if v is not None]
    return bool(non_na) and all(abs(v) < 0.04 for v in non_na)


def _cell_bg(v: float | None) -> str:
    if v is None:
        return "#F0F0F0"
    if v >= 0.10:
        return "#9ED89E"
    if v >= 0.04:
        return "#D4EDD4"
    if v <= -0.10:
        return "#E89888"
    if v <= -0.04:
        return "#F5CCCC"
    return "#FFFFFF"


def _cell_txt(v: float | None) -> str:
    if v is None:
        return "—"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.3f}"


def _cell_txt_color(v: float | None) -> str:
    if v is None:
        return "#888888"
    if v >= 0.10:
        return "#1B6B1B"
    if v >= 0.04:
        return "#2E7A2E"
    if v <= -0.10:
        return "#7A1010"
    if v <= -0.04:
        return "#8B2020"
    return "#111111"


def render_table(data: dict, output_path: Path, filter_white_rows: bool = False) -> None:
    # ---- Layout constants (data-coordinate units) ----
    CW_CAT  = 1.70   # category label column width
    CW_VAR  = 2.55   # variation name column width
    CW_D    = 1.65   # each data column width
    RH_HDR1 = 0.68   # demo-group header row height
    RH_HDR2 = 0.78   # demo-value header row height
    RH_DATA = 0.50   # data row height

    N_DEMO  = len(ALL_DEMO_COLS)
    TOTAL_W = CW_CAT + CW_VAR + N_DEMO * CW_D

    # Build row list (var rows only — no separate cat header rows)
    rows: list[dict] = []
    for grp_display, cat_key, var_list in TABLE_GROUPS:
        for var_val, var_display in var_list:
            var_name = f"{cat_key}:{var_val}"
            cells: list = []
            for demo_col, demo_val in ALL_DEMO_COLS:
                if cat_key in MALE_ONLY and demo_val == "female":
                    cells.append(None)
                elif cat_key in FEMALE_ONLY and demo_val == "male":
                    cells.append(None)
                else:
                    cells.append(data.get((var_name, demo_col, demo_val)))
            rows.append({"label": var_display, "cells": cells, "cat": grp_display})
    n_data = len(rows)
    TOTAL_H = RH_HDR1 + RH_HDR2 + n_data * RH_DATA
    data_top = TOTAL_H - RH_HDR1 - RH_HDR2  # y-coord of top of first data row

    SCALE_W = 0.44
    SCALE_H = 0.40
    fig_w = TOTAL_W * SCALE_W
    fig_h = TOTAL_H * SCALE_H + 0.8

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, TOTAL_W)
    ax.set_ylim(-0.90, TOTAL_H + 0.7)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    def cell_rect(x, y, w, h, color):
        ax.add_patch(mpatches.Rectangle((x, y), w, h,
                                        facecolor=color, edgecolor="none", zorder=1))

    def txt(x, y, s, fs=6.5, bold=False, italic=False, color="#111111", ha="center", va="center"):
        ax.text(x, y, s, fontsize=fs, fontweight="bold" if bold else "normal",
                fontstyle="italic" if italic else "normal",
                color=color, ha=ha, va=va, zorder=3, multialignment="center")

    def hline(y_pos, lw=0.7, color="#BBBBBB"):
        ax.plot([0, TOTAL_W], [y_pos, y_pos], color=color, lw=lw, zorder=5)

    def vline(x_pos, y_bot, y_top, lw=0.8, color="#AAAAAA"):
        ax.plot([x_pos, x_pos], [y_bot, y_top], color=color, lw=lw, zorder=5)

    # ---- Header row 1: demo group labels ----
    y = TOTAL_H
    cell_rect(0,      y - RH_HDR1, CW_CAT, RH_HDR1, "#D0D8E8")
    cell_rect(CW_CAT, y - RH_HDR1, CW_VAR, RH_HDR1, "#D0D8E8")
    txt(CW_CAT / 2,        y - RH_HDR1 / 2, "Category",  fs=7.5, bold=True)
    txt(CW_CAT + CW_VAR / 2, y - RH_HDR1 / 2, "Variation", fs=7.5, bold=True)

    HDR1_COLORS = ["#BBC9E6", "#CDD8EE", "#BBC9E6", "#CDD8EE"]
    x0 = CW_CAT + CW_VAR
    for gi, (grp_label, demo_col, demo_vals) in enumerate(DEMO_GROUPS):
        w = len(demo_vals) * CW_D
        cell_rect(x0, y - RH_HDR1, w, RH_HDR1, HDR1_COLORS[gi])
        txt(x0 + w / 2, y - RH_HDR1 / 2, grp_label, fs=7.5, bold=True)
        x0 += w
    y -= RH_HDR1

    # ---- Header row 2: demo value labels ----
    HDR2_COLORS = ["#D5DFF5", "#E6EDF9", "#D5DFF5", "#E6EDF9"]
    cell_rect(0,      y - RH_HDR2, CW_CAT, RH_HDR2, "#E0E8F4")
    cell_rect(CW_CAT, y - RH_HDR2, CW_VAR, RH_HDR2, "#E0E8F4")
    x0 = CW_CAT + CW_VAR
    for gi, (grp_label, demo_col, demo_vals) in enumerate(DEMO_GROUPS):
        for val in demo_vals:
            cell_rect(x0, y - RH_HDR2, CW_D, RH_HDR2, HDR2_COLORS[gi])
            disp = (val.replace("middle-aged adult", "mid-aged\nadult")
                       .replace("Middle Eastern", "Mid.\nEastern"))
            txt(x0 + CW_D / 2, y - RH_HDR2 / 2, disp, fs=6.3, italic=True)
            x0 += CW_D
    y -= RH_HDR2

    # ---- Data rows ----
    cat_y_top: dict[str, float] = {}
    cat_y_bot: dict[str, float] = {}

    for row in rows:
        rh  = RH_DATA
        cat_lbl = row["cat"]

        # Category col background
        cell_rect(0, y - rh, CW_CAT, rh, "#F2F2F2")
        # Variation name col background + text
        cell_rect(CW_CAT, y - rh, CW_VAR, rh, "#FAFAFA")
        txt(CW_CAT + 0.14, y - rh / 2, row["label"], fs=6.5, italic=True, ha="left", va="center")

        # Data cells
        x0 = CW_CAT + CW_VAR
        for ci in range(N_DEMO):
            v = row["cells"][ci]
            cell_rect(x0, y - rh, CW_D, rh, _cell_bg(v))
            txt(x0 + CW_D / 2, y - rh / 2, _cell_txt(v), fs=6.2, color=_cell_txt_color(v))
            x0 += CW_D

        if cat_lbl not in cat_y_top:
            cat_y_top[cat_lbl] = y
        cat_y_bot[cat_lbl] = y - rh
        y -= rh

    # table bottom is now at y ≈ 0

    # ---- Category merged labels ----
    for cat_lbl, y_top in cat_y_top.items():
        y_bot = cat_y_bot[cat_lbl]
        cell_h = y_top - y_bot
        lbl = cat_lbl if cell_h >= RH_DATA * 1.8 else cat_lbl.replace("\n", " ")
        txt(CW_CAT / 2, (y_top + y_bot) / 2, lbl, fs=6.5, bold=True, color="#222222")

    # ---- Horizontal lines ----
    hline(TOTAL_H,  lw=1.5, color="#444444")            # outer top
    hline(TOTAL_H - RH_HDR1, lw=0.6, color="#888888")  # between HDR1 and HDR2
    hline(data_top, lw=1.2, color="#555555")            # below headers / above data
    first_cat = rows[0]["cat"] if rows else None
    for cat_lbl, y_top in cat_y_top.items():
        if cat_lbl != first_cat:
            hline(y_top, lw=0.7, color="#AAAAAA")       # between category groups
    hline(0, lw=1.5, color="#444444")                   # outer bottom

    # ---- Vertical separators between demo groups (header rows only) ----
    x0 = CW_CAT + CW_VAR
    for gi, (grp_label, demo_col, demo_vals) in enumerate(DEMO_GROUPS):
        if gi > 0:
            vline(x0, data_top, TOTAL_H, lw=1.2, color="#6677AA")
        x0 += len(demo_vals) * CW_D

    # ---- Legend ----
    legend_items = [
        ("#9ED89E", "Strong pos. (Δ ≥ +0.10)"),
        ("#D4EDD4", "Moderate pos. (+0.04 ≤ Δ < +0.10)"),
        ("#FFFFFF", "Neutral (|Δ| < 0.04)"),
        ("#E89888", "Strong neg. (Δ ≤ −0.10)"),
        ("#F5CCCC", "Moderate neg. (−0.10 < Δ ≤ −0.04)"),
        ("#F0F0F0", "N/A (not applicable for gender)"),
    ]
    item_w = TOTAL_W / 3
    for i, (color, label) in enumerate(legend_items):
        lx = (i % 3) * item_w
        ly = -0.18 - (i // 3) * 0.34
        ax.add_patch(mpatches.Rectangle(
            (lx, ly - 0.20), 0.28, 0.20,
            facecolor=color, edgecolor="#AAAAAA", linewidth=0.5, zorder=3,
        ))
        ax.text(lx + 0.34, ly - 0.10, label, fontsize=6.2,
                va="center", ha="left", color="#333333", zorder=3)

    # ---- Title ----
    fig.text(0.5, 0.995,
             "Mean Δ per appearance variation × demographic group  (averaged across 6 VLMs and all scenarios)",
             ha="center", va="top", fontsize=9.0, fontweight="bold", color="#111111")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved table: {output_path}")


# ---------------------------------------------------------------------------
# LaTeX export helpers
# ---------------------------------------------------------------------------

def _latex_color_name(v: float | None) -> str:
    if v is None:
        return "naGray"
    if v >= 0.10:
        return "posstrong"
    if v >= 0.04:
        return "posmid"
    if v <= -0.10:
        return "negstrong"
    if v <= -0.04:
        return "negmid"
    return "neutral"


def _latex_fmt(v: float | None) -> str:
    if v is None:
        return r"\cellcolor{naGray}{-}"
    s = f"{v:+.3f}"
    cname = _latex_color_name(v)
    return rf"\cellcolor{{{cname}}}${s}$"


def write_latex_table(data: dict, output_path: Path) -> None:
    """Write a copy-paste-ready LaTeX table matching the PNG layout."""
    lines: list[str] = []
    lines.append(r"% Auto-generated LaTeX table from create_main_table.py")
    # color defs (RGB)
    lines.append(r"\definecolor{posstrong}{RGB}{158,216,158}")
    lines.append(r"\definecolor{posmid}{RGB}{212,237,212}")
    lines.append(r"\definecolor{negmid}{RGB}{245,204,204}")
    lines.append(r"\definecolor{negstrong}{RGB}{232,152,136}")
    lines.append(r"\definecolor{neutral}{RGB}{245,245,245}")
    lines.append(r"\definecolor{naGray}{RGB}{240,240,240}")
    lines.append("")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.1}")
    lines.append("")

    col_spec = "ll|" + "ccc|" + "cc|" + "ccccc|" + "ccc"
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Category} & \textbf{Variation} & \multicolumn{3}{c}{\textbf{Age}} & \multicolumn{2}{c}{\textbf{Gender}} & \multicolumn{5}{c}{\textbf{Ethnicity}} & \multicolumn{3}{c}{\textbf{Body}} \\")
    lines.append(r"& & YA & MA & EL & M & F & As & Af & Eu & ME & La & Th & No & Ob \\")
    lines.append(r"\midrule")

    for grp_display, cat_key, var_list in TABLE_GROUPS:
        nrows = len(var_list)
        for i, (var_val, var_display) in enumerate(var_list):
            var_name = f"{cat_key}:{var_val}"
            cells: list[str] = []
            for demo_col, demo_val in ALL_DEMO_COLS:
                if cat_key in MALE_ONLY and demo_val == "female":
                    cells.append(_latex_fmt(None))
                elif cat_key in FEMALE_ONLY and demo_val == "male":
                    cells.append(_latex_fmt(None))
                else:
                    v = data.get((var_name, demo_col, demo_val))
                    cells.append(_latex_fmt(v))

            # Build row: category cell only on first row (multirow), then variation and cells
            if i == 0:
                left = rf"\multirow{{{nrows}}}{{*}}{{{grp_display}}} & {var_display}"
            else:
                left = f"& {var_display}"
            row = left + " & " + " & ".join(cells) + " \\\\"
            lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Mean prediction shift $\Delta$ per appearance variation and demographic group, averaged across all models and scenarios. Positive values (green) indicate shifts toward the positive pole; negative (red) toward the negative pole. Abbreviations: YA = young adult, MA = middle-aged adult, EL = elderly; M = male, F = female; As = Asian, Af = African, Eu = European, ME = Middle Eastern, La = Latino; Th = thin, No = normal, Ob = obese.}")
    lines.append(r"\label{tab:main_table}")
    lines.append(r"\end{table*}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote LaTeX table: {output_path}")

# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Computing delta means …")
    data = compute_data()
    print(f"  Keys computed: {len(data)}")

    csv_path = OUTPUT_DIR / "main_table_data.csv"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_csv(data, csv_path)

    print("Rendering table …")
    render_table(data, OUTPUT_DIR / "main_table.png")

    print("Rendering filtered table (neutral-only rows removed) …")
    render_table(data, OUTPUT_DIR / "main_table_filtered.png", filter_white_rows=True)

    print("Writing LaTeX table …")
    write_latex_table(data, OUTPUT_DIR / "main_table.tex")


if __name__ == "__main__":
    main()
