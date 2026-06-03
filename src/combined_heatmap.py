#!/usr/bin/env python3
"""Combined heatmap: base-face preferences + male deltas + female deltas.

Three panels side by side, sharing the same 25-scenario x-axis.
Space saved vs three separate figures:
  - single suptitle
  - x-axis tick labels shown only on middle panel
  - one shared colorbar for male/female delta panels
  - one colorbar for base-face P(option A) panel

Usage:
  python3 src/combined_heatmap.py --model gemma3
  python3 src/combined_heatmap.py          # all models
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

EVALUATION_ROOT = Path("output/evaluation")
OUTPUT_DIR      = Path("output/evaluation/eval_charts")

MODEL_DISPLAY = {
    "gemma3":    "Gemma-3",
    "gemma4":    "Gemma-4",
    "internvl":  "InternVL3",
    "llava_next":"LLaVA-v1.6",
    "pixtral":   "Pixtral",
    "qwen3":     "Qwen3-VL",
}

# Canonical category order for variation rows
CATEGORY_ORDER = [
    ("Accessories",  "accessories"),
    ("Eyewear",      "eyewear"),
    ("Facial Hair",  "facial_hair_male"),
    ("Fashion",      "fashion_style"),
    ("Hair Color",   "hair_color"),
    ("Hair Length",  "hair_length"),
    ("Hair Style",   "hair_style"),
    ("Lip Makeup",   "lip_makeup_female"),
    ("Makeup",       "makeup_female"),
    ("Piercings",    "piercings"),
    ("Skin",         "skin_irregularities"),
    ("Tattoos",      "tattoos"),
]
FEMALE_ONLY = {"makeup_female", "lip_makeup_female"}
MALE_ONLY   = {"facial_hair_male"}

# Canonical order for base-face demographic rows
DEMO_ROW_ORDER = [
    ("body_type", "normal"),
    ("body_type", "obese"),
    ("body_type", "thin"),
    ("ethnicity", "Asian"),
    ("ethnicity", "African"),
    ("ethnicity", "European"),
    ("ethnicity", "Middle Eastern"),
    ("ethnicity", "Latino"),
    ("gender", "male"),
    ("gender", "female"),
    ("age", "young adult"),
    ("age", "middle-aged adult"),
    ("age", "elderly"),
]
DEMO_ROW_LABELS = {
    ("body_type", "normal"):         "body: normal",
    ("body_type", "obese"):          "body: obese",
    ("body_type", "thin"):           "body: thin",
    ("ethnicity", "Asian"):          "ethn.: Asian",
    ("ethnicity", "African"):        "ethn.: African",
    ("ethnicity", "European"):       "ethn.: European",
    ("ethnicity", "Middle Eastern"): "ethn.: Mid. Eastern",
    ("ethnicity", "Latino"):         "ethn.: Latino",
    ("gender", "male"):              "gender: male",
    ("gender", "female"):            "gender: female",
    ("age", "young adult"):          "age: young",
    ("age", "middle-aged adult"):    "age: mid-aged",
    ("age", "elderly"):              "age: elderly",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_scenario_labels(model_dir: Path) -> dict[int, str]:
    labels: dict[int, str] = {}
    with (model_dir / "paired_deltas.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = int(row["scenario"])
            if sid not in labels:
                labels[sid] = f"{row['option_a']} / {row['option_b']}"
    return dict(sorted(labels.items()))


def load_base_matrix(model_dir: Path, scenario_ids: list[int]):
    """→ (matrix [n_rows × n_scen], row_labels)"""
    data: dict[tuple, dict[int, float]] = defaultdict(dict)
    with (model_dir / "base_faces_category_scenario_heatmap_values.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["category_type"], row["category_value"])
            data[key][int(row["scenario"])] = float(row["mean_p_option_a"])

    rows = [k for k in DEMO_ROW_ORDER if k in data]
    mat  = np.full((len(rows), len(scenario_ids)), np.nan)
    for ri, key in enumerate(rows):
        for ci, sid in enumerate(scenario_ids):
            mat[ri, ci] = data[key].get(sid, np.nan)
    labels = [DEMO_ROW_LABELS.get(k, f"{k[0]}:{k[1]}") for k in rows]
    return mat, labels


def load_variation_matrix(model_dir: Path, gender: str, scenario_ids: list[int]):
    """→ (matrix, row_labels, cat_sep_positions, cat_label_list)"""
    acc: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    with (model_dir / "paired_deltas.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["gender"] != gender:
                continue
            acc[row["variation_name"]][int(row["scenario"])].append(float(row["delta"]))

    row_order: list[tuple[str, str]] = []   # (var_name, short_label)
    cat_seps:  list[int]             = []   # row indices where a new category starts
    cat_spans: list[tuple[int, int, str]] = []  # (start_row, end_row, cat_display)

    for cat_display, cat_key in CATEGORY_ORDER:
        if gender == "male"   and cat_key in FEMALE_ONLY: continue
        if gender == "female" and cat_key in MALE_ONLY:   continue
        cat_vars = sorted(v for v in acc if v.split(":")[0].strip().lower() == cat_key)
        if not cat_vars:
            continue
        if row_order:
            cat_seps.append(len(row_order))
        start = len(row_order)
        for v in cat_vars:
            short = v.split(":", 1)[1].strip() if ":" in v else v
            row_order.append((v, short))
        cat_spans.append((start, len(row_order) - 1, cat_display))

    mat = np.full((len(row_order), len(scenario_ids)), np.nan)
    for ri, (var, _) in enumerate(row_order):
        for ci, sid in enumerate(scenario_ids):
            vals = acc[var].get(sid, [])
            if vals:
                mat[ri, ci] = float(np.mean(vals))

    labels = [short for _, short in row_order]
    return mat, labels, cat_seps, cat_spans


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _draw_heatmap(ax, mat, row_labels, cmap, vmin, vmax,
                  scen_labels, show_xlabels, cat_seps=None, cat_spans=None):
    im = ax.imshow(mat, aspect="auto", cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    n_rows, n_scen = mat.shape

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=7.5)
    ax.set_xticks(range(n_scen))
    if show_xlabels:
        ax.set_xticklabels(scen_labels, rotation=45, ha="right", fontsize=7)
    else:
        ax.set_xticklabels([], visible=False)
    ax.tick_params(axis="both", length=0)

    # Category separator lines and labels for variation panels
    if cat_seps:
        for b in cat_seps:
            ax.axhline(b - 0.5, color="#666", linewidth=0.8, alpha=0.8)
    if cat_spans:
        for start, end, cat in cat_spans:
            ax.text(n_scen + 0.3, (start + end) / 2, cat,
                    va="center", ha="left", fontsize=7, color="#555555",
                    transform=ax.transData, clip_on=False)
    return im


def plot_combined(model: str, out_path: Path) -> None:
    model_dir   = EVALUATION_ROOT / model
    scen_labels_map = load_scenario_labels(model_dir)
    scenario_ids    = sorted(scen_labels_map.keys())
    scen_tick       = [scen_labels_map[s] for s in scenario_ids]

    base_mat,   base_rows                             = load_base_matrix(model_dir, scenario_ids)
    male_mat,   male_rows,   male_seps,   male_spans  = load_variation_matrix(model_dir, "male",   scenario_ids)
    female_mat, female_rows, female_seps, female_spans= load_variation_matrix(model_dir, "female", scenario_ids)

    n_base, n_male, n_female = base_mat.shape[0], male_mat.shape[0], female_mat.shape[0]
    n_scen = len(scenario_ids)

    # Symmetric delta limit shared across male/female
    delta_lim = float(max(np.nanmax(np.abs(male_mat)), np.nanmax(np.abs(female_mat))))
    delta_lim = min(round(delta_lim + 0.02, 2), 0.90)

    # ---- Figure sizing ----
    CELL_H  = 0.26   # inches per row
    CELL_W  = 0.36   # inches per scenario column
    YLABEL_W = 1.65  # inches for y-tick labels
    CB_W    = 0.18   # colorbar width inches
    CB_PAD  = 0.08
    HGAP    = 0.55   # horizontal gap between panels (inches)

    panel_w  = n_scen * CELL_W
    n_rows   = max(n_base, n_male, n_female)
    fig_h    = n_rows * CELL_H + 2.8   # +2.8 for title + x-labels
    fig_w    = (3 * YLABEL_W + 3 * panel_w + 2 * HGAP
                + 2 * (CB_W + CB_PAD) + 0.5)

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    # Fractional positions
    l_frac  = 0.01
    r_frac  = 0.99
    t_frac  = 0.92
    b_frac  = 0.18   # leaves room for rotated x-labels

    total_w = r_frac - l_frac
    cb_frac = CB_W / fig_w
    gap_frac = HGAP / fig_w
    yl_frac  = YLABEL_W / fig_w
    pw_frac  = panel_w / fig_w

    # Place axes manually for tight control
    # Order left→right: [ylabel | panel_base | cb1 | gap | ylabel | panel_male | gap | ylabel | panel_female | cb2]
    x0 = l_frac + yl_frac

    x_base   = x0
    x_cb1    = x_base + pw_frac + 0.005
    x_male   = x_cb1  + cb_frac + CB_PAD/fig_w + gap_frac + yl_frac
    x_female = x_male + pw_frac + gap_frac + yl_frac
    x_cb2    = x_female + pw_frac + 0.005

    panel_h = t_frac - b_frac

    ax_base   = fig.add_axes([x_base,   b_frac, pw_frac, panel_h])
    ax_cb1    = fig.add_axes([x_cb1,    b_frac, cb_frac, panel_h])
    ax_male   = fig.add_axes([x_male,   b_frac, pw_frac, panel_h])
    ax_female = fig.add_axes([x_female, b_frac, pw_frac, panel_h])
    ax_cb2    = fig.add_axes([x_cb2,    b_frac, cb_frac, panel_h])

    # ---- Draw panels ----
    im1 = _draw_heatmap(ax_base, base_mat, base_rows,
                        "RdYlBu_r", 0.0, 1.0,
                        scen_tick, show_xlabels=True)

    im2 = _draw_heatmap(ax_male, male_mat, male_rows,
                        "RdYlBu", -delta_lim, delta_lim,
                        scen_tick, show_xlabels=True,
                        cat_seps=male_seps, cat_spans=male_spans)

    im3 = _draw_heatmap(ax_female, female_mat, female_rows,
                        "RdYlBu", -delta_lim, delta_lim,
                        scen_tick, show_xlabels=True,
                        cat_seps=female_seps, cat_spans=female_spans)

    # ---- Titles ----
    ax_base.set_title("Base Faces\n$P(\\mathrm{option\\ A})$",
                      fontsize=9.5, fontweight="bold", pad=5)
    ax_male.set_title("Variation Deltas — Male\nMean $\\Delta$",
                      fontsize=9.5, fontweight="bold", pad=5)
    ax_female.set_title("Variation Deltas — Female\nMean $\\Delta$",
                        fontsize=9.5, fontweight="bold", pad=5)

    # ---- Colorbars ----
    cb1 = plt.colorbar(im1, cax=ax_cb1)
    cb1.set_label("$P(\\mathrm{option\\ A})$", fontsize=8)
    cb1.ax.tick_params(labelsize=7.5)

    cb2 = plt.colorbar(im3, cax=ax_cb2)
    cb2.set_label("Mean $\\Delta$", fontsize=8)
    cb2.ax.tick_params(labelsize=7.5)

    # ---- Suptitle ----
    name = MODEL_DISPLAY.get(model, model)
    fig.text(0.5, 0.97,
             f"{name} — Base-Face Preferences and Variation Delta Shifts",
             ha="center", va="top",
             fontsize=12, fontweight="bold", color="#111111")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None,
                        help="Model folder name, e.g. gemma3. Omit for all.")
    args = parser.parse_args()

    if args.model:
        models = [args.model]
    else:
        models = [
            d.name for d in sorted(EVALUATION_ROOT.iterdir())
            if d.is_dir()
            and not d.name.startswith("model_comparison")
            and (d / "paired_deltas.csv").exists()
        ]

    for model in models:
        print(f"Processing {model} …")
        plot_combined(model, OUTPUT_DIR / f"combined_heatmap_{model}.png")


if __name__ == "__main__":
    main()
