#!/usr/bin/env python3
"""Improved variation-level whisker chart, split by gender.

Two stacked panels (Female / Male). For each variation:
  - Dot    = mean delta across all paired observations
  - Lower whisker = mean of positive clipped negative deltas  (neg tendency)
  - Upper whisker = mean of positive clipped positive deltas  (pos tendency)

This is the moved implementation from the repository root.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory


EVALUATION_ROOT = Path("output/evaluation")
OUTPUT_DIR      = Path("output/evaluation/eval_charts")

MODEL_ORDER = ["gemma3", "gemma4", "internvl", "llava_next", "pixtral", "qwen3"]

MODEL_DISPLAY = {
    "gemma3":    "Gemma-3",
    "gemma4":    "Gemma-4",
    "internvl":  "InternVL3",
    "llava_next": "LLaVA-1.6",
    "pixtral":   "Pixtral",
    "qwen3":     "Qwen3-VL",
}

MODEL_COLORS = {
    "gemma3":    "#0072B2",
    "gemma4":    "#E69F00",
    "internvl":  "#D55E00",
    "llava_next": "#009E73",
    "pixtral":   "#CC79A7",
    "qwen3":     "#56B4E9",
}

CATEGORY_ORDER = [
    "skin_irregularities",
    "hair_color",
    "hair_length",
    "hair_style",
    "facial_hair_male",
    "makeup_female",
    "lip_makeup_female",
    "tattoos",
    "fashion_style",
    "eyewear",
    "piercings",
    "accessories",
]

CATEGORY_DISPLAY = {
    "skin_irregularities": "Skin",
    "hair_color":          "Hair color",
    "hair_length":         "Hair length",
    "hair_style":          "Hair style",
    "facial_hair_male":    "Facial hair",
    "makeup_female":       "Makeup",
    "lip_makeup_female":   "Lip makeup",
    "tattoos":             "Tattoos",
    "fashion_style":       "Fashion",
    "eyewear":             "Eyewear",
    "piercings":           "Piercings",
    "accessories":         "Accessories",
}

GENDERS = ["female", "male"]

# ---- Sizes / weights — tune for readability ----
MARKER_SIZE       = 9.5
MARKER_EDGE_W     = 1.4
WHISKER_LINEWIDTH = 3.2
CAP_SIZE          = 6.0
CAP_THICK         = 2.8
JITTER_SPAN       = 0.44

FS_SUPTITLE   = 26
FS_SUBTITLE   = 14
FS_PANEL      = 22
FS_Y_LABEL    = 20
FS_Y_TICKS    = 15
FS_X_TICKS    = 15
FS_CAT_LABEL  = 12
FS_LEGEND     = 17
FS_FOOTNOTE   = 13

SIGNED_CMAP = LinearSegmentedColormap.from_list("signed_bg", ["#F1C9C9", "#FCFCFD", "#CDEBD3"])


def _available_models(root: Path) -> list[str]:
    found = [m for m in MODEL_ORDER if (root / m / "paired_deltas.csv").exists()]
    extras = [
        e.name for e in sorted(root.iterdir())
        if e.is_dir()
        and e.name not in found
        and not e.name.startswith("model_comparison")
        and (e / "paired_deltas.csv").exists()
    ]
    return found + extras


def _sort_key(var: str) -> tuple[int, str, str]:
    cat, _, val = var.partition(":" )
    cat = cat.strip().lower()
    try:
        idx = CATEGORY_ORDER.index(cat)
    except ValueError:
        idx = len(CATEGORY_ORDER)
    return idx, cat, val.strip().lower()


def _variation_label(var: str) -> str:
    cat, _, val = var.partition(":")
    cat = cat.strip().lower()
    val = val.strip()
    if cat == "fashion_style" and "/" in val:
        val = val.split("/", 1)[1].strip()
    return val if val else CATEGORY_DISPLAY.get(cat, cat)


def load_delta_arrays(root: Path, models: list[str]):
    bucket: dict[str, dict[str, dict[str, list[float]]]] = {
        m: {g: defaultdict(list) for g in GENDERS} for m in models
    }
    all_vars: set[str] = set()

    for model in models:
        csv_path = root / model / "paired_deltas.csv"
        if not csv_path.exists():
            continue
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                g = row.get("gender", "").strip().lower()
                if g not in GENDERS:
                    continue
                var = row.get("variation_name", "").strip()
                if not var:
                    continue
                try:
                    delta = float(row["delta"])
                except (KeyError, ValueError):
                    continue
                bucket[model][g][var].append(delta)
                all_vars.add(var)

    var_order = sorted(all_vars, key=_sort_key)
    out = {m: {g: {v: np.array(bucket[m][g].get(v, []), dtype=float)
                   for v in var_order} for g in GENDERS} for m in models}
    return var_order, out


def summarize(deltas, models, variations):
    summary = {m: {g: {} for g in GENDERS} for m in models}
    for m in models:
        for g in GENDERS:
            for v in variations:
                arr = deltas[m][g][v]
                if arr.size == 0:
                    summary[m][g][v] = (np.nan, np.nan, np.nan, 0)
                else:
                    summary[m][g][v] = (
                        float(np.mean(arr)),
                        float(np.mean(np.clip(-arr, 0, None))),
                        float(np.mean(np.clip(arr,  0, None))),
                        int(arr.size),
                    )
    return summary


def _y_limits(summary, models, variations):
    vals = []
    for m in models:
        for g in GENDERS:
            for v in variations:
                mean, neg, pos, n = summary[m][g][v]
                if n == 0 or not np.isfinite(mean):
                    continue
                vals.append(mean - (neg if np.isfinite(neg) else 0))
                vals.append(mean + (pos if np.isfinite(pos) else 0))
    if not vals:
        return -0.2, 0.2
    bound = max(abs(min(vals)), abs(max(vals)), 0.05) * 1.22
    return -bound, bound


def _cat_spans(variations: list[str]) -> list[tuple[str, int, int]]:
    spans, start, cur = [], 0, variations[0].split(":", 1)[0].strip().lower()
    for i, v in enumerate(variations[1:], 1):
        cat = v.split(":", 1)[0].strip().lower()
        if cat != cur:
            spans.append((cur, start, i - 1))
            cur, start = cat, i
    spans.append((cur, start, len(variations) - 1))
    return spans


def _cat_boundaries(variations: list[str]) -> list[int]:
    bounds, prev = [], None
    for i, v in enumerate(variations):
        cat = v.split(":", 1)[0].strip().lower()
        if prev and cat != prev:
            bounds.append(i)
        prev = cat
    return bounds


def _gender_sets(summary, models, variations):
    f_vars, m_vars = set(), set()
    for m in models:
        for v in variations:
            if summary[m]["female"][v][3] > 0:
                f_vars.add(v)
            if summary[m]["male"][v][3] > 0:
                m_vars.add(v)
    return f_vars, m_vars


def plot(summary, models, variations, output_path: Path) -> None:
    f_set, m_set = _gender_sets(summary, models, variations)
    shared     = [v for v in variations if v in f_set and v in m_set]
    f_only     = [v for v in variations if v in f_set and v not in m_set]
    m_only     = [v for v in variations if v in m_set and v not in f_set]

    panel_vars = {
        "female": shared + f_only,
        "male":   shared + m_only,
    }

    n_max     = max(len(panel_vars["female"]), len(panel_vars["male"]))
    fig_h     = max(22.0, 0.52 * n_max + 9.0)
    fig_w     = 22.0

    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h), sharey=False)
    fig.patch.set_facecolor("white")

    x_min, x_max = _y_limits(summary, models, variations)
    offsets = np.linspace(-JITTER_SPAN / 2, JITTER_SPAN / 2, len(models))

    for panel_idx, gender in enumerate(GENDERS):
        ax       = axes[panel_idx]
        pvars    = panel_vars[gender]
        n_pvars  = len(pvars)
        y        = np.arange(n_pvars, dtype=float)
        y_lo, y_hi = -0.7, n_pvars - 0.3

        ax.set_facecolor("#FAFAFA")

        ax.imshow(
            np.linspace(0, 1, 512).reshape(1, -1),
            extent=[x_min, x_max, y_lo, y_hi],
            cmap=SIGNED_CMAP,
            origin="lower",
            aspect="auto",
            interpolation="bicubic",
            alpha=0.55,
            zorder=-4,
        )

        left_label_x = x_min + 0.03 * (x_max - x_min)
        for gi, (cat, s, e) in enumerate(_cat_spans(pvars)):
            ax.axhspan(
                s - 0.5, e + 0.5,
                facecolor="#E8EEF3" if gi % 2 == 0 else "#F5F7F9",
                edgecolor="#C0C8D0",
                linewidth=0.8,
                alpha=0.40 if gi % 2 == 0 else 0.15,
                zorder=-2,
            )
            ax.text(
                left_label_x, (s + e) / 2,
                CATEGORY_DISPLAY.get(cat, cat.replace("_", " ").title()),
                ha="left", va="center",
                fontsize=FS_CAT_LABEL, color="#8A96A2",
                zorder=2,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.65),
            )

        for b in _cat_boundaries(pvars):
            ax.axhline(b - 0.5, color="#B8BFC7", linewidth=1.1, alpha=0.9, zorder=0)

        n_shared = len(shared)
        specific = f_only if gender == "female" else m_only
        if specific and n_shared > 0:
            label_txt = "Female-specific" if gender == "female" else "Male-specific"
            ax.axhline(
                n_shared - 0.5,
                color="#5A5A5A", linewidth=2.2, linestyle="--",
                alpha=0.65, zorder=2,
            )
            specific_center = n_shared + (len(specific) - 1) / 2
            ax.text(
                x_max * 0.97, specific_center,
                label_txt,
                ha="right", va="center",
                fontsize=12, color="#5A5A5A", style="italic",
                zorder=5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BBBBBB",
                          linewidth=0.8, alpha=0.85),
            )

        ax.axvline(0.0, color="#2D2D2D", linewidth=1.4, alpha=0.9, zorder=1)

        for mi, model in enumerate(models):
            color   = MODEL_COLORS.get(model, "#444444")
            y_pos   = y + offsets[mi]
            means   = np.array([summary[model][gender][v][0] for v in pvars])
            negs    = np.array([summary[model][gender][v][1] for v in pvars])
            poss    = np.array([summary[model][gender][v][2] for v in pvars])
            valid   = np.array([summary[model][gender][v][3] > 0 for v in pvars])

            negs = np.where(np.isfinite(negs), negs, 0.0)
            poss = np.where(np.isfinite(poss), poss, 0.0)

            if valid.any():
                ax.errorbar(
                    means[valid],
                    y_pos[valid],
                    xerr=np.vstack([negs[valid], poss[valid]]),
                    fmt="o",
                    ms=MARKER_SIZE,
                    mfc=color,
                    mec="white",
                    mew=MARKER_EDGE_W,
                    ecolor=color,
                    elinewidth=WHISKER_LINEWIDTH,
                    capsize=CAP_SIZE,
                    capthick=CAP_THICK,
                    alpha=0.92,
                    zorder=4,
                )

        ax.set_ylim(y_lo, y_hi)
        ax.set_xlim(x_min, x_max)
        ax.invert_yaxis()
        ax.xaxis.grid(True, linestyle="--", linewidth=0.9, color="#CCCCCC", alpha=0.9)
        ax.yaxis.grid(False)

        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#AAAAAA")
        ax.spines["bottom"].set_color("#AAAAAA")
        ax.tick_params(axis="both", length=0)

        ax.set_xlabel(r"Delta shift ($\Delta_i$)", fontsize=FS_Y_LABEL, fontweight="bold", labelpad=10)
        ax.set_title(
            "Female  ♀" if gender == "female" else "Male  ♂",
            fontsize=FS_PANEL, fontweight="semibold", pad=12,
        )
        ax.tick_params(axis="x", labelsize=FS_Y_TICKS)
        ax.set_yticks(y)
        ax.set_yticklabels(
            [_variation_label(v) for v in pvars],
            fontsize=FS_X_TICKS,
        )

    fig.text(
        0.5, 0.978,
        "Variation-Level Delta Shifts Across Models — Split by Gender",
        ha="center", va="top",
        fontsize=FS_SUPTITLE, fontweight="bold", color="#1A1A1A",
    )
    fig.text(
        0.5, 0.950,
        "Dot = mean Δ  ·  Left whisker = negative-shift tendency  ·  "
        "Right whisker = positive-shift tendency",
        ha="center", va="top",
        fontsize=FS_LEGEND, color="#555555",
    )

    handles = [
        Line2D(
            [0], [0],
            marker="o", linestyle="-",
            color=MODEL_COLORS.get(m, "#444"),
            markerfacecolor=MODEL_COLORS.get(m, "#444"),
            markeredgecolor="white", markeredgewidth=1.0,
            linewidth=2.2, markersize=10,
            label=MODEL_DISPLAY.get(m, m),
        )
        for m in models
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=len(models),
        frameon=False,
        fontsize=FS_LEGEND,
        handlelength=2.2,
        columnspacing=1.6,
    )

    fig.text(
        0.5, 0.022,
        "Background: red region = negative-associated direction  ·  "
        "green region = positive-associated direction",
        ha="center", fontsize=FS_LEGEND, color="#777777",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0.0, 0.09, 1.0, 0.92])
    fig.savefig(output_path, dpi=280, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    models = _available_models(EVALUATION_ROOT)
    if not models:
        raise RuntimeError("No model folders with paired_deltas.csv found.")
    print(f"Models: {models}")

    variations, deltas = load_delta_arrays(EVALUATION_ROOT, models)
    summary            = summarize(deltas, models, variations)
    print(f"Variations: {len(variations)}")

    plot(summary, models, variations, OUTPUT_DIR / "whisker_full_variations.png")


if __name__ == "__main__":
    main()
