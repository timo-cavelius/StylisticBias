#!/usr/bin/env python3
"""Create a sample image figure showing base faces and their variations.

Layout:
  - 3 female rows, then 3 male rows (separated by a thin gap)
  - Column 0     : base face (coloured border per gender)
  - Columns 1..N : one variation per category; fashion_style gets 3 columns
  - Last column  : ". . ." — one per row to indicate more subjects exist
  - Label row below each gender section: category names + sample values

Within each gender section every column shows the SAME variation value
(most common across the group). Blank gray cell if a folder lacks that value.

Output: output/evaluation/eval_charts/sample_image_figure.png

Usage:
  python3 src/create_sample_image_figure.py [--seed N] [--n-categories N]
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FACES_DIR      = Path("output/final_dataset")
OUTPUT_DIR     = Path("output/evaluation/eval_charts")
WHITELIST_PATH = Path("config/variation_features_whitelist.json")


def _load_whitelist() -> dict[str, set[str]]:
    with WHITELIST_PATH.open() as f:
        raw = json.load(f)
    return {cat: set(vals) for cat, vals in raw.items()}

CAT_DISPLAY = {
    "tattoos":             "Tattoos",
    "eyewear":             "Eyewear",
    "hair_style":          "Hair Style",
    "hair_color":          "Hair Color",
    "hair_length":         "Hair Length",
    "skin_irregularities": "Skin Irreg.",
    "accessories":         "Accessories",
    "piercings":           "Piercings",
    "fashion_style_0":     "Fashion Style",
    "fashion_style_1":     "Fashion Style",
    "fashion_style_2":     "Fashion Style",
}

# Column order — three fashion_style slots always last
PREFERRED_ORDER = [
    "tattoos",
    "eyewear",
    "hair_style",
    "hair_color",
    "hair_length",
    "skin_irregularities",
    "accessories",
    "piercings",
    "fashion_style_0",
    "fashion_style_1",
    "fashion_style_2",
]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _detect_gender(folder_path: Path) -> str:
    for fname in sorted(os.listdir(folder_path)):
        if fname.endswith("_metadata.json"):
            with (folder_path / fname).open() as f:
                prompt = json.load(f).get("prompt", "")
            return "female" if "female" in prompt else "male"
    return "unknown"


_WHITELIST = _load_whitelist()


def _load_all_variations(folder_path: Path) -> dict[str, list[tuple[str, Path]]]:
    """Return {category: [(val, path), ...]} — whitelisted images only, sorted by val."""
    cats: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for fname in sorted(os.listdir(folder_path)):
        if not fname.endswith("_metadata.json"):
            continue
        with (folder_path / fname).open() as f:
            data = json.load(f)
        var      = data.get("characteristics", {}).get("variation", {})
        img_path = folder_path / fname.replace("_metadata.json", ".png")
        if not img_path.exists():
            continue
        for cat, val in var.items():
            allowed = _WHITELIST.get(cat)
            if allowed is not None and val not in allowed:
                continue
            cats[cat].append((val, img_path))
    return {cat: sorted(items, key=lambda x: x[0]) for cat, items in cats.items()}


def _canonical_variations(
    folders: list[Path],
    all_variations: dict[Path, dict[str, list[tuple[str, Path]]]],
    categories: list[str],
    rng: random.Random,
) -> dict[Path, dict[str, tuple[str, Path]]]:
    """Build per-folder variation dict so all folders share the same variation
    value per column (randomly chosen from the values available in the group).

    fashion_style_0/1/2 → 3 randomly chosen distinct fashion values.
    """
    fashion_slots = [c for c in categories if c.startswith("fashion_style_")]
    regular_cats  = [c for c in categories if not c.startswith("fashion_style_")]

    # Regular categories: randomly pick one value that exists in the group
    canonical_regular: dict[str, str] = {}
    for cat in regular_cats:
        vals = sorted({val for f in folders for val, _ in all_variations[f].get(cat, [])})
        if vals:
            canonical_regular[cat] = rng.choice(vals)

    # Fashion style slots: randomly pick N distinct values from those available
    fashion_vals = sorted({val for f in folders for val, _ in all_variations[f].get("fashion_style", [])})
    fashion_canonical = rng.sample(fashion_vals, min(len(fashion_slots), len(fashion_vals)))

    # Build per-folder result
    result: dict[Path, dict[str, tuple[str, Path]]] = {}
    for f in folders:
        fvars: dict[str, tuple[str, Path]] = {}

        for cat, cval in canonical_regular.items():
            match = next(
                (item for item in all_variations[f].get(cat, []) if item[0] == cval),
                None,
            )
            if match:
                fvars[cat] = match

        fashion_map = {val: path for val, path in all_variations[f].get("fashion_style", [])}
        for i, cval in enumerate(fashion_canonical):
            slot = f"fashion_style_{i}"
            if cval in fashion_map:
                fvars[slot] = (cval, fashion_map[cval])

        result[f] = fvars
    return result


def _base_image(folder_path: Path) -> Path:
    return folder_path / (folder_path.name + ".png")


def discover_folders(faces_dir: Path) -> tuple[list[Path], list[Path]]:
    female, male = [], []
    for entry in sorted(faces_dir.iterdir()):
        if not entry.is_dir():
            continue
        g = _detect_gender(entry)
        (female if g == "female" else male if g == "male" else []).append(entry)
    return female, male


def select_folders(
    female: list[Path], male: list[Path],
    n: int = 3, seed: int | None = None,
) -> tuple[list[Path], list[Path]]:
    rng = random.Random(seed)
    return rng.sample(female, n), rng.sample(male, n)


def find_categories(
    all_folders: list[Path],
    all_variations: dict[Path, dict[str, list[tuple[str, Path]]]],
    preferred_order: list[str],
    max_cats: int,
) -> list[str]:
    """Union of all categories across folders, ordered by PREFERRED_ORDER."""
    union: set[str] = set()
    for f in all_folders:
        for cat, items in all_variations[f].items():
            if cat == "fashion_style":
                for i in range(min(len(items), 3)):
                    union.add(f"fashion_style_{i}")
            else:
                union.add(cat)
    ordered = [c for c in preferred_order if c in union]
    extras  = sorted(union - set(ordered))
    return (ordered + extras)[:max_cats]


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def _load_img(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def _blank_ax(ax: plt.Axes, facecolor: str = "#F4F4F4") -> None:
    ax.set_facecolor(facecolor)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.4); sp.set_edgecolor("#dddddd")


def create_figure(
    female_folders: list[Path],
    male_folders:   list[Path],
    categories:     list[str],
    variations:     dict[Path, dict[str, tuple[str, Path]]],
    output_path:    Path,
) -> None:
    n_faces    = len(female_folders)
    n_var_cols = len(categories)
    n_cols     = 1 + n_var_cols + 1   # base | vars | dots

    img_h    = 1.65
    label_h  = 0.62
    spacer_h = 0.30

    fig_w = n_cols * 1.65 + 0.2
    fig_h = 2 * (n_faces * img_h + label_h) + spacer_h

    height_ratios = (
        [img_h] * n_faces + [label_h] +
        [spacer_h] +
        [img_h] * n_faces + [label_h]
    )
    width_ratios = [1.0] * (1 + n_var_cols) + [0.45]

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    gs  = gridspec.GridSpec(
        len(height_ratios), n_cols,
        figure=fig,
        height_ratios=height_ratios,
        width_ratios=width_ratios,
        hspace=0.01,
        wspace=0.0,
        left=0.01, right=0.995,
        top=0.97,  bottom=0.02,
    )

    f_img_start = 0
    f_label_row = n_faces
    spacer_row  = n_faces + 1
    m_img_start = n_faces + 2
    m_label_row = n_faces + 2 + n_faces

    col_headers = ["Base"] + [CAT_DISPLAY.get(c, c) for c in categories]

    for gender, folders, img_start, label_row in [
        ("female", female_folders, f_img_start, f_label_row),
        ("male",   male_folders,   m_img_start, m_label_row),
    ]:
        color = "#E8A020" if gender == "female" else "#4A80B0"

        # ---- image rows ----
        for row_idx, folder in enumerate(folders):
            gs_row   = img_start + row_idx
            var_data = variations[folder]

            # Base image
            ax0 = fig.add_subplot(gs[gs_row, 0])
            ax0.imshow(_load_img(_base_image(folder)), aspect="auto")
            ax0.set_xticks([]); ax0.set_yticks([])
            for sp in ax0.spines.values():
                sp.set_linewidth(1.6); sp.set_edgecolor(color)
            if row_idx == 0:
                ax0.set_title(
                    gender.capitalize(), fontsize=12,
                    fontweight="bold", color=color, pad=4,
                )

            # Variation images
            for col_idx, cat in enumerate(categories, start=1):
                ax = fig.add_subplot(gs[gs_row, col_idx])
                if cat in var_data:
                    ax.imshow(_load_img(var_data[cat][1]), aspect="auto")
                    ax.set_xticks([]); ax.set_yticks([])
                    for sp in ax.spines.values():
                        sp.set_linewidth(0.4); sp.set_edgecolor("#cccccc")
                else:
                    _blank_ax(ax)

            # ". . ." — horizontal, black, bold
            ax_dots = fig.add_subplot(gs[gs_row, -1])
            ax_dots.axis("off")
            ax_dots.text(
                0.5, 0.5, ". . .",
                ha="center", va="center",
                fontsize=11, color="black",
                fontweight="bold",
                transform=ax_dots.transAxes,
            )

        # ---- category label row ----
        for col_idx, header in enumerate(col_headers):
            ax = fig.add_subplot(gs[label_row, col_idx])
            ax.axis("off")
            ax.text(
                0.5, 0.92, header,
                ha="center", va="top",
                fontsize=9.5,
                fontweight="bold" if col_idx == 0 else "normal",
                color="#333333",
                transform=ax.transAxes,
            )
            if col_idx > 0:
                cat = categories[col_idx - 1]
                vals = [
                    variations[f][cat][0]
                    for f in folders if cat in variations[f]
                ]
                if vals:
                    unique = list(dict.fromkeys(vals))
                    hint   = " / ".join(unique[:2]) + (" …" if len(unique) > 2 else "")
                    ax.text(
                        0.5, 0.38, hint,
                        ha="center", va="top",
                        fontsize=7, color="#999999",
                        style="italic",
                        transform=ax.transAxes,
                    )
        ax = fig.add_subplot(gs[label_row, -1])
        ax.axis("off")

    fig.add_subplot(gs[spacer_row, :]).axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",         type=int,  default=None)
    parser.add_argument("--n-faces",      type=int,  default=3)
    parser.add_argument("--n-categories", type=int,  default=11)
    parser.add_argument("--faces-dir",    type=Path, default=FACES_DIR)
    parser.add_argument("--output-dir",   type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    print("Scanning face folders …")
    female_all, male_all = discover_folders(args.faces_dir)
    print(f"  Female: {len(female_all)}  Male: {len(male_all)}")

    female_sel, male_sel = select_folders(
        female_all, male_all, n=args.n_faces, seed=args.seed,
    )
    all_sel = female_sel + male_sel
    print(f"Selected female: {[f.name for f in female_sel]}")
    print(f"Selected male:   {[f.name for f in male_sel]}")

    print("Loading variation metadata …")
    all_variations = {f: _load_all_variations(f) for f in all_sel}

    categories = find_categories(all_sel, all_variations, PREFERRED_ORDER, args.n_categories)
    print(f"Categories ({len(categories)}): {categories}")

    # Align per gender: randomly chosen variation value per column within each group
    rng = random.Random(args.seed)
    variations_f = _canonical_variations(female_sel, all_variations, categories, rng)
    variations_m = _canonical_variations(male_sel,   all_variations, categories, rng)
    variations   = {**variations_f, **variations_m}

    create_figure(
        female_sel, male_sel,
        categories, variations,
        args.output_dir / "sample_image_figure.png",
    )


if __name__ == "__main__":
    main()
