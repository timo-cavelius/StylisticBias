#!/usr/bin/env python3
"""Create a gender-split, variation-level grouped boxplot across models.

Design goals:
- Female panel on top, Male panel below
- Variation labels shown under BOTH panels
- Only variations used by that gender are shown in that gender's panel
- Grouped boxplots per variation (one box per model)
- Boxplot statistics shown explicitly via matplotlib defaults:
  median, Q1/Q3, whiskers to extremes (min/max)
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


MODEL_ORDER = ["gemma3", "gemma4", "llava_next", "pixtral", "qwen3"]
MODEL_DISPLAY = {
    "gemma3": "Gemma 3",
    "gemma4": "Gemma 4",
    "llava_next": "LLaVA 1.6",
    "pixtral": "Pixtral",
    "qwen3": "Qwen 3",
}

# Publication-friendly and colorblind-safe palette.
MODEL_COLORS = {
    "gemma3": "#0072B2",
    "gemma4": "#E69F00",
    "llava_next": "#009E73",
    "pixtral": "#CC79A7",
    "qwen3": "#56B4E9",
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
    "hair_color": "Hair color",
    "hair_length": "Hair length",
    "hair_style": "Hair style",
    "facial_hair_male": "Facial hair",
    "makeup_female": "Makeup",
    "lip_makeup_female": "Lip makeup",
    "tattoos": "Tattoos",
    "fashion_style": "Fashion",
    "eyewear": "Eyewear",
    "piercings": "Piercings",
    "accessories": "Accessories",
}

GENDERS = ["female", "male"]

SIGNED_Y_CMAP = LinearSegmentedColormap.from_list(
    "signed_y_bg",
    ["#F1C9C9", "#FCFCFD", "#CDEBD3"],
)


def _available_models(evaluation_root: Path, requested: list[str] | None) -> list[str]:
    if requested:
        return requested

    selected = []
    for model in MODEL_ORDER:
        if (evaluation_root / model / "paired_deltas.csv").exists():
            selected.append(model)
    return selected


def _variation_sort_key(variation_name: str) -> tuple[int, str, str]:
    if ":" in variation_name:
        category, value = variation_name.split(":", 1)
    else:
        category, value = variation_name, ""
    category = category.strip().lower()
    value = value.strip().lower()
    try:
        cat_idx = CATEGORY_ORDER.index(category)
    except ValueError:
        cat_idx = len(CATEGORY_ORDER)
    return cat_idx, category, value


def _format_variation_label(variation_name: str) -> str:
    if ":" in variation_name:
        category, value = variation_name.split(":", 1)
    else:
        category, value = variation_name, ""

    category = category.strip().lower()
    value = value.strip()

    cat_display = CATEGORY_DISPLAY.get(category, category.replace("_", " ").title())
    if not value:
        return cat_display
    return f"{cat_display}: {value}"


def load_delta_arrays(
    evaluation_root: Path,
    models: list[str],
) -> tuple[dict[str, list[str]], dict[str, dict[str, dict[str, np.ndarray]]]]:
    """Load raw deltas and return gender-specific variation lists.

    Returns:
      variations_by_gender: {gender: [variation names in sorted order]}
      deltas: model -> gender -> variation -> np.ndarray
    """

    bucket: dict[str, dict[str, dict[str, list[float]]]] = {
        model: {gender: defaultdict(list) for gender in GENDERS}
        for model in models
    }
    seen_by_gender: dict[str, set[str]] = {gender: set() for gender in GENDERS}

    for model in models:
        csv_path = evaluation_root / model / "paired_deltas.csv"
        if not csv_path.exists():
            continue

        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                gender = str(row.get("gender", "")).strip().lower()
                if gender not in GENDERS:
                    continue

                variation_name = str(row.get("variation_name", "")).strip()
                if not variation_name:
                    continue

                try:
                    delta = float(row.get("delta", "0"))
                except Exception:
                    continue

                bucket[model][gender][variation_name].append(delta)
                seen_by_gender[gender].add(variation_name)

    variations_by_gender: dict[str, list[str]] = {
        gender: sorted(seen_by_gender[gender], key=_variation_sort_key)
        for gender in GENDERS
    }

    deltas: dict[str, dict[str, dict[str, np.ndarray]]] = {
        model: {gender: {} for gender in GENDERS}
        for model in models
    }
    for model in models:
        for gender in GENDERS:
            for variation in variations_by_gender[gender]:
                deltas[model][gender][variation] = np.array(
                    bucket[model][gender].get(variation, []),
                    dtype=float,
                )

    return variations_by_gender, deltas


def _category_boundaries(variations: list[str]) -> list[int]:
    boundaries = []
    prev_category = None
    for idx, variation in enumerate(variations):
        category = variation.split(":", 1)[0].strip().lower()
        if prev_category is None:
            prev_category = category
            continue
        if category != prev_category:
            boundaries.append(idx)
            prev_category = category
    return boundaries


def _compute_y_limits(
    deltas: dict[str, dict[str, dict[str, np.ndarray]]],
    models: list[str],
    variations_by_gender: dict[str, list[str]],
) -> tuple[float, float]:
    vals = []
    for model in models:
        for gender in GENDERS:
            for variation in variations_by_gender[gender]:
                arr = deltas[model][gender][variation]
                if arr.size == 0:
                    continue
                vals.append(float(np.min(arr)))
                vals.append(float(np.max(arr)))

    if not vals:
        return -0.2, 0.2

    lo = min(vals)
    hi = max(vals)
    bound = max(abs(lo), abs(hi), 0.05) * 1.12
    return -bound, bound


def _draw_gender_panel(
    ax,
    gender: str,
    models: list[str],
    variations: list[str],
    deltas: dict[str, dict[str, dict[str, np.ndarray]]],
    y_min: float,
    y_max: float,
) -> None:
    n_var = len(variations)
    x = np.arange(n_var)
    x_min, x_max = -0.7, n_var - 0.3

    y_gradient = np.linspace(0.0, 1.0, 512).reshape(-1, 1)
    ax.imshow(
        y_gradient,
        extent=[x_min, x_max, y_min, y_max],
        cmap=SIGNED_Y_CMAP,
        aspect="auto",
        interpolation="bicubic",
        alpha=0.50,
        zorder=-4,
    )

    # Category separators
    for boundary in _category_boundaries(variations):
        ax.axvline(boundary - 0.5, color="#C6CBD1", linewidth=0.9, alpha=0.85, zorder=0)

    ax.axhline(0.0, color="#2D2D2D", linewidth=1.2, alpha=0.9, zorder=1)

    offsets = np.linspace(-0.32, 0.32, max(1, len(models)))
    box_width = min(0.17, 0.78 / max(len(models), 1))

    for idx, model in enumerate(models):
        color = MODEL_COLORS.get(model, "#3A3A3A")
        positions = x + offsets[idx]

        box_data = []
        box_pos = []
        for j, variation in enumerate(variations):
            arr = deltas[model][gender][variation]
            if arr.size == 0:
                continue
            box_data.append(arr)
            box_pos.append(positions[j])

        if not box_data:
            continue

        bp = ax.boxplot(
            box_data,
            positions=box_pos,
            widths=box_width,
            whis=(0, 100),
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
            zorder=3,
        )

        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_edgecolor(color)
            patch.set_alpha(0.32)
            patch.set_linewidth(1.0)

        for median in bp["medians"]:
            median.set_color(color)
            median.set_linewidth(2.0)

        for whisker in bp["whiskers"]:
            whisker.set_color(color)
            whisker.set_linewidth(1.15)

        for cap in bp["caps"]:
            cap.set_color(color)
            cap.set_linewidth(1.15)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, color="#D3D7DB", alpha=0.9)
    ax.grid(axis="x", visible=False)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#A1A6AB")
    ax.spines["bottom"].set_color("#A1A6AB")

    ax.set_ylabel("Delta shift", fontsize=11)
    ax.set_title("Female" if gender == "female" else "Male", fontsize=14, weight="semibold", pad=9)
    ax.tick_params(axis="y", labelsize=10)

    # Requested: variation names shown under female panel as well.
    ax.set_xticks(x)
    ax.set_xticklabels(
        [_format_variation_label(v) for v in variations],
        rotation=75,
        ha="right",
        fontsize=8.2,
    )
    ax.tick_params(axis="x", labelbottom=True)


def plot(
    models: list[str],
    variations_by_gender: dict[str, list[str]],
    deltas: dict[str, dict[str, dict[str, np.ndarray]]],
    output_path: Path,
) -> None:
    y_min, y_max = _compute_y_limits(deltas, models, variations_by_gender)

    # Taller figure as requested.
    female_n = len(variations_by_gender["female"])
    male_n = len(variations_by_gender["male"])
    fig_width = max(16.0, 0.44 * max(female_n, male_n) + 7.0)
    fig, axes = plt.subplots(2, 1, figsize=(fig_width, 13.5))

    _draw_gender_panel(
        ax=axes[0],
        gender="female",
        models=models,
        variations=variations_by_gender["female"],
        deltas=deltas,
        y_min=y_min,
        y_max=y_max,
    )

    _draw_gender_panel(
        ax=axes[1],
        gender="male",
        models=models,
        variations=variations_by_gender["male"],
        deltas=deltas,
        y_min=y_min,
        y_max=y_max,
    )

    fig.suptitle(
        "Variation-Level Delta Distributions by Model and Gender",
        fontsize=18,
        weight="semibold",
        y=0.975,
    )
    fig.text(
        0.5,
        0.944,
        "Boxplots show median, quartiles (Q1/Q3), and whiskers to lower/upper extremes (min/max).",
        ha="center",
        fontsize=11,
        color="#4E4E4E",
    )

    legend_handles = [
        Patch(
            facecolor=MODEL_COLORS.get(model, "#3A3A3A"),
            edgecolor=MODEL_COLORS.get(model, "#3A3A3A"),
            alpha=0.32,
            label=MODEL_DISPLAY.get(model, model),
        )
        for model in models
    ]
    legend_handles.append(
        Line2D([0], [0], color="#2D2D2D", linewidth=2.0, label="Median")
    )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.048),
        ncol=max(3, len(legend_handles)),
        frameon=False,
        fontsize=10.5,
        handlelength=2.0,
        columnspacing=1.1,
    )

    fig.text(
        0.5,
        0.013,
        "Signed background: red = negative-associated direction, green = positive-associated direction.",
        ha="center",
        fontsize=10,
        color="#666666",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0.02, 0.10, 0.995, 0.90])
    fig.savefig(output_path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def write_summary_csv(
    models: list[str],
    variations_by_gender: dict[str, list[str]],
    deltas: dict[str, dict[str, dict[str, np.ndarray]]],
    output_csv: Path,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model",
                "gender",
                "variation_name",
                "n",
                "mean",
                "median",
                "q1",
                "q3",
                "min",
                "max",
            ]
        )
        for model in models:
            for gender in GENDERS:
                for variation in variations_by_gender[gender]:
                    arr = deltas[model][gender][variation]
                    if arr.size == 0:
                        writer.writerow([model, gender, variation, 0, "", "", "", "", "", ""])
                        continue
                    writer.writerow(
                        [
                            model,
                            gender,
                            variation,
                            int(arr.size),
                            f"{np.mean(arr):.8f}",
                            f"{np.median(arr):.8f}",
                            f"{np.percentile(arr, 25):.8f}",
                            f"{np.percentile(arr, 75):.8f}",
                            f"{np.min(arr):.8f}",
                            f"{np.max(arr):.8f}",
                        ]
                    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create variation-level grouped boxplots by gender across models."
    )
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=Path("output/evaluation"),
        help="Root folder containing model evaluation outputs.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional explicit model folder list.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/evaluation/plots/main_model_variation_boxplot_gender_split.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("output/evaluation/plots/main_model_variation_boxplot_gender_split.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    models = _available_models(args.evaluation_root, args.models)
    if not models:
        raise RuntimeError("No model folders with paired_deltas.csv were found.")

    variations_by_gender, deltas = load_delta_arrays(args.evaluation_root, models)
    plot(models, variations_by_gender, deltas, args.output)
    write_summary_csv(models, variations_by_gender, deltas, args.summary_csv)

    print("Included models:")
    for model in models:
        print(f"  - {model}")
    print(f"Female variations plotted: {len(variations_by_gender['female'])}")
    print(f"Male variations plotted: {len(variations_by_gender['male'])}")
    print(f"Saved figure: {args.output}")
    print(f"Saved summary: {args.summary_csv}")


if __name__ == "__main__":
    main()
