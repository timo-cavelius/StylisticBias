#!/usr/bin/env python3
"""Create a variation-level model comparison chart with asymmetric whiskers.

Figure design:
- Two stacked panels: Female (top), Male (bottom)
- X-axis: all individual variations (e.g., eyewear:Sunglasses)
- Dot: mean delta across all paired observations for model/gender/variation
- Asymmetric whiskers:
    lower = tendency toward negative deltas  = mean(max(-delta, 0))
    upper = tendency toward positive deltas  = mean(max(delta, 0))

This captures both direction preference and directional tendency magnitude.
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


MODEL_ORDER = ["gemma3", "gemma4", "llava_next", "pixtral", "qwen3"]
MODEL_DISPLAY = {
    "gemma3": "Gemma 3",
    "gemma4": "Gemma 4",
    "llava_next": "LLaVA 1.6",
    "pixtral": "Pixtral",
    "qwen3": "Qwen 3",
}

# Publication-friendly, colorblind-safe palette.
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
    return value


def load_delta_arrays(
    evaluation_root: Path,
    models: list[str],
) -> tuple[list[str], dict[str, dict[str, dict[str, np.ndarray]]]]:
    """Load raw delta arrays per model/gender/variation."""

    bucket: dict[str, dict[str, dict[str, list[float]]]] = {
        model: {gender: defaultdict(list) for gender in GENDERS}
        for model in models
    }
    all_variations: set[str] = set()

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
                all_variations.add(variation_name)

    variation_order = sorted(all_variations, key=_variation_sort_key)

    out: dict[str, dict[str, dict[str, np.ndarray]]] = {
        model: {gender: {} for gender in GENDERS}
        for model in models
    }

    for model in models:
        for gender in GENDERS:
            for variation in variation_order:
                values = np.array(bucket[model][gender].get(variation, []), dtype=float)
                out[model][gender][variation] = values

    return variation_order, out


def summarize(
    deltas: dict[str, dict[str, dict[str, np.ndarray]]],
    models: list[str],
    variations: list[str],
) -> dict[str, dict[str, dict[str, tuple[float, float, float, int]]]]:
    """Return (mean, neg_tendency, pos_tendency, n) per model/gender/variation."""

    summary: dict[str, dict[str, dict[str, tuple[float, float, float, int]]]] = {
        model: {gender: {} for gender in GENDERS}
        for model in models
    }

    for model in models:
        for gender in GENDERS:
            for variation in variations:
                vals = deltas[model][gender][variation]
                if vals.size == 0:
                    summary[model][gender][variation] = (np.nan, np.nan, np.nan, 0)
                    continue

                mean_delta = float(np.mean(vals))
                pos_tendency = float(np.mean(np.clip(vals, 0.0, None)))
                neg_tendency = float(np.mean(np.clip(-vals, 0.0, None)))
                summary[model][gender][variation] = (
                    mean_delta,
                    neg_tendency,
                    pos_tendency,
                    int(vals.size),
                )

    return summary


def _compute_y_limits(
    summary: dict[str, dict[str, dict[str, tuple[float, float, float, int]]]],
    models: list[str],
    variations: list[str],
) -> tuple[float, float]:
    vals = []
    for model in models:
        for gender in GENDERS:
            for variation in variations:
                mean, neg, pos, n = summary[model][gender][variation]
                if n <= 0 or not np.isfinite(mean):
                    continue
                low = mean - (neg if np.isfinite(neg) else 0.0)
                high = mean + (pos if np.isfinite(pos) else 0.0)
                vals.extend([low, high])

    if not vals:
        return -0.2, 0.2

    lo = min(vals)
    hi = max(vals)
    bound = max(abs(lo), abs(hi), 0.05) * 1.18
    return -bound, bound


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


def _category_spans(variations: list[str]) -> list[tuple[str, int, int]]:
    if not variations:
        return []

    spans: list[tuple[str, int, int]] = []
    start_idx = 0
    current_category = variations[0].split(":", 1)[0].strip().lower()

    for idx, variation in enumerate(variations[1:], start=1):
        category = variation.split(":", 1)[0].strip().lower()
        if category != current_category:
            spans.append((current_category, start_idx, idx - 1))
            current_category = category
            start_idx = idx

    spans.append((current_category, start_idx, len(variations) - 1))
    return spans


def plot(
    summary: dict[str, dict[str, dict[str, tuple[float, float, float, int]]]],
    models: list[str],
    variations: list[str],
    output_path: Path,
) -> None:
    n_var = len(variations)
    x = np.arange(n_var)

    fig_width = max(18.0, 0.46 * n_var + 8.5)
    fig, axes = plt.subplots(2, 1, figsize=(fig_width, 15.75), sharex=True)

    y_min, y_max = _compute_y_limits(summary, models, variations)
    x_min, x_max = -0.7, n_var - 0.3

    y_gradient = np.linspace(0.0, 1.0, 512).reshape(-1, 1)

    offsets = np.linspace(-0.28, 0.28, max(1, len(models)))

    for panel_idx, gender in enumerate(GENDERS):
        ax = axes[panel_idx]
        ax.set_facecolor("#FCFCFD")

        # Signed interpretation background on y-axis (negative red, positive green).
        ax.imshow(
            y_gradient,
            extent=[x_min, x_max, y_min, y_max],
            cmap=SIGNED_Y_CMAP,
            origin="lower",
            aspect="auto",
            interpolation="bicubic",
            alpha=0.50,
            zorder=-4,
        )

        # Category group boxes with light labels at top, to make grouping explicit.
        top_y = y_max - 0.06 * (y_max - y_min)
        for category, start_idx, end_idx in _category_spans(variations):
            left = start_idx - 0.5
            right = end_idx + 0.5
            ax.axvspan(
                left,
                right,
                facecolor="#EFF2F5",
                edgecolor="#C6CCD3",
                linewidth=0.9,
                alpha=0.25,
                zorder=-2,
            )
            ax.text(
                (left + right) / 2,
                top_y,
                CATEGORY_DISPLAY.get(category, category.replace("_", " ").title()),
                ha="center",
                va="top",
                fontsize=9.5,
                color="#9AA2AB",
                zorder=-1,
            )

        # Category separators.
        for boundary in _category_boundaries(variations):
            ax.axvline(boundary - 0.5, color="#C6CBD1", linewidth=0.9, alpha=0.85, zorder=0)

        ax.axhline(0.0, color="#2D2D2D", linewidth=1.2, alpha=0.9, zorder=1)

        for idx, model in enumerate(models):
            color = MODEL_COLORS.get(model, "#3A3A3A")
            means = []
            negs = []
            poss = []
            valid = []

            for variation in variations:
                mean, neg, pos, n = summary[model][gender][variation]
                means.append(mean)
                negs.append(neg if np.isfinite(neg) else 0.0)
                poss.append(pos if np.isfinite(pos) else 0.0)
                valid.append(n > 0 and np.isfinite(mean))

            means_arr = np.array(means, dtype=float)
            neg_arr = np.array(negs, dtype=float)
            pos_arr = np.array(poss, dtype=float)
            valid_mask = np.array(valid, dtype=bool)

            x_pos = x + offsets[idx]
            if np.any(valid_mask):
                ax.errorbar(
                    x_pos[valid_mask],
                    means_arr[valid_mask],
                    yerr=np.vstack([neg_arr[valid_mask], pos_arr[valid_mask]]),
                    fmt="o",
                    ms=4.6,
                    mfc=color,
                    mec="white",
                    mew=0.6,
                    ecolor=color,
                    elinewidth=1.15,
                    capsize=2.0,
                    alpha=0.95,
                    zorder=4,
                )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(axis="y", linestyle="--", linewidth=0.8, color="#D3D7DB", alpha=0.9)
        ax.grid(axis="x", visible=False)

        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color("#A1A6AB")
        ax.spines["bottom"].set_color("#A1A6AB")

        ax.set_ylabel("Delta shift", fontsize=11)
        ax.set_title("Female" if gender == "female" else "Male", fontsize=14, weight="semibold", pad=10)
        ax.tick_params(axis="y", labelsize=10)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        [_format_variation_label(v) for v in variations],
        rotation=75,
        ha="right",
        fontsize=8.6,
    )
    axes[1].set_xlabel("Variation", fontsize=11, labelpad=10)

    fig.suptitle(
        "Variation-Level Delta Shifts Across Models, Split by Gender",
        fontsize=18,
        weight="semibold",
        y=0.975,
    )
    fig.text(
        0.5,
        0.944,
        "Dot = mean delta. Lower whisker = tendency toward negative shifts. Upper whisker = tendency toward positive shifts.",
        ha="center",
        fontsize=11,
        color="#4E4E4E",
    )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="-",
            color=MODEL_COLORS.get(model, "#3A3A3A"),
            markerfacecolor=MODEL_COLORS.get(model, "#3A3A3A"),
            markeredgecolor="white",
            markeredgewidth=0.6,
            linewidth=1.8,
            label=MODEL_DISPLAY.get(model, model),
        )
        for model in models
    ]

    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.058),
        ncol=max(1, len(models)),
        frameon=False,
        fontsize=11,
        handlelength=2.0,
        columnspacing=1.4,
    )

    fig.text(
        0.5,
        0.014,
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
    summary: dict[str, dict[str, dict[str, tuple[float, float, float, int]]]],
    models: list[str],
    variations: list[str],
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
                "mean_delta",
                "neg_tendency_mean_clip",
                "pos_tendency_mean_clip",
                "n",
            ]
        )
        for model in models:
            for gender in GENDERS:
                for variation in variations:
                    mean, neg, pos, n = summary[model][gender][variation]
                    writer.writerow(
                        [
                            model,
                            gender,
                            variation,
                            "" if not np.isfinite(mean) else f"{mean:.8f}",
                            "" if not np.isfinite(neg) else f"{neg:.8f}",
                            "" if not np.isfinite(pos) else f"{pos:.8f}",
                            n,
                        ]
                    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot full variation-level model whisker chart by gender.")
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=Path("output/evaluation"),
        help="Root folder with per-model evaluation outputs.",
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
        default=Path("output/evaluation/plots/main_model_variation_whisker_full_variations.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("output/evaluation/plots/main_model_variation_whisker_full_variations.csv"),
        help="Output summary CSV path.",
    )
    args = parser.parse_args()

    models = _available_models(args.evaluation_root, args.models)
    if not models:
        raise RuntimeError("No model folders with paired_deltas.csv were found.")

    variations, deltas = load_delta_arrays(args.evaluation_root, models)
    summary = summarize(deltas, models, variations)

    plot(summary, models, variations, args.output)
    write_summary_csv(summary, models, variations, args.summary_csv)

    print("Included models:")
    for model in models:
        print(f"  - {model}")
    print(f"Variations plotted: {len(variations)}")
    print(f"Saved figure: {args.output}")
    print(f"Saved summary: {args.summary_csv}")


if __name__ == "__main__":
    main()
