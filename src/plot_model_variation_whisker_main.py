#!/usr/bin/env python3
"""Create a publication-grade model comparison chart with whiskers.

The chart compares mean delta shifts by variation category across models,
separately for female and male, in a single figure.

For each model x gender x category:
- point: mean of scenario-level category deltas
- whisker: standard deviation across scenarios (scenario-to-scenario variability)
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

# Colorblind-safe, publication-friendly palette (Okabe-Ito inspired).
MODEL_COLORS = {
    "gemma3": "#0072B2",
    "gemma4": "#E69F00",
    "llava_next": "#009E73",
    "pixtral": "#CC79A7",
    "qwen3": "#56B4E9",
}

VARIATION_CATEGORY_ORDER = [
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

CATEGORY_LABELS = {
    "skin_irregularities": "Skin irregularities",
    "hair_color": "Hair color",
    "hair_length": "Hair length",
    "hair_style": "Hair style",
    "facial_hair_male": "Facial hair (M)",
    "makeup_female": "Makeup (F)",
    "lip_makeup_female": "Lip makeup (F)",
    "tattoos": "Tattoos",
    "fashion_style": "Fashion style",
    "eyewear": "Eyewear",
    "piercings": "Piercings",
    "accessories": "Accessories",
}

GENDERS = ["female", "male"]

SIGNED_BG_CMAP = LinearSegmentedColormap.from_list(
    "signed_bg",
    ["#F1C9C9", "#FCFCFD", "#CDEBD3"],
)


def _model_dirs(evaluation_root: Path, models: list[str] | None) -> list[str]:
    if models:
        return models

    discovered = []
    for model in MODEL_ORDER:
        if (evaluation_root / model / "variation_impact_summary.csv").exists():
            discovered.append(model)
    return discovered


def _category_from_variation_name(variation_name: str) -> str:
    return variation_name.split(":", 1)[0].strip().lower()


def load_scenario_level_category_means(
    evaluation_root: Path,
    models: list[str],
) -> dict[str, dict[str, dict[str, list[float]]]]:
    """Return nested dict: model -> gender -> category -> [scenario mean deltas]."""

    # Temporary structure for weighted aggregation within each scenario.
    # key: (model, gender, category, scenario) -> weighted_sum, weight_sum
    weighted_sum: dict[tuple[str, str, str, int], float] = defaultdict(float)
    weight_sum: dict[tuple[str, str, str, int], float] = defaultdict(float)

    for model in models:
        csv_path = evaluation_root / model / "variation_impact_summary.csv"
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
                category = _category_from_variation_name(variation_name)
                if category not in VARIATION_CATEGORY_ORDER:
                    continue

                try:
                    scenario = int(row.get("scenario", "0"))
                    mean_delta = float(row.get("mean_delta", "0"))
                    n_pairs = float(row.get("n_pairs", "0"))
                except Exception:
                    continue

                if scenario <= 0 or n_pairs <= 0:
                    continue

                key = (model, gender, category, scenario)
                weighted_sum[key] += n_pairs * mean_delta
                weight_sum[key] += n_pairs

    # Convert scenario-level weighted means into model/gender/category lists.
    out: dict[str, dict[str, dict[str, list[float]]]] = {
        model: {gender: {cat: [] for cat in VARIATION_CATEGORY_ORDER} for gender in GENDERS}
        for model in models
    }

    for key, ws in weighted_sum.items():
        model, gender, category, _scenario = key
        w = weight_sum[key]
        if w <= 0:
            continue
        out[model][gender][category].append(ws / w)

    return out


def summarize_category_stats(
    scenario_means: dict[str, dict[str, dict[str, list[float]]]],
    models: list[str],
) -> dict[str, dict[str, dict[str, tuple[float, float, int]]]]:
    """Return model -> gender -> category -> (mean, std, n_scenarios)."""

    summary: dict[str, dict[str, dict[str, tuple[float, float, int]]]] = {
        model: {gender: {} for gender in GENDERS} for model in models
    }

    for model in models:
        for gender in GENDERS:
            for category in VARIATION_CATEGORY_ORDER:
                values = np.array(scenario_means[model][gender][category], dtype=float)
                if values.size == 0:
                    summary[model][gender][category] = (np.nan, np.nan, 0)
                    continue
                mean = float(np.mean(values))
                std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
                summary[model][gender][category] = (mean, std, int(values.size))

    return summary


def _x_limits(summary: dict[str, dict[str, dict[str, tuple[float, float, int]]]], models: list[str]) -> tuple[float, float]:
    vals = []
    for model in models:
        for gender in GENDERS:
            for category in VARIATION_CATEGORY_ORDER:
                mean, std, n = summary[model][gender][category]
                if n <= 0 or not np.isfinite(mean):
                    continue
                s = std if np.isfinite(std) else 0.0
                vals.extend([mean - s, mean + s])

    if not vals:
        return -0.1, 0.1

    lo = float(min(vals))
    hi = float(max(vals))
    bound = max(abs(lo), abs(hi), 0.02)
    bound *= 1.20
    return -bound, bound


def plot_main_chart(
    summary: dict[str, dict[str, dict[str, tuple[float, float, int]]]],
    models: list[str],
    output_path: Path,
) -> None:
    y = np.arange(len(VARIATION_CATEGORY_ORDER))

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(16, 9),
        sharey=True,
    )

    x_min, x_max = _x_limits(summary, models)
    y_bottom = -0.5
    y_top = len(VARIATION_CATEGORY_ORDER) - 0.5
    bg_gradient = np.linspace(0.0, 1.0, 512).reshape(1, -1)

    offsets = np.linspace(-0.24, 0.24, max(len(models), 1))

    for ax_idx, gender in enumerate(GENDERS):
        ax = axes[ax_idx]
        ax.set_facecolor("#FCFCFD")

        # Subtle signed background: negative (left) red, positive (right) green.
        ax.imshow(
            bg_gradient,
            extent=[x_min, x_max, y_bottom, y_top],
            cmap=SIGNED_BG_CMAP,
            aspect="auto",
            interpolation="bicubic",
            alpha=0.58,
            zorder=-3,
        )

        for row in range(len(VARIATION_CATEGORY_ORDER)):
            if row % 2 == 0:
                ax.axhspan(row - 0.5, row + 0.5, color="#F4F6F8", alpha=0.6, zorder=0)

        ax.axvline(0.0, color="#2F2F2F", linewidth=1.2, alpha=0.85, zorder=1)

        for idx, model in enumerate(models):
            color = MODEL_COLORS.get(model, "#333333")
            means = []
            stds = []
            valid = []
            for cat in VARIATION_CATEGORY_ORDER:
                mean, std, n = summary[model][gender][cat]
                means.append(mean)
                stds.append(std if np.isfinite(std) else 0.0)
                valid.append(n > 0 and np.isfinite(mean))

            means_arr = np.array(means, dtype=float)
            stds_arr = np.array(stds, dtype=float)
            valid_mask = np.array(valid, dtype=bool)

            y_pos = y + offsets[idx]
            if np.any(valid_mask):
                ax.errorbar(
                    means_arr[valid_mask],
                    y_pos[valid_mask],
                    xerr=stds_arr[valid_mask],
                    fmt="o",
                    ms=5.5,
                    mfc=color,
                    mec="white",
                    mew=0.7,
                    ecolor=color,
                    elinewidth=1.4,
                    capsize=2.5,
                    alpha=0.95,
                    zorder=4,
                )

        ax.set_xlim(x_min, x_max)
        ax.grid(axis="x", linestyle="--", linewidth=0.8, color="#D3D7DB", alpha=0.85)
        ax.grid(axis="y", visible=False)

        ax.set_title("Female" if gender == "female" else "Male", fontsize=14, pad=14, weight="semibold")

        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color("#A0A4A8")
        ax.spines["bottom"].set_color("#A0A4A8")

    axes[0].set_yticks(y)
    axes[0].set_yticklabels([CATEGORY_LABELS[c] for c in VARIATION_CATEGORY_ORDER], fontsize=11)
    axes[0].invert_yaxis()

    # Keep category labels visible on both panels for easier scanning.
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([CATEGORY_LABELS[c] for c in VARIATION_CATEGORY_ORDER], fontsize=11)
    axes[1].invert_yaxis()

    for ax in axes:
        ax.tick_params(axis="x", labelsize=10)

    fig.suptitle(
        "Model Comparison of Variation-Induced Delta Shifts Across Scenarios",
        fontsize=18,
        y=0.97,
        weight="semibold",
    )
    fig.text(
        0.5,
        0.935,
        "Points: mean scenario delta per category; whiskers: standard deviation across scenarios.",
        ha="center",
        fontsize=11,
        color="#4D4D4D",
    )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="-",
            color=MODEL_COLORS.get(model, "#333333"),
            markerfacecolor=MODEL_COLORS.get(model, "#333333"),
            markeredgecolor="white",
            markeredgewidth=0.7,
            linewidth=2,
            label=MODEL_DISPLAY.get(model, model),
        )
        for model in models
    ]

    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=max(1, len(models)),
        frameon=False,
        fontsize=11,
        handlelength=2.1,
        columnspacing=1.4,
    )

    fig.text(
        0.5,
        0.006,
        "Positive values indicate increased preference toward option A; negative values toward option B.",
        ha="center",
        fontsize=10,
        color="#616161",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0.03, 0.06, 0.99, 0.90])
    fig.savefig(output_path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def write_summary_csv(
    summary: dict[str, dict[str, dict[str, tuple[float, float, int]]]],
    models: list[str],
    output_csv: Path,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "gender", "category", "mean_delta", "std_across_scenarios", "n_scenarios"])
        for model in models:
            for gender in GENDERS:
                for category in VARIATION_CATEGORY_ORDER:
                    mean, std, n = summary[model][gender][category]
                    writer.writerow(
                        [
                            model,
                            gender,
                            category,
                            "" if not np.isfinite(mean) else f"{mean:.8f}",
                            "" if not np.isfinite(std) else f"{std:.8f}",
                            n,
                        ]
                    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot main model-variation whisker chart by gender.")
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=Path("output/evaluation"),
        help="Root folder containing model evaluation subfolders.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional model folder list (default: auto-detect known models).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/evaluation/plots/main_model_variation_whisker_gender.png"),
        help="Output figure path.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("output/evaluation/plots/main_model_variation_whisker_gender.csv"),
        help="Output summary CSV path.",
    )
    args = parser.parse_args()

    models = _model_dirs(args.evaluation_root, args.models)
    if not models:
        raise RuntimeError("No model folders with variation_impact_summary.csv were found.")

    scenario_means = load_scenario_level_category_means(args.evaluation_root, models)
    summary = summarize_category_stats(scenario_means, models)

    plot_main_chart(summary, models, args.output)
    write_summary_csv(summary, models, args.summary_csv)

    print("Included models:")
    for model in models:
        print(f"  - {model}")
    print(f"Saved figure: {args.output}")
    print(f"Saved summary: {args.summary_csv}")


if __name__ == "__main__":
    main()
