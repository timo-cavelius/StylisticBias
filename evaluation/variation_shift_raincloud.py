#!/usr/bin/env python3
"""Create a publication-style raincloud/violin plot for variation shift distributions.

This script loads `paired_deltas.csv` files from model folders in `output/evaluation`,
groups variations into Intrinsic/Stylistic/Contextual categories, and visualizes the
distribution of shift magnitudes (|delta|).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


GROUP_TO_CATEGORIES = {
    "Intrinsic": {"skin_irregularities", "hair_color", "hair_length"},
    "Stylistic": {
        "hair_style",
        "facial_hair_male",
        "makeup_female",
        "lip_makeup_female",
        "tattoos",
        "fashion_style",
    },
    "Contextual": {"eyewear", "piercings", "accessories", "headwear"},
}

CATEGORY_TO_GROUP = {
    category: group
    for group, categories in GROUP_TO_CATEGORIES.items()
    for category in categories
}

GROUP_ORDER = ["Intrinsic", "Stylistic", "Contextual"]

# Colorblind-safe palette frequently used in papers (Okabe-Ito family).
GROUP_COLORS = {
    "Intrinsic": "#0072B2",  # blue
    "Stylistic": "#D55E00",  # vermillion
    "Contextual": "#009E73",  # bluish green
}


def _is_model_folder(path: Path) -> bool:
    if not path.is_dir():
        return False
    name = path.name
    return not name.startswith("model_comparison_")


def _find_model_dirs(evaluation_root: Path, requested_models: list[str] | None) -> list[Path]:
    if requested_models:
        return [evaluation_root / model for model in requested_models]

    candidates = [p for p in evaluation_root.iterdir() if _is_model_folder(p)]
    return sorted(candidates)


def _category_from_variation_name(variation_name: str) -> str:
    return variation_name.split(":", 1)[0].strip().lower()


def load_shift_data(evaluation_root: Path, models: list[str] | None) -> tuple[dict[str, list[float]], dict[str, int]]:
    grouped: dict[str, list[float]] = {group: [] for group in GROUP_ORDER}
    per_model_counts: dict[str, int] = {}

    for model_dir in _find_model_dirs(evaluation_root, models):
        csv_path = model_dir / "paired_deltas.csv"
        if not csv_path.exists():
            continue

        loaded_count = 0
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                variation_name = (row.get("variation_name") or "").strip()
                if not variation_name:
                    continue

                category = _category_from_variation_name(variation_name)
                group = CATEGORY_TO_GROUP.get(category)
                if not group:
                    continue

                try:
                    delta = float(row.get("delta", "0"))
                except Exception:
                    continue

                grouped[group].append(abs(delta))
                loaded_count += 1

        per_model_counts[model_dir.name] = loaded_count

    return grouped, per_model_counts


def _tail_summary(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"n": 0, "median": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0, "tail_ge_010": 0.0}

    return {
        "n": float(values.size),
        "median": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "tail_ge_010": float(np.mean(values >= 0.10)),
    }


def _write_summary_csv(output_csv: Path, grouped_values: dict[str, np.ndarray]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "n", "median", "p90", "p95", "max", "share_ge_0.10"])
        for group in GROUP_ORDER:
            summary = _tail_summary(grouped_values[group])
            writer.writerow(
                [
                    group,
                    int(summary["n"]),
                    f"{summary['median']:.6f}",
                    f"{summary['p90']:.6f}",
                    f"{summary['p95']:.6f}",
                    f"{summary['max']:.6f}",
                    f"{summary['tail_ge_010']:.6f}",
                ]
            )


def create_plot(grouped: dict[str, list[float]], output_png: Path) -> None:
    grouped_values = {group: np.array(grouped[group], dtype=float) for group in GROUP_ORDER}

    if all(values.size == 0 for values in grouped_values.values()):
        raise RuntimeError("No valid delta data found. Check evaluation files.")

    # Create a robust x-limit that still shows tails without being dominated by outliers.
    non_empty = [vals for vals in grouped_values.values() if vals.size > 0]
    upper = max(float(np.percentile(vals, 99.5)) for vals in non_empty)
    upper = max(upper, 0.20)
    x_limit = min(max(upper * 1.10, 0.22), 1.0)

    fig, ax = plt.subplots(figsize=(11.5, 7.4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FCFCFC")

    y_positions = np.arange(len(GROUP_ORDER), 0, -1)

    violin_data = [grouped_values[group] for group in GROUP_ORDER]
    parts = ax.violinplot(
        violin_data,
        positions=y_positions,
        vert=False,
        widths=0.78,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for idx, body in enumerate(parts["bodies"]):
        group = GROUP_ORDER[idx]
        color = GROUP_COLORS[group]
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.24)
        body.set_linewidth(1.2)

    # Box layer to give a clean central tendency summary.
    box = ax.boxplot(
        violin_data,
        positions=y_positions,
        vert=False,
        widths=0.20,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#2F2F2F", "linewidth": 1.8},
        whiskerprops={"color": "#595959", "linewidth": 1.1},
        capprops={"color": "#595959", "linewidth": 1.1},
    )

    for idx, patch in enumerate(box["boxes"]):
        group = GROUP_ORDER[idx]
        patch.set_facecolor(GROUP_COLORS[group])
        patch.set_edgecolor("#2F2F2F")
        patch.set_alpha(0.55)
        patch.set_linewidth(1.0)

    # Rain points: light jittered scatter for distribution texture.
    rng = np.random.default_rng(42)
    max_points_per_group = 2500

    for idx, group in enumerate(GROUP_ORDER):
        values = grouped_values[group]
        if values.size == 0:
            continue

        if values.size > max_points_per_group:
            sample_idx = rng.choice(values.size, size=max_points_per_group, replace=False)
            plot_values = values[sample_idx]
        else:
            plot_values = values

        y = y_positions[idx] - 0.24 - rng.uniform(0.02, 0.18, size=plot_values.size)
        ax.scatter(
            plot_values,
            y,
            s=10,
            color=GROUP_COLORS[group],
            alpha=0.20,
            linewidths=0,
            zorder=3,
        )

    # Axes and annotation styling.
    ax.set_yticks(y_positions)
    ax.set_yticklabels(GROUP_ORDER, fontsize=12)
    ax.set_xlim(0, x_limit)
    ax.set_xlabel("Absolute judgment shift |delta|", fontsize=12)
    ax.set_ylabel("Variation Group", fontsize=12)

    ax.grid(axis="x", linestyle="--", linewidth=0.8, color="#D6D6D6", alpha=0.9)
    ax.grid(axis="y", visible=False)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#8A8A8A")
    ax.spines["bottom"].set_color("#8A8A8A")

    summaries = {group: _tail_summary(grouped_values[group]) for group in GROUP_ORDER}
    for idx, group in enumerate(GROUP_ORDER):
        s = summaries[group]
        y = y_positions[idx] + 0.24
        note = (
            f"n={int(s['n']):,}  p95={s['p95']:.3f}  "
            f"max={s['max']:.3f}  >=0.10: {100*s['tail_ge_010']:.1f}%"
        )
        ax.text(
            x_limit * 0.56,
            y,
            note,
            fontsize=10,
            color="#3E3E3E",
            ha="left",
            va="center",
        )

    fig.suptitle(
        "Distribution of Variation-Induced Judgment Shifts",
        fontsize=14,
        weight="semibold",
        y=0.992,
    )
    fig.text(
        0.5,
        0.922,
        "Raincloud-style violin summary across all evaluated models; heavier tails imply more extreme shifts.",
        fontsize=10.5,
        color="#5A5A5A",
        ha="center",
    )

    interpretation_handles = [
        Line2D(
            [],
            [],
            color="#7A7A7A",
            linewidth=2.2,
            marker=r"$\sim$",
            markersize=13,
            markerfacecolor="none",
            label="distribution density",
        ),
        Patch(facecolor="#7A7A7A", edgecolor="#2F2F2F", alpha=0.55, label="median + interquartile range + whiskers"),
        Line2D([], [], marker="o", linestyle="None", markersize=6, markerfacecolor="#7A7A7A", markeredgewidth=0, alpha=0.6, label="individual observations"),
    ]
    fig.legend(
        handles=interpretation_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),
        fontsize=10,
        handlelength=1.8,
        columnspacing=1.6,
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0.02, 0.11, 0.98, 0.80])
    plt.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot raincloud/violin distributions for variation shift magnitudes.")
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=Path("output/evaluation"),
        help="Folder that contains per-model evaluation outputs.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional list of model folder names to include.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/evaluation/plots/variation_shift_raincloud_violin.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("output/evaluation/plots/variation_shift_tail_summary.csv"),
        help="Output CSV path for group tail summaries.",
    )
    args = parser.parse_args()

    grouped, counts = load_shift_data(args.evaluation_root, args.models)
    if not counts:
        raise RuntimeError("No model data could be loaded from paired_deltas.csv files.")

    grouped_values = {group: np.array(grouped[group], dtype=float) for group in GROUP_ORDER}
    _write_summary_csv(args.summary_csv, grouped_values)
    create_plot(grouped, args.output)

    print("Loaded rows by model:")
    for model_name in sorted(counts):
        print(f"  {model_name}: {counts[model_name]:,}")
    print(f"Saved plot: {args.output}")
    print(f"Saved summary: {args.summary_csv}")


if __name__ == "__main__":
    main()
