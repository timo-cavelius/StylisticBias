#!/usr/bin/env python3
"""Compare model evaluation outputs and generate paper-ready summaries.

This script scans all model folders under output/evaluation, reads the
precomputed evaluation artifacts (paired deltas, base-face bias summaries,
scenario-level stats), and writes a new comparison folder with:

1) CSV tables for global and scenario-level model comparison
2) A markdown + text overview suitable as a paper draft base
3) Cross-model visualization plots

Usage:
  python3 src/compare_model_evaluations.py
  python3 src/compare_model_evaluations.py --evaluation-root output/evaluation
  python3 src/compare_model_evaluations.py --models gemma3 llava_next
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
from scipy import stats as scipy_stats


REQUIRED_FILES = [
    "paired_delta_statistics.json",
    "variation_impact_summary.csv",
    "base_faces_significant_biases.json",
    "base_faces_category_scenario_summary.csv",
    "base_faces_counts.json",
    "paired_deltas.csv",
]

BASE_RADAR_CATEGORIES = ["age", "gender", "ethnicity", "body_type"]
VARIATION_RADAR_CATEGORIES = [
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

VARIATION_CATEGORY_GROUPS = {
    "Intrinsic": ["skin_irregularities", "hair_color", "hair_length"],
    "Stylistic": ["hair_style", "facial_hair_male", "makeup_female", "lip_makeup_female", "tattoos", "fashion_style"],
    "Contextual": ["eyewear", "piercings", "headwear"],
}

HEATMAP_CMAP = "coolwarm"


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _bh_correction(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values in input order."""
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [1.0] * n
    prev = 1.0
    for rank, (i, p) in enumerate(reversed(indexed), 1):
        adj = min(p * n / (n - rank + 1), prev)
        prev = adj
        adjusted[i] = adj
    return adjusted


def _safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value, default=0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _read_key_value_text(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        values[key.strip()] = raw.strip()
    return values


def _discover_model_dirs(evaluation_root: Path, selected_models: list[str] | None) -> list[Path]:
    selected_set = set(selected_models or [])
    model_dirs: list[Path] = []

    for entry in sorted(evaluation_root.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.startswith("model_comparison"):
            continue
        if selected_set and entry.name not in selected_set:
            continue

        has_required = all((entry / rel).exists() for rel in REQUIRED_FILES)
        if has_required:
            model_dirs.append(entry)

    return model_dirs


def _load_scenario_labels(csv_path: Path) -> dict[int, str]:
    labels: dict[int, str] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            scenario = _safe_int(row.get("scenario"), 0)
            label = (row.get("scenario_label") or "").strip()
            if scenario > 0 and label and scenario not in labels:
                labels[scenario] = label
    return labels


def _compute_paired_delta_distribution_metrics(csv_path: Path) -> dict[str, float]:
    deltas: list[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            deltas.append(_safe_float(row.get("delta"), 0.0))

    if not deltas:
        return {
            "n": 0,
            "zero_prop": 0.0,
            "abs_ge_0_25": 0.0,
            "abs_ge_0_5": 0.0,
            "abs_ge_0_75": 0.0,
        }

    n = len(deltas)
    abs_vals = [abs(d) for d in deltas]
    return {
        "n": n,
        "zero_prop": sum(1 for d in deltas if d == 0.0) / n,
        "abs_ge_0_25": sum(1 for d in abs_vals if d >= 0.25) / n,
        "abs_ge_0_5": sum(1 for d in abs_vals if d >= 0.5) / n,
        "abs_ge_0_75": sum(1 for d in abs_vals if d >= 0.75) / n,
    }


def _compute_base_polarization_metrics(csv_path: Path) -> tuple[float, float, dict[str, float]]:
    abs_distances: list[float] = []
    std_values: list[float] = []
    by_category_sum: dict[str, float] = defaultdict(float)
    by_category_n: dict[str, int] = defaultdict(int)

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            category = (row.get("category_type") or "unknown").strip()
            mean_a = _safe_float(row.get("mean_p_option_a"), 0.5)
            std_a = _safe_float(row.get("std_p_option_a"), 0.0)

            distance = abs(mean_a - 0.5)
            abs_distances.append(distance)
            std_values.append(std_a)

            by_category_sum[category] += distance
            by_category_n[category] += 1

    overall = float(np.mean(abs_distances)) if abs_distances else 0.0
    mean_std = float(np.mean(std_values)) if std_values else 0.0

    by_category: dict[str, float] = {}
    for category in sorted(by_category_sum):
        by_category[category] = by_category_sum[category] / max(by_category_n[category], 1)

    return overall, mean_std, by_category


def _compute_variation_category_strength(csv_path: Path) -> dict[str, float]:
    weighted_sum: dict[str, float] = defaultdict(float)
    weight_sum: dict[str, float] = defaultdict(float)

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            variation_name = (row.get("variation_name") or "").strip()
            if not variation_name:
                continue

            category = variation_name.split(":", 1)[0].strip()
            mean_delta = abs(_safe_float(row.get("mean_delta"), 0.0))
            n_pairs = _safe_float(row.get("n_pairs"), 0.0)
            if n_pairs <= 0:
                continue

            weighted_sum[category] += n_pairs * mean_delta
            weight_sum[category] += n_pairs

    strengths: dict[str, float] = {}
    for category in sorted(weight_sum):
        strengths[category] = weighted_sum[category] / max(weight_sum[category], 1e-12)
    return strengths


def _compute_gender_delta_metrics(csv_path: Path) -> tuple[dict[str, float], dict[str, float]]:
    weighted_sum: dict[str, float] = defaultdict(float)
    weighted_abs_sum: dict[str, float] = defaultdict(float)
    weight_sum: dict[str, float] = defaultdict(float)

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            gender = (row.get("gender") or "unknown").strip().lower()
            n_pairs = _safe_float(row.get("n_pairs"), 0.0)
            mean_delta = _safe_float(row.get("mean_delta"), 0.0)
            if n_pairs <= 0:
                continue

            weighted_sum[gender] += n_pairs * mean_delta
            weighted_abs_sum[gender] += n_pairs * abs(mean_delta)
            weight_sum[gender] += n_pairs

    weighted_mean: dict[str, float] = {}
    weighted_abs_mean: dict[str, float] = {}
    for gender in sorted(weight_sum):
        w = max(weight_sum[gender], 1e-12)
        weighted_mean[gender] = weighted_sum[gender] / w
        weighted_abs_mean[gender] = weighted_abs_sum[gender] / w

    return weighted_mean, weighted_abs_mean


def _load_base_face_category_variation_strength(csv_path: Path) -> dict[str, dict]:
    """Load category variation strength from base-face analysis.

    Returns a dict mapping category_type -> {
        'strength': float,
        'ci_lower': float | None,
        'ci_upper': float | None,
        'wilcoxon_p': float | None,
        'wilcoxon_p_bh': float | None,  # within-model BH correction
    }
    """
    result: dict[str, dict] = {}
    if not csv_path.exists():
        return result

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            category = (row.get("category_type") or "").strip()
            if not category:
                continue
            result[category] = {
                "strength": _safe_float(row.get("variation_strength"), math.nan),
                "ci_lower": _safe_float(row.get("ci_lower_95"), math.nan),
                "ci_upper": _safe_float(row.get("ci_upper_95"), math.nan),
                "wilcoxon_p": _safe_float(row.get("wilcoxon_p"), math.nan),
                "wilcoxon_p_bh": _safe_float(row.get("wilcoxon_p_bh"), math.nan),
            }

    return result


def _load_base_face_category_variation_strength_by_label(csv_path: Path) -> dict[tuple[str, str], float]:
    """Load label-level category variation strength from base-face analysis."""
    strengths: dict[tuple[str, str], float] = {}
    if not csv_path.exists():
        return strengths

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            category_type = (row.get("category_type") or "").strip()
            category_value = (row.get("category_value") or "").strip()
            strength = _safe_float(row.get("variation_strength"), 0.0)
            if category_type and category_value:
                strengths[(category_type, category_value)] = strength

    return strengths


def _format_radar_label(label: str) -> str:
    wrapped = {
        "lip_makeup_female": "lip makeup\nfemale",
        "skin_irregularities": "skin\nirregularities",
        "facial_hair_male": "facial hair\nmale",
        "makeup_female": "makeup\nfemale",
    }
    return wrapped.get(label, label.replace("_", " "))


def _draw_variation_group_arcs(ax, categories: list[str], radial_max: float):
    if not categories:
        return

    # Keep an alias so "headwear" can map to the existing "accessories" feature.
    index_by_category = {category: idx for idx, category in enumerate(categories)}
    if "accessories" in index_by_category and "headwear" not in index_by_category:
        index_by_category["headwear"] = index_by_category["accessories"]

    step = 2 * np.pi / len(categories)
    theta_centers = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)

    arc_r = radial_max * 1.08
    for group_name, group_categories in VARIATION_CATEGORY_GROUPS.items():
        indices = sorted({index_by_category[c] for c in group_categories if c in index_by_category})
        if not indices:
            continue

        start_idx = indices[0]
        end_idx = indices[-1]
        # Use a reduced span to keep visible gaps between group segments.
        theta_start = theta_centers[start_idx] - step * 0.28
        theta_end = theta_centers[end_idx] + step * 0.28
        arc_theta = np.linspace(theta_start, theta_end, 120)

        ax.plot(
            arc_theta,
            np.full_like(arc_theta, arc_r),
            color="#d9d9d9",
            linewidth=9,
            solid_capstyle="round",
            clip_on=False,
            zorder=1,
        )

        # Group names intentionally omitted to keep the outer ring clean.


def _plot_radar_comparison(
    models: list[dict],
    categories: list[str],
    metric_key: str,
    title: str,
    output_path: Path,
    value_label: str,
    normalize_axes: bool = True,
):
    if not models or not categories:
        return

    values_by_model = []
    for model in models:
        values_by_model.append([_safe_float(model.get(metric_key, {}).get(category), 0.0) for category in categories])

    arr = np.array(values_by_model, dtype=float)
    if normalize_axes:
        axis_max = np.nanmax(arr, axis=0) if arr.size else np.array([])
        axis_max = np.where(np.isfinite(axis_max) & (axis_max > 0), axis_max, 1.0)
        plot_values = arr / axis_max
        plot_values = np.clip(plot_values, 0.0, 1.0)
        radial_max = 1.0
        y_ticks = [0.25, 0.5, 0.75, 1.0]
        y_labels = ["0.25", "0.5", "0.75", "1.0"]
    else:
        plot_values = np.where(np.isfinite(arr), arr, 0.0)
        radial_max = float(np.nanmax(plot_values)) if plot_values.size else 1.0
        radial_max = max(radial_max, 1e-6)
        y_ticks = [radial_max * 0.25, radial_max * 0.5, radial_max * 0.75, radial_max]
        y_labels = [f"{tick:.3f}" for tick in y_ticks]

    theta = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
    theta = np.concatenate([theta, [theta[0]]])

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(models), 1)))
    is_raw_variation = metric_key == "variation_category_strength" and not normalize_axes
    model_display_names = {
        "gemma3": "Gemma 3",
        "gemma4": "Gemma 4",
        "llava_next": "LLaVA 1.6",
        "pixtral": "Pixtral",
        "qwen3": "Qwen 3",
    }

    for idx, model in enumerate(models):
        values = np.concatenate([plot_values[idx], [plot_values[idx][0]]]) if len(categories) else np.array([])
        ax.plot(
            theta,
            values,
            color=colors[idx],
            linewidth=2.2,
            label=model["name"],
            solid_joinstyle="miter",
            solid_capstyle="butt",
        )
        ax.fill(theta, values, color=colors[idx], alpha=0.08)

    if is_raw_variation:
        _draw_variation_group_arcs(ax, categories, radial_max)

    ax.set_xticks(theta[:-1])
    if is_raw_variation:
        ax.set_xticklabels([_format_radar_label(category) for category in categories], fontsize=13)
        ax.tick_params(axis="x", pad=40)
    else:
        ax.set_xticklabels([_format_radar_label(category) for category in categories], fontsize=9)
    ax.set_ylim(0, radial_max)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_title(title, pad=22)
    if is_raw_variation:
        legend_handles = []
        for idx, model in enumerate(models):
            color = colors[idx]
            display_name = model_display_names.get(model["name"], model["name"])
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    linewidth=8,
                    marker="o",
                    markersize=12,
                    markerfacecolor="white",
                    markeredgecolor=color,
                    markeredgewidth=2,
                    label=display_name,
                )
            )
        ax.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.30),
            ncol=max(len(models), 1),
            frameon=False,
            fontsize=11,
            handlelength=2.0,
            columnspacing=1.8,
        )
    else:
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    if is_raw_variation:
        fig.tight_layout(rect=[0.02, 0.09, 0.98, 0.95])
    else:
        fig.tight_layout(rect=[0, 0.04, 1, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _compute_bias_rate_metrics(
    significant_biases_path: Path,
    base_counts_path: Path,
    scenario_count: int,
) -> tuple[dict[str, int], dict[str, int], dict[str, float], dict[str, int]]:
    biases = _read_json(significant_biases_path)
    counts = _read_json(base_counts_path)

    by_category_sig: dict[str, int] = defaultdict(int)
    by_direction: dict[str, int] = defaultdict(int)

    for row in biases:
        category = str(row.get("category_type", "unknown"))
        by_category_sig[category] += 1

        direction = str(row.get("bias_direction", "unknown"))
        by_direction[direction] += 1

    denominators: dict[str, int] = {}
    rates: dict[str, float] = {}
    for category, value_counts in counts.items():
        if not isinstance(value_counts, dict):
            continue
        denominator = len(value_counts) * scenario_count
        denominators[category] = denominator
        rates[category] = by_category_sig.get(category, 0) / max(denominator, 1)

    return dict(by_category_sig), denominators, rates, dict(by_direction)


def _parse_model(model_dir: Path) -> dict:
    paired_stats = _read_json(model_dir / "paired_delta_statistics.json")
    per_scenario = paired_stats.get("per_scenario", {})
    overall = paired_stats.get("overall", {})

    scenario_count = len(per_scenario)
    scenario_metrics: dict[int, dict] = {}
    for key, value in per_scenario.items():
        sid = _safe_int(key, 0)
        if sid <= 0:
            continue
        scenario_metrics[sid] = {
            "mean_delta": _safe_float(value.get("mean_delta"), 0.0),
            "std_delta": _safe_float(value.get("std_delta"), 0.0),
            "cohens_d": _safe_float(value.get("cohens_d"), 0.0),
            "wilcoxon_p": _safe_float(
                (((value.get("tests") or {}).get("wilcoxon_signed_rank") or {}).get("p_value")),
                math.nan,
            ),
        }

    summary_txt = _read_key_value_text(model_dir / "summary.txt")
    base_summary_txt = _read_key_value_text(model_dir / "base_faces_summary.txt")

    dist_metrics = _compute_paired_delta_distribution_metrics(model_dir / "paired_deltas.csv")
    base_polarization, base_mean_std, polarization_by_category = _compute_base_polarization_metrics(
        model_dir / "base_faces_category_scenario_summary.csv"
    )
    weighted_mean_gender, weighted_abs_gender = _compute_gender_delta_metrics(
        model_dir / "variation_impact_summary.csv"
    )
    variation_category_strength = _compute_variation_category_strength(model_dir / "variation_impact_summary.csv")
    
    # Load base-face category variation strength (demographic attribute effects)
    base_face_category_variation = _load_base_face_category_variation_strength(
        model_dir / "category_variation_strength.csv"
    )
    base_face_category_variation_by_label = _load_base_face_category_variation_strength_by_label(
        model_dir / "category_variation_strength_by_label.csv"
    )

    bias_sig, bias_denoms, bias_rates, bias_direction = _compute_bias_rate_metrics(
        model_dir / "base_faces_significant_biases.json",
        model_dir / "base_faces_counts.json",
        scenario_count,
    )

    sig_scenarios = [
        sid
        for sid, m in scenario_metrics.items()
        if (not math.isnan(m["wilcoxon_p"])) and m["wilcoxon_p"] < 0.05
    ]
    nonsig_scenarios = [
        sid
        for sid, m in scenario_metrics.items()
        if (not math.isnan(m["wilcoxon_p"])) and m["wilcoxon_p"] >= 0.05
    ]

    sorted_by_mean = sorted(scenario_metrics.items(), key=lambda kv: kv[1]["mean_delta"])
    bottom3 = sorted_by_mean[:3]
    top3 = sorted_by_mean[-3:][::-1]

    scenario_labels = _load_scenario_labels(model_dir / "variation_impact_summary.csv")

    return {
        "name": model_dir.name,
        "path": model_dir,
        "scenario_labels": scenario_labels,
        "summary_txt": summary_txt,
        "base_summary_txt": base_summary_txt,
        "overall": {
            "n_pairs": _safe_int(overall.get("n_pairs"), 0),
            "mean_delta": _safe_float(overall.get("mean_delta"), 0.0),
            "std_delta": _safe_float(overall.get("std_delta"), 0.0),
            "cohens_d": _safe_float(overall.get("cohens_d"), 0.0),
            "wilcoxon_p": _safe_float(
                (((overall.get("tests") or {}).get("wilcoxon_signed_rank") or {}).get("p_value")),
                math.nan,
            ),
            "recommended_test": str((overall.get("tests") or {}).get("recommended_test") or ""),
        },
        "faces_processed": _safe_int(summary_txt.get("faces_processed"), 0),
        "faces_used": _safe_int(base_summary_txt.get("faces_used"), 0),
        "significant_bias_rows": _safe_int(base_summary_txt.get("significant_bias_rows"), 0),
        "scenario_metrics": scenario_metrics,
        "sign_counts": {
            "positive": sum(1 for m in scenario_metrics.values() if m["mean_delta"] > 0),
            "negative": sum(1 for m in scenario_metrics.values() if m["mean_delta"] < 0),
            "zero": sum(1 for m in scenario_metrics.values() if m["mean_delta"] == 0),
        },
        "significant_scenarios": sig_scenarios,
        "nonsignificant_scenarios": nonsig_scenarios,
        "top3_positive": top3,
        "top3_negative": bottom3,
        "distribution": dist_metrics,
        "base_polarization": base_polarization,
        "base_mean_std": base_mean_std,
        "polarization_by_category": polarization_by_category,
        "weighted_mean_delta_by_gender": weighted_mean_gender,
        "weighted_abs_delta_by_gender": weighted_abs_gender,
        "variation_category_strength": variation_category_strength,
        "base_face_category_variation": base_face_category_variation,
        "base_face_category_variation_by_label": base_face_category_variation_by_label,
        "bias_sig_by_category": bias_sig,
        "bias_denominators": bias_denoms,
        "bias_rates": bias_rates,
        "bias_direction": bias_direction,
    }


def _write_csv(path: Path, header: list[str], rows: list[list]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _plot_global_overview(models: list[dict], output_path: Path):
    names = [m["name"] for m in models]
    mean_delta = [m["overall"]["mean_delta"] for m in models]
    std_delta = [m["overall"]["std_delta"] for m in models]

    x = np.arange(len(names))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 5))
    bars = ax1.bar(x - width / 2, mean_delta, width=width, label="mean delta", color="#2d6a4f")
    ax1.axhline(0.0, color="black", linewidth=1)
    ax1.set_ylabel("mean delta")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15)

    ax2 = ax1.twinx()
    line = ax2.plot(x + width / 2, std_delta, marker="o", color="#bc4749", linewidth=2, label="std delta")
    ax2.set_ylabel("std delta")

    ax1.set_title("Overall Variation Impact by Model")
    for bar, val in zip(bars, mean_delta):
        ax1.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_tail_rates(models: list[dict], output_path: Path):
    names = [m["name"] for m in models]
    metrics = ["abs_ge_0_25", "abs_ge_0_5", "abs_ge_0_75"]
    labels = ["|delta| >= 0.25", "|delta| >= 0.5", "|delta| >= 0.75"]
    colors = ["#386641", "#6a994e", "#a7c957"]

    x = np.arange(len(names))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        vals = [m["distribution"][metric] for m in models]
        ax.bar(x + (i - 1) * width, vals, width=width, label=label, color=color)

    ax.set_title("Delta Tail Rates by Model")
    ax.set_ylabel("proportion of pairs")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.legend()
    ax.set_ylim(0, 1)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_scenario_heatmap(models: list[dict], scenario_ids: list[int], scenario_labels: dict[int, str], output_path: Path):
    matrix = []
    for model in models:
        row = [model["scenario_metrics"].get(sid, {}).get("mean_delta", np.nan) for sid in scenario_ids]
        matrix.append(row)

    arr = np.array(matrix, dtype=float)
    with np.errstate(invalid="ignore"):
        avg_row = np.nanmean(arr, axis=0, keepdims=True)
    arr_with_avg = np.vstack([arr, avg_row])

    fig, ax = plt.subplots(figsize=(14, 4 + 0.55 * len(models) + 0.55))
    vmax = np.nanmax(np.abs(arr_with_avg)) if np.isfinite(arr_with_avg).any() else 1.0
    vmax = max(vmax, 1e-6)
    im = ax.imshow(
        arr_with_avg,
        cmap=HEATMAP_CMAP,
        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
        aspect="auto",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("mean delta", fontsize=14)
    cbar.ax.tick_params(labelsize=13)

    ax.set_yticks(np.arange(len(models) + 1))
    ax.set_yticklabels([m["name"] for m in models] + ["avg model mean"], fontsize=15)
    ax.axhline(len(models) - 0.5, color="black", linewidth=1.0)

    short_x_labels = []
    for sid in scenario_ids:
        label = scenario_labels.get(sid, f"scenario {sid}")
        if "|" in label:
            left, right = [part.strip() for part in label.split("|", 1)]
            formatted_label = f"{left} / {right}"
        else:
            formatted_label = label.strip()
        short_x_labels.append(f"{sid}: {formatted_label}")
    ax.set_xticks(np.arange(len(scenario_ids)))
    ax.set_xticklabels(short_x_labels, rotation=70, ha="right", fontsize=13)

    ax.set_title("Scenario-Level Mean Delta Across Models", fontsize=20)
    ax.set_xlabel("scenario", fontsize=16, fontweight="bold")
    ax.set_ylabel("model", fontsize=16, fontweight="bold")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_scenario_variation_strength_heatmap(
    models: list[dict],
    scenario_ids: list[int],
    scenario_labels: dict[int, str],
    output_path: Path,
):
    matrix = []
    for model in models:
        row = [model["scenario_metrics"].get(sid, {}).get("std_delta", np.nan) for sid in scenario_ids]
        matrix.append(row)

    arr = np.array(matrix, dtype=float)
    with np.errstate(invalid="ignore"):
        avg_row = np.nanmean(arr, axis=0, keepdims=True)
    arr_with_avg = np.vstack([arr, avg_row])

    fig, ax = plt.subplots(figsize=(14, 4 + 0.55 * len(models) + 0.55))
    vmax = np.nanmax(arr_with_avg) if np.isfinite(arr_with_avg).any() else 1.0
    vmax = max(vmax, 1e-6)
    im = ax.imshow(
        arr_with_avg,
        cmap="Reds",
        vmin=0.0,
        vmax=vmax,
        aspect="auto",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("scenario variation strength (std delta)", fontsize=14)
    cbar.ax.tick_params(labelsize=13)

    ax.set_yticks(np.arange(len(models) + 1))
    ax.set_yticklabels([m["name"] for m in models] + ["avg model variation"], fontsize=15)
    ax.axhline(len(models) - 0.5, color="black", linewidth=1.0)

    short_x_labels = []
    for sid in scenario_ids:
        label = scenario_labels.get(sid, f"scenario {sid}")
        if "|" in label:
            left, right = [part.strip() for part in label.split("|", 1)]
            formatted_label = f"{left} / {right}"
        else:
            formatted_label = label.strip()
        short_x_labels.append(f"{sid}: {formatted_label}")
    ax.set_xticks(np.arange(len(scenario_ids)))
    ax.set_xticklabels(short_x_labels, rotation=70, ha="right", fontsize=13)

    ax.set_title("Scenario-Level Variation Strength Across Models", fontsize=20)
    ax.set_xlabel("scenario", fontsize=16, fontweight="bold")
    ax.set_ylabel("model", fontsize=16, fontweight="bold")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_category_bias_rates(models: list[dict], output_path: Path):
    categories = sorted({cat for m in models for cat in m["bias_rates"].keys()})
    names = [m["name"] for m in models]
    x = np.arange(len(categories))

    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model in enumerate(models):
        vals = [model["bias_rates"].get(cat, 0.0) for cat in categories]
        offset = (i - (len(models) - 1) / 2) * width
        ax.bar(x + offset, vals, width=width, label=model["name"])

    ax.set_title("Significant Bias Rate by Category Type")
    ax.set_ylabel("rate (significant rows / possible rows)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=15)
    ax.set_ylim(0, 1)
    ax.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_gender_abs_delta(models: list[dict], output_path: Path):
    genders = sorted({g for m in models for g in m["weighted_abs_delta_by_gender"].keys()})
    x = np.arange(len(genders))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, model in enumerate(models):
        vals = [model["weighted_abs_delta_by_gender"].get(g, 0.0) for g in genders]
        offset = (i - (len(models) - 1) / 2) * width
        ax.bar(x + offset, vals, width=width, label=model["name"])

    ax.set_title("Weighted Mean |Delta| by Gender")
    ax.set_ylabel("weighted mean |delta|")
    ax.set_xticks(x)
    ax.set_xticklabels(genders)
    ax.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_base_face_category_variation(models: list[dict], output_path: Path):
    """Plot how demographic categories affect model judgments across models."""
    categories = sorted({cat for m in models for cat in m["base_face_category_variation"].keys()})
    if not categories or not models:
        return

    category_display = {
        "age": "Age",
        "body_type": "Body type",
        "ethnicity": "Ethnicity",
        "gender": "Gender",
    }
    
    x = np.arange(len(categories))
    width = 0.8 / max(len(models), 1)
    group_width = width * len(models)

    fig, ax = plt.subplots(figsize=(10, 5))
    model_colors = {
        "gemma3": "#2A82B8",
        "gemma4": "#D9A33A",
        "llava_next": "#3A9F84",
        "pixtral": "#C589AE",
        "qwen3": "#70B7DE",
        "internvl": "#C6783A",
    }

    avg_vals = []
    for cat in categories:
        vals_for_cat = [
            m["base_face_category_variation"][cat]["strength"]
            for m in models
            if cat in m["base_face_category_variation"]
            and not math.isnan(m["base_face_category_variation"][cat]["strength"])
        ]
        avg_vals.append(float(np.mean(vals_for_cat)) if vals_for_cat else 0.0)

    ax.bar(
        x,
        avg_vals,
        width=group_width,
        color="#D7D7D7",
        edgecolor="#C4C4C4",
        linewidth=0.6,
        alpha=0.55,
        zorder=0,
        label="Category average",
    )

    for i, model in enumerate(models):
        vals = [
            model["base_face_category_variation"].get(cat, {}).get("strength", 0.0) or 0.0
            for cat in categories
        ]
        offset = (i - (len(models) - 1) / 2) * width
        ax.bar(
            x + offset,
            vals,
            width=width,
            label=model["name"],
            color=model_colors.get(model["name"], "#4E79A7"),
            edgecolor="#8C8C8C",
            linewidth=0.5,
            alpha=0.90,
            zorder=2,
        )

    ax.set_title("Base-Face Category Variation Strength\n(How much judgments differ across demographic groups)")
    ax.set_ylabel("variation strength (std of means across groups)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([category_display.get(cat, cat.replace("_", " ").title()) for cat in categories])
    ax.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _build_pairwise_comparison(models: list[dict], scenario_ids: list[int]) -> list[dict]:
    pairwise: list[dict] = []
    for m1, m2 in itertools.combinations(models, 2):
        sign_flips = 0
        common = 0
        scenario_diff_sum = 0.0
        scenario_abs_diff_sum = 0.0

        for sid in scenario_ids:
            d1 = m1["scenario_metrics"].get(sid, {}).get("mean_delta")
            d2 = m2["scenario_metrics"].get(sid, {}).get("mean_delta")
            if d1 is None or d2 is None:
                continue
            common += 1
            scenario_diff_sum += d2 - d1
            scenario_abs_diff_sum += abs(d2 - d1)
            if (d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0):
                sign_flips += 1

        row = {
            "model_a": m1["name"],
            "model_b": m2["name"],
            "mean_delta_diff_b_minus_a": m2["overall"]["mean_delta"] - m1["overall"]["mean_delta"],
            "std_delta_ratio_a_div_b": m1["overall"]["std_delta"] / max(m2["overall"]["std_delta"], 1e-12),
            "scenario_sign_flips": sign_flips,
            "scenario_overlap": common,
            "mean_scenario_diff_b_minus_a": scenario_diff_sum / max(common, 1),
            "mean_abs_scenario_diff": scenario_abs_diff_sum / max(common, 1),
        }
        pairwise.append(row)
    return pairwise


_DEMOGRAPHIC_CATEGORY_VALUES: dict[str, list[str]] = {
    "age": ["young adult", "middle-aged adult", "elderly"],
    "gender": ["male", "female"],
    "ethnicity": ["Asian", "African", "European", "Middle Eastern", "Latino"],
    "body_type": ["normal", "obese", "thin"],
}
_BINARY_CATEGORIES = {"gender"}


def _load_official_scenario_ids(config_path: Path) -> set[int]:
    """Return the set of 1-based scenario IDs from judgement_scenarios.json."""
    data = _read_json(config_path)
    if not isinstance(data, list) or not data:
        return set()
    return set(range(1, len(data) + 1))


def _compute_demographic_sensitivity(model_dirs: list[Path]) -> list[dict]:
    """For each model × demographic attribute, compute the percentage of scenarios
    where face-level predictions differ significantly across demographic groups.

    Only the official 25 scenarios (from config/judgement_scenarios.json) are used,
    even for models that have additional scenarios.

    Test choice:
      - Binary attribute (gender): Mann-Whitney U test (two-sided)
      - Multi-level attribute (age 3, body_type 3, ethnicity 5): Kruskal-Wallis test

    BH correction is applied within each (model, category) pair across scenarios
    to control the false discovery rate.

    Returns a list of dicts, one per (model, category), with keys:
      model, category_type, test, n_sig_bh, n_total, pct_sig_bh,
      and per-scenario detail list.
    """
    official_scenarios = _load_official_scenario_ids(
        Path("config/judgement_scenarios.json")
    )

    results: list[dict] = []

    for model_dir in model_dirs:
        prob_path = model_dir / "base_faces_probability_scores.json"
        if not prob_path.exists():
            continue
        rows = _read_json(prob_path)
        if not isinstance(rows, list):
            continue

        # group: category_type -> scenario -> category_value -> [p_option_a]
        grouped: dict[str, dict[int, dict[str, list[float]]]] = {
            cat: defaultdict(lambda: defaultdict(list))
            for cat in _DEMOGRAPHIC_CATEGORY_VALUES
        }
        for row in rows:
            scen = row.get("scenario")
            p_a = row.get("p_option_a")
            if scen is None or p_a is None:
                continue
            scen_id = int(scen)
            if official_scenarios and scen_id not in official_scenarios:
                continue
            for cat, allowed in _DEMOGRAPHIC_CATEGORY_VALUES.items():
                val = row.get(cat)
                if val in allowed:
                    grouped[cat][scen_id][val].append(float(p_a))

        for cat, allowed_vals in _DEMOGRAPHIC_CATEGORY_VALUES.items():
            is_binary = cat in _BINARY_CATEGORIES

            # Compute raw p-value and effect size per scenario
            # Effect sizes: eta-squared (η²) for Kruskal-Wallis, rank-biserial r for Mann-Whitney U
            scenario_raw: list[tuple[int, float, float]] = []  # (scen_id, raw_p, effect_size)
            for scen_id, val_map in sorted(grouped[cat].items()):
                groups = [
                    val_map[v] for v in allowed_vals
                    if v in val_map and len(val_map[v]) >= 3
                ]
                if len(groups) < 2:
                    continue
                # If all observations are identical, KW tie correction → NaN.
                # Treat as p=1.0, effect_size=0.0 (no differential effect).
                all_vals = [v for g in groups for v in g]
                if len(set(all_vals)) == 1:
                    scenario_raw.append((scen_id, 1.0, 0.0))
                    continue
                try:
                    if is_binary:
                        stat, p = scipy_stats.mannwhitneyu(
                            groups[0], groups[1], alternative="two-sided"
                        )
                        n1, n2 = len(groups[0]), len(groups[1])
                        # rank-biserial correlation: r = 1 - 2U/(n1*n2)
                        effect = abs(1.0 - (2.0 * stat) / (n1 * n2)) if n1 * n2 > 0 else 0.0
                    else:
                        stat, p = scipy_stats.kruskal(*groups)
                        n_total_obs = sum(len(g) for g in groups)
                        k = len(groups)
                        # eta-squared: η² = (H - k + 1) / (n - k)
                        denom = n_total_obs - k
                        effect = (stat - k + 1) / denom if denom > 0 else 0.0
                        effect = max(0.0, float(effect))  # clamp to [0, 1]
                    p = float(p) if math.isfinite(p) else 1.0
                    scenario_raw.append((scen_id, p, float(effect)))
                except Exception:
                    pass

            if not scenario_raw:
                continue

            # BH correction within this (model, category) pair
            raw_ps = [p for _, p, _ in scenario_raw]
            corrected = _bh_correction(raw_ps)

            per_scenario = []
            for (scen_id, raw_p, eff), bh_p in zip(scenario_raw, corrected):
                per_scenario.append({
                    "scenario": scen_id,
                    "p_raw": raw_p,
                    "p_bh": bh_p,
                    "effect_size": eff,
                    "significant": bh_p < 0.05,
                    "sig_001": bh_p < 0.001,
                })

            n_total = len(per_scenario)
            n_sig = sum(1 for s in per_scenario if s["significant"])
            n_sig_001 = sum(1 for s in per_scenario if s["sig_001"])
            effect_sizes = [s["effect_size"] for s in per_scenario]
            bh_ps = [s["p_bh"] for s in per_scenario]

            results.append({
                "model": model_dir.name,
                "category_type": cat,
                "test": "mannwhitneyu" if is_binary else "kruskal",
                "effect_metric": "rank_biserial_r" if is_binary else "eta_squared",
                "n_sig_bh": n_sig,
                "n_sig_001": n_sig_001,
                "n_total": n_total,
                "pct_sig_bh": n_sig / n_total if n_total else math.nan,
                "pct_sig_001": n_sig_001 / n_total if n_total else math.nan,
                "median_effect_size": float(np.median(effect_sizes)),
                "mean_effect_size": float(np.mean(effect_sizes)),
                "median_p_bh": float(np.median(bh_ps)),
                "per_scenario": per_scenario,
            })

    return results


def _write_demographic_sensitivity(sensitivity: list[dict], output_dir: Path) -> None:
    """Write demographic sensitivity results to CSV files in output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    categories = ["age", "body_type", "ethnicity", "gender"]
    model_order = list(dict.fromkeys(r["model"] for r in sensitivity))

    # --- Summary table: one row per model ---
    # Columns per category: pct_sig_bh, pct_sig_001, n_sig_bh, n_sig_001,
    #   n_total, median_effect_size, median_p_bh, effect_metric
    col_groups = [
        ("pct_sig_bh",        lambda e: f"{e['pct_sig_bh'] * 100:.1f}"),
        ("pct_sig_001",       lambda e: f"{e['pct_sig_001'] * 100:.1f}"),
        ("n_sig_bh",          lambda e: e["n_sig_bh"]),
        ("n_sig_001",         lambda e: e["n_sig_001"]),
        ("n_total",           lambda e: e["n_total"]),
        ("median_effect_size", lambda e: f"{e['median_effect_size']:.4f}"),
        ("median_p_bh",       lambda e: f"{e['median_p_bh']:.4e}"),
        ("effect_metric",     lambda e: e["effect_metric"]),
    ]
    summary_header = ["model"] + [
        f"{cat}_{col}" for col, _ in col_groups for cat in categories
    ]
    idx: dict[tuple[str, str], dict] = {
        (r["model"], r["category_type"]): r for r in sensitivity
    }
    summary_rows = []
    for model in model_order:
        row: list = [model]
        for col, extractor in col_groups:
            for cat in categories:
                entry = idx.get((model, cat))
                row.append(extractor(entry) if entry else "")
        summary_rows.append(row)

    _write_csv(
        output_dir / "demographic_sensitivity_summary.csv",
        summary_header,
        summary_rows,
    )

    # --- Per-scenario detail: one row per (model, category, scenario) ---
    detail_header = [
        "model", "category_type", "test", "effect_metric", "scenario",
        "p_raw", "p_bh", "significant", "sig_001", "effect_size",
    ]
    detail_rows = []
    for r in sorted(sensitivity, key=lambda x: (x["model"], x["category_type"])):
        for s in r["per_scenario"]:
            detail_rows.append([
                r["model"],
                r["category_type"],
                r["test"],
                r["effect_metric"],
                s["scenario"],
                s["p_raw"],
                s["p_bh"],
                s["significant"],
                s["sig_001"],
                s["effect_size"],
            ])
    _write_csv(
        output_dir / "demographic_sensitivity_per_scenario.csv",
        detail_header,
        detail_rows,
    )


def _write_outputs(models: list[dict], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_ids = sorted({sid for m in models for sid in m["scenario_metrics"].keys()})
    scenario_labels: dict[int, str] = {}
    for m in models:
        for sid, label in m["scenario_labels"].items():
            scenario_labels.setdefault(sid, label)

    overview_rows = []
    for m in models:
        overview_rows.append(
            [
                m["name"],
                m["faces_processed"],
                m["faces_used"],
                m["overall"]["n_pairs"],
                m["overall"]["mean_delta"],
                m["overall"]["std_delta"],
                m["overall"]["cohens_d"],
                m["overall"]["wilcoxon_p"],
                m["distribution"]["zero_prop"],
                m["distribution"]["abs_ge_0_25"],
                m["distribution"]["abs_ge_0_5"],
                m["distribution"]["abs_ge_0_75"],
                len(m["significant_scenarios"]),
                len(m["nonsignificant_scenarios"]),
                m["base_polarization"],
                m["base_mean_std"],
                m["significant_bias_rows"],
            ]
        )

    _write_csv(
        output_dir / "model_overview.csv",
        [
            "model",
            "faces_processed",
            "faces_used",
            "pairs_total",
            "mean_delta",
            "std_delta",
            "cohens_d",
            "wilcoxon_p",
            "zero_delta_proportion",
            "tail_abs_delta_ge_0_25",
            "tail_abs_delta_ge_0_5",
            "tail_abs_delta_ge_0_75",
            "scenario_count_significant_wilcoxon",
            "scenario_count_nonsignificant_wilcoxon",
            "base_polarization_abs_dist_to_0_5",
            "base_mean_std_p_option_a",
            "significant_bias_rows",
        ],
        overview_rows,
    )

    scenario_rows = []
    for sid in scenario_ids:
        row = [sid, scenario_labels.get(sid, "")]
        for m in models:
            metric = m["scenario_metrics"].get(sid, {})
            row.extend(
                [
                    metric.get("mean_delta", ""),
                    metric.get("cohens_d", ""),
                    metric.get("wilcoxon_p", ""),
                ]
            )
        scenario_rows.append(row)

    scenario_header = ["scenario", "scenario_label"]
    for m in models:
        scenario_header.extend(
            [
                f"{m['name']}_mean_delta",
                f"{m['name']}_cohens_d",
                f"{m['name']}_wilcoxon_p",
            ]
        )

    _write_csv(output_dir / "scenario_comparison.csv", scenario_header, scenario_rows)

    scenario_variation_rows = []
    for sid in scenario_ids:
        row_vals = []
        for m in models:
            metric = m["scenario_metrics"].get(sid, {})
            row_vals.append(metric.get("std_delta", ""))

        numeric_vals = [float(v) for v in row_vals if isinstance(v, (int, float))]
        avg_val = float(np.mean(numeric_vals)) if numeric_vals else ""
        scenario_variation_rows.append([sid, scenario_labels.get(sid, ""), *row_vals, avg_val])

    scenario_variation_header = ["scenario", "scenario_label"]
    for m in models:
        scenario_variation_header.append(f"{m['name']}_variation_strength")
    scenario_variation_header.append("avg_model_variation_strength")

    _write_csv(
        output_dir / "scenario_variation_strength_comparison.csv",
        scenario_variation_header,
        scenario_variation_rows,
    )

    bias_rate_rows = []
    for m in models:
        all_categories = sorted(set(m["bias_denominators"].keys()) | set(m["bias_sig_by_category"].keys()))
        for cat in all_categories:
            bias_rate_rows.append(
                [
                    m["name"],
                    cat,
                    m["bias_sig_by_category"].get(cat, 0),
                    m["bias_denominators"].get(cat, 0),
                    m["bias_rates"].get(cat, 0.0),
                ]
            )
    _write_csv(
        output_dir / "category_bias_rates.csv",
        ["model", "category_type", "significant_count", "denominator", "significant_rate"],
        bias_rate_rows,
    )

    # Write base-face category variation strength comparison
    base_face_categories = sorted({cat for m in models for cat in m["base_face_category_variation"].keys()})
    if base_face_categories:
        # Apply cross-model BH correction across all (model × category) raw p-values
        cross_model_keys: list[tuple[str, str]] = []  # (model_name, category)
        cross_model_raw_p: list[float] = []
        for m in models:
            for cat in base_face_categories:
                p = (m["base_face_category_variation"].get(cat) or {}).get("wilcoxon_p", math.nan)
                if not math.isnan(p):
                    cross_model_keys.append((m["name"], cat))
                    cross_model_raw_p.append(p)

        cross_model_bh: dict[tuple[str, str], float] = {}
        if cross_model_raw_p:
            corrected = _bh_correction(cross_model_raw_p)
            for key, p_adj in zip(cross_model_keys, corrected):
                cross_model_bh[key] = p_adj

        # Build long-format CSV: one row per (category, model)
        base_face_var_rows = []
        for cat in base_face_categories:
            for m in models:
                metrics = m["base_face_category_variation"].get(cat) or {}
                p_bh_cross = cross_model_bh.get((m["name"], cat), "")
                base_face_var_rows.append([
                    cat,
                    m["name"],
                    metrics.get("strength", ""),
                    metrics.get("ci_lower", ""),
                    metrics.get("ci_upper", ""),
                    metrics.get("wilcoxon_p", ""),
                    metrics.get("wilcoxon_p_bh", ""),
                    p_bh_cross if p_bh_cross != "" else "",
                ])

        _write_csv(
            output_dir / "base_face_category_variation_strength.csv",
            ["category_type", "model", "variation_strength",
             "ci_lower_95", "ci_upper_95", "wilcoxon_p",
             "wilcoxon_p_bh_within_model", "wilcoxon_p_bh_cross_model"],
            base_face_var_rows,
        )

    # Write base-face category variation strength comparison by label
    base_face_label_keys = sorted(
        {key for m in models for key in m["base_face_category_variation_by_label"].keys()}
    )
    if base_face_label_keys:
        base_face_label_rows = []
        for category_type, category_value in base_face_label_keys:
            row = [category_type, category_value]
            for m in models:
                row.append(m["base_face_category_variation_by_label"].get((category_type, category_value), ""))
            base_face_label_rows.append(row)

        base_face_label_header = ["category_type", "category_value"] + [m["name"] for m in models]
        _write_csv(
            output_dir / "base_face_category_variation_strength_by_label.csv",
            base_face_label_header,
            base_face_label_rows,
        )

    # Write variation category strength comparison (raw values used in radar plot)
    variation_strength_categories = sorted({cat for m in models for cat in m["variation_category_strength"].keys()})
    if variation_strength_categories:
        variation_strength_rows = []
        for cat in variation_strength_categories:
            row = [cat]
            for m in models:
                row.append(m["variation_category_strength"].get(cat, ""))
            variation_strength_rows.append(row)

        variation_strength_header = ["variation_category"] + [m["name"] for m in models]
        _write_csv(
            output_dir / "variation_category_strength.csv",
            variation_strength_header,
            variation_strength_rows,
        )

    pairwise_rows_dict = _build_pairwise_comparison(models, scenario_ids)
    pairwise_rows = [
        [
            row["model_a"],
            row["model_b"],
            row["mean_delta_diff_b_minus_a"],
            row["std_delta_ratio_a_div_b"],
            row["scenario_sign_flips"],
            row["scenario_overlap"],
            row["mean_scenario_diff_b_minus_a"],
            row["mean_abs_scenario_diff"],
        ]
        for row in pairwise_rows_dict
    ]
    _write_csv(
        output_dir / "pairwise_model_comparison.csv",
        [
            "model_a",
            "model_b",
            "mean_delta_diff_b_minus_a",
            "std_delta_ratio_a_div_b",
            "scenario_sign_flips",
            "scenario_overlap",
            "mean_scenario_diff_b_minus_a",
            "mean_abs_scenario_diff",
        ],
        pairwise_rows,
    )

    plots_dir = output_dir / "plots"
    _plot_global_overview(models, plots_dir / "global_mean_std_comparison.png")
    _plot_tail_rates(models, plots_dir / "delta_tail_rates.png")
    _plot_scenario_heatmap(models, scenario_ids, scenario_labels, plots_dir / "scenario_mean_delta_heatmap.png")
    _plot_scenario_variation_strength_heatmap(
        models,
        scenario_ids,
        scenario_labels,
        plots_dir / "scenario_variation_strength_heatmap.png",
    )
    _plot_category_bias_rates(models, plots_dir / "category_bias_rate_comparison.png")
    _plot_gender_abs_delta(models, plots_dir / "gender_weighted_abs_delta.png")
    _plot_base_face_category_variation(models, plots_dir / "base_face_category_variation_strength.png")

    base_radar_categories = [category for category in BASE_RADAR_CATEGORIES if any(category in m["polarization_by_category"] for m in models)]
    variation_radar_categories = [category for category in VARIATION_RADAR_CATEGORIES if any(category in m["variation_category_strength"] for m in models)]
    # Base-face radar plots removed per user request (no longer needed)
    _plot_radar_comparison(
        models,
        variation_radar_categories,
        "variation_category_strength",
        "Variation Category Shift Strength (Radar)",
        plots_dir / "variation_category_radar.png",
        "weighted mean abs(mean delta) by variation category",
    )
    _plot_radar_comparison(
        models,
        variation_radar_categories,
        "variation_category_strength",
        "Variation Category Shift Strength",
        plots_dir / "variation_category_radar_raw.png",
        "weighted mean abs(mean delta) by variation category",
        normalize_axes=False,
    )

    summary_lines = []
    summary_lines.append("Model Evaluation Comparison Summary")
    summary_lines.append("=" * 38)
    summary_lines.append("")
    summary_lines.append(f"models compared: {', '.join(m['name'] for m in models)}")
    summary_lines.append(f"scenarios covered: {len(scenario_ids)}")
    summary_lines.append("")

    summary_lines.append("Global Metrics")
    summary_lines.append("-" * 14)
    for m in models:
        o = m["overall"]
        summary_lines.append(
            (
                f"{m['name']}: pairs={o['n_pairs']}, mean_delta={o['mean_delta']:.6f}, "
                f"std_delta={o['std_delta']:.6f}, cohens_d={o['cohens_d']:.6f}, "
                f"wilcoxon_p={o['wilcoxon_p']:.3e}"
            )
        )
    summary_lines.append("")

    summary_lines.append("Variation Robustness")
    summary_lines.append("-" * 20)
    for m in models:
        d = m["distribution"]
        summary_lines.append(
            (
                f"{m['name']}: zero={d['zero_prop']:.4f}, |delta|>=0.25={d['abs_ge_0_25']:.4f}, "
                f"|delta|>=0.5={d['abs_ge_0_5']:.4f}, |delta|>=0.75={d['abs_ge_0_75']:.4f}"
            )
        )
    summary_lines.append("")

    summary_lines.append("Scenario Significance")
    summary_lines.append("-" * 21)
    for m in models:
        summary_lines.append(
            (
                f"{m['name']}: significant={len(m['significant_scenarios'])}/{len(m['scenario_metrics'])}; "
                f"non-significant={m['nonsignificant_scenarios']}"
            )
        )
    summary_lines.append("")

    summary_lines.append("Base-Face Category Variation Strength")
    summary_lines.append("(How much judgments differ across demographic groups)")
    summary_lines.append("-" * 39)
    base_face_categories_sorted = sorted({cat for m in models for cat in m["base_face_category_variation"].keys()})
    if base_face_categories_sorted:
        for cat in base_face_categories_sorted:
            summary_lines.append(f"{cat}:")
            for m in models:
                metrics = m["base_face_category_variation"].get(cat) or {}
                strength = metrics.get("strength")
                ci_lo = metrics.get("ci_lower")
                ci_hi = metrics.get("ci_upper")
                p_bh = metrics.get("wilcoxon_p_bh")
                if strength is not None and not math.isnan(strength):
                    ci_str = (
                        f" [{ci_lo:.3f}, {ci_hi:.3f}]"
                        if (ci_lo is not None and not math.isnan(ci_lo))
                        else ""
                    )
                    p_str = (
                        f" p_bh={p_bh:.3e}" if (p_bh is not None and not math.isnan(p_bh)) else ""
                    )
                    summary_lines.append(f"  {m['name']}: {strength:.6f}{ci_str}{p_str}")
    summary_lines.append("")

    summary_lines.append("Top Scenario Shifts (mean delta)")
    summary_lines.append("-" * 31)
    for m in models:
        summary_lines.append(f"{m['name']}")
        for sid, metric in m["top3_positive"]:
            summary_lines.append(
                f"  + scenario {sid} ({scenario_labels.get(sid, 'n/a')}): {metric['mean_delta']:.6f}"
            )
        for sid, metric in m["top3_negative"]:
            summary_lines.append(
                f"  - scenario {sid} ({scenario_labels.get(sid, 'n/a')}): {metric['mean_delta']:.6f}"
            )
    summary_lines.append("")

    summary_lines.append("Base-Face Bias Rates")
    summary_lines.append("-" * 20)
    for m in models:
        summary_lines.append(f"{m['name']}")
        for cat in sorted(m["bias_rates"]):
            summary_lines.append(
                (
                    f"  {cat}: {m['bias_sig_by_category'].get(cat, 0)}/"
                    f"{m['bias_denominators'].get(cat, 0)} = {m['bias_rates'][cat]:.4f}"
                )
            )
    summary_lines.append("")

    if len(models) >= 2:
        summary_lines.append("Pairwise Model Differences")
        summary_lines.append("-" * 25)
        for row in pairwise_rows_dict:
            summary_lines.append(
                (
                    f"{row['model_a']} vs {row['model_b']}: mean_delta_diff(b-a)="
                    f"{row['mean_delta_diff_b_minus_a']:.6f}, sign_flips="
                    f"{row['scenario_sign_flips']}/{row['scenario_overlap']}, "
                    f"mean_abs_scenario_diff={row['mean_abs_scenario_diff']:.6f}"
                )
            )
        summary_lines.append("")

    summary_lines.append("Paper Draft Snippet")
    summary_lines.append("-" * 19)
    summary_lines.append(
        "Across all compared models, we observe statistically significant but small average"
    )
    summary_lines.append(
        "global variation effects. Differences are primarily driven by robustness profiles"
    )
    summary_lines.append(
        "(tail behavior of delta) and scenario-specific directionality rather than by global"
    )
    summary_lines.append(
        "mean shifts alone. In particular, scenario-level sign flips between model pairs"
    )
    summary_lines.append(
        "indicate that model choice materially changes the direction of inferred social"
    )
    summary_lines.append(
        "attribute judgments under controlled visual perturbations."
    )
    summary_lines.append("")
    summary_lines.append("Radar Charts")
    summary_lines.append("-" * 12)
    summary_lines.append("variation_category_radar.png compares category-level variation strengths across variation types.")
    summary_lines.append("variation_category_radar_raw.png uses raw (unnormalized) values.")

    (output_dir / "comparison_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    md_lines = [
        "# Model Comparison Overview",
        "",
        f"Compared models: **{', '.join(m['name'] for m in models)}**",
        "",
        "## Global Overview",
        "",
        "| model | pairs | mean delta | std delta | cohen d | wilcoxon p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for m in models:
        o = m["overall"]
        md_lines.append(
            f"| {m['name']} | {o['n_pairs']} | {o['mean_delta']:.6f} | {o['std_delta']:.6f} | {o['cohens_d']:.6f} | {o['wilcoxon_p']:.3e} |"
        )

    md_lines.extend(
        [
            "",
            "## Robustness Indicators",
            "",
            "| model | zero delta | |delta|>=0.25 | |delta|>=0.5 | |delta|>=0.75 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for m in models:
        d = m["distribution"]
        md_lines.append(
            f"| {m['name']} | {d['zero_prop']:.4f} | {d['abs_ge_0_25']:.4f} | {d['abs_ge_0_5']:.4f} | {d['abs_ge_0_75']:.4f} |"
        )

    md_lines.extend(
        [
            "",
            "## Files",
            "",
            "- `model_overview.csv`: global metrics per model",
            "- `scenario_comparison.csv`: scenario-level means/effects/p-values per model",
            "- `category_bias_rates.csv`: significant bias rates by category",
            "- `base_face_category_variation_strength.csv`: demographic category variation strength comparison",
            "- `variation_category_strength.csv`: raw variation-category strengths used by radar plots",
            "- `pairwise_model_comparison.csv`: pairwise model deltas and sign flips",
            "- `plots/base_face_category_variation_strength.png`: bar chart showing demographic effect strength across models",
            "- `plots/variation_category_radar.png`: radar chart for variation-category shift strength",
            "- `plots/variation_category_radar_raw.png`: raw-value radar chart for variation-category shift strength",
            "- `plots/`: cross-model visual comparisons",
        ]
    )
    (output_dir / "comparison_overview.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Compare evaluation outputs across models.")
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=Path("output/evaluation"),
        help="Root directory containing per-model evaluation folders.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional list of model folder names to include. By default, include all valid folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional explicit output directory. If omitted, a timestamped folder is created.",
    )
    args = parser.parse_args()

    if not args.evaluation_root.exists():
        raise FileNotFoundError(f"evaluation root not found: {args.evaluation_root}")

    model_dirs = _discover_model_dirs(args.evaluation_root, args.models)
    if len(model_dirs) < 2:
        raise RuntimeError(
            "Need at least 2 model folders with required files for comparison. "
            f"Found: {[p.name for p in model_dirs]}"
        )

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is None:
        output_dir = args.evaluation_root / f"model_comparison_{timestamp}"
    else:
        output_dir = args.output_dir

    models = [_parse_model(path) for path in model_dirs]
    _write_outputs(models, output_dir)

    # Demographic sensitivity table
    sensitivity = _compute_demographic_sensitivity(model_dirs)
    _write_demographic_sensitivity(sensitivity, output_dir)

    print(f"Compared models: {[m['name'] for m in models]}")
    print(f"Saved comparison outputs to: {output_dir}")


if __name__ == "__main__":
    main()
