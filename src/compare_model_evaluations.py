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
import numpy as np


REQUIRED_FILES = [
    "paired_delta_statistics.json",
    "variation_impact_summary.csv",
    "base_faces_significant_biases.json",
    "base_faces_category_scenario_summary.csv",
    "base_faces_counts.json",
    "paired_deltas.csv",
]


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
        "faces_used": _safe_int(base_summary_txt.get("faces_used_excluding_thin_latino"), 0),
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

    fig, ax = plt.subplots(figsize=(14, 4 + 0.5 * len(models)))
    vmax = np.nanmax(np.abs(arr)) if np.isfinite(arr).any() else 1.0
    vmax = max(vmax, 1e-6)
    im = ax.imshow(arr, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("mean delta")

    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels([m["name"] for m in models])

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
    ax.set_xticklabels(short_x_labels, rotation=70, ha="right", fontsize=8)

    ax.set_title("Scenario-Level Mean Delta Across Models")
    ax.set_xlabel("scenario")
    ax.set_ylabel("model")

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
    _plot_category_bias_rates(models, plots_dir / "category_bias_rate_comparison.png")
    _plot_gender_abs_delta(models, plots_dir / "gender_weighted_abs_delta.png")

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
            "- `pairwise_model_comparison.csv`: pairwise model deltas and sign flips",
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

    print(f"Compared models: {[m['name'] for m in models]}")
    print(f"Saved comparison outputs to: {output_dir}")


if __name__ == "__main__":
    main()
