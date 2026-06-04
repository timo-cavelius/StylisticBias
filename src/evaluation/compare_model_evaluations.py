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


# ... remainder unchanged (moved from root; original file retained as shim)
