#!/usr/bin/env python3
"""Recompute category variation strength with face-level independent p-values.

The effect size is kept identical to the existing Table 2 pipeline:
for each category type, compute the per-scenario std of the category-value
means and average those stds across scenarios.

The p-value is made stricter by using a face-level label permutation test.
For each permutation, demographic labels are shuffled across faces while
preserving the scenario structure and recomputing the same statistic.

Outputs:
  output/evaluation/<model>/category_variation_strength_independent.csv
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "output" / "evaluation"

MODEL_ORDER = ["gemma3", "gemma4", "internvl", "llava_next", "pixtral", "qwen3"]

CATEGORY_VALUES = {
    "age": ["young adult", "middle-aged adult", "elderly"],
    "gender": ["male", "female"],
    "ethnicity": ["Asian", "African", "European", "Middle Eastern", "Latino"],
    "body_type": ["normal", "obese", "thin"],
}

OFFICIAL_SCENARIOS = set(range(1, 26))
N_BOOTSTRAP = 500
N_PERMUTATIONS = 1000


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    m = _mean(values)
    assert m is not None
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


def _bh_correction(p_values: list[float]) -> list[float]:
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [1.0] * n
    prev = 1.0
    for rank, (idx, p) in enumerate(reversed(indexed), 1):
        adj = min(p * n / (n - rank + 1), prev)
        prev = adj
        adjusted[idx] = adj
    return adjusted


def _read_model_rows(model: str) -> list[dict]:
    path = EVAL_DIR / model / "base_faces_probability_scores.csv"
    if not path.exists():
        return []

    rows: list[dict] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                scenario = int(row["scenario"])
                p_option_a = float(row["p_option_a"])
            except Exception:
                continue
            if scenario not in OFFICIAL_SCENARIOS:
                continue

            rows.append(
                {
                    "face_folder": (row.get("face_folder") or "").strip(),
                    "scenario": scenario,
                    "p_option_a": p_option_a,
                    "age": (row.get("age") or "").strip(),
                    "gender": (row.get("gender") or "").strip(),
                    "ethnicity": (row.get("ethnicity") or "").strip(),
                    "body_type": (row.get("body_type") or "").strip(),
                }
            )
    return rows


def _scenario_face_map(rows: list[dict]) -> dict[int, dict[str, dict[str, float]]]:
    """Return scenario -> face -> row with demographics and p_option_a."""
    result: dict[int, dict[str, dict[str, float]]] = defaultdict(dict)
    for row in rows:
        face = row["face_folder"]
        if not face:
            continue
        result[row["scenario"]][face] = row
    return result


def _compute_strength_for_labels(
    scenario_faces: dict[int, dict[str, dict[str, float]]],
    labels_by_face: dict[str, str],
    category_values: list[str],
) -> tuple[float | None, dict[int, float]]:
    per_scenario: dict[int, float] = {}
    scenario_stds: list[float] = []

    for scenario, face_rows in sorted(scenario_faces.items()):
        groups: dict[str, list[float]] = defaultdict(list)
        for face, row in face_rows.items():
            label = labels_by_face.get(face)
            if label in category_values:
                groups[label].append(float(row["p_option_a"]))

        means = [(_mean(vals)) for vals in groups.values() if _mean(vals) is not None]
        if len(means) < 2:
            continue
        std_val = _std(means)
        if std_val is not None:
            per_scenario[scenario] = std_val
            scenario_stds.append(std_val)

    return (_mean(scenario_stds), per_scenario)


def _bootstrap_ci(
    scenario_faces: dict[int, dict[str, dict[str, float]]],
    labels_by_face: dict[str, str],
    category_values: list[str],
    rng: np.random.Generator,
    n_bootstrap: int = N_BOOTSTRAP,
) -> tuple[float | None, float | None]:
    if not scenario_faces:
        return None, None

    boot_strengths: list[float] = []
    for _ in range(n_bootstrap):
        boot_stds: list[float] = []
        for scenario, face_rows in scenario_faces.items():
            groups: dict[str, list[float]] = defaultdict(list)
            for face, row in face_rows.items():
                label = labels_by_face.get(face)
                if label in category_values:
                    groups[label].append(float(row["p_option_a"]))

            boot_means: list[float] = []
            for vals in groups.values():
                if len(vals) == 0:
                    continue
                sampled = rng.choice(np.asarray(vals, dtype=float), size=len(vals), replace=True)
                boot_means.append(float(np.mean(sampled)))

            if len(boot_means) >= 2:
                s = _std(boot_means)
                if s is not None:
                    boot_stds.append(s)

        if boot_stds:
            v = _mean(boot_stds)
            if v is not None:
                boot_strengths.append(v)

    if not boot_strengths:
        return None, None

    return (
        float(np.percentile(boot_strengths, 2.5)),
        float(np.percentile(boot_strengths, 97.5)),
    )


def _permutation_p_value(
    scenario_faces: dict[int, dict[str, dict[str, float]]],
    labels_by_face: dict[str, str],
    category_values: list[str],
    observed: float | None,
    rng: np.random.Generator,
    n_permutations: int = N_PERMUTATIONS,
) -> float | None:
    if observed is None:
        return None

    faces = sorted(labels_by_face)
    if len(faces) < 2:
        return None

    labels = np.array([labels_by_face[f] for f in faces], dtype=object)
    ge_count = 0
    n_valid = 0

    for _ in range(n_permutations):
        perm_labels = labels.copy()
        rng.shuffle(perm_labels)
        perm_map = {face: label for face, label in zip(faces, perm_labels)}
        perm_strength, _ = _compute_strength_for_labels(scenario_faces, perm_map, category_values)
        if perm_strength is None:
            continue
        n_valid += 1
        if perm_strength >= observed - 1e-12:
            ge_count += 1

    if n_valid == 0:
        return None
    return float((ge_count + 1) / (n_valid + 1))


def _compute_model_stats(rows: list[dict]) -> dict[str, dict]:
    scenario_faces = _scenario_face_map(rows)
    faces = sorted({row["face_folder"] for row in rows if row["face_folder"]})

    result: dict[str, dict] = {}
    raw_p_values: dict[str, float] = {}
    rng = np.random.default_rng(42)

    for category_type, category_values in CATEGORY_VALUES.items():
        labels_by_face = {}
        for face in faces:
            face_row = None
            for scenario in sorted(scenario_faces):
                face_row = scenario_faces[scenario].get(face)
                if face_row is not None:
                    break
            if face_row is not None:
                labels_by_face[face] = face_row.get(category_type, "")

        strength, per_scenario = _compute_strength_for_labels(scenario_faces, labels_by_face, category_values)
        ci_lower, ci_upper = _bootstrap_ci(scenario_faces, labels_by_face, category_values, rng=rng)
        p_perm = _permutation_p_value(scenario_faces, labels_by_face, category_values, strength, rng=rng)

        if p_perm is not None:
            raw_p_values[category_type] = p_perm

        result[category_type] = {
            "variation_strength": strength,
            "n_scenarios": len(per_scenario),
            "n_faces": len(labels_by_face),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_perm": p_perm,
            "p_perm_bh": None,
        }

    if raw_p_values:
        cats = list(raw_p_values)
        corrected = _bh_correction([raw_p_values[c] for c in cats])
        for cat, p_adj in zip(cats, corrected):
            result[cat]["p_perm_bh"] = p_adj

    return result


def _write_csv(path: Path, rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "category_type",
                "variation_strength",
                "n_scenarios",
                "n_faces",
                "ci_lower_95",
                "ci_upper_95",
                "perm_p",
                "perm_p_bh",
                "test_method",
            ]
        )
        writer.writerows(rows)


def main() -> None:
    models = [d.name for d in EVAL_DIR.iterdir() if d.is_dir() and d.name in MODEL_ORDER]
    models.sort(key=MODEL_ORDER.index)

    print(f"Found models: {models}")

    for model in models:
        rows = _read_model_rows(model)
        if not rows:
            print(f"  [{model}] SKIP - no base_faces_probability_scores.csv")
            continue

        print(f"  [{model}] {len(rows)} rows loaded, recomputing independent p-values...")
        result = _compute_model_stats(rows)

        out_rows = []
        for cat in sorted(result):
            m = result[cat]
            out_rows.append(
                [
                    cat,
                    m["variation_strength"],
                    m["n_scenarios"],
                    m["n_faces"],
                    m["ci_lower"],
                    m["ci_upper"],
                    m["p_perm"],
                    m["p_perm_bh"],
                    "face_label_permutation",
                ]
            )

        out_path = EVAL_DIR / model / "category_variation_strength_independent.csv"
        _write_csv(out_path, out_rows)

        for cat, m in sorted(result.items()):
            print(
                f"    {cat}: strength={m['variation_strength']:.4f}, faces={m['n_faces']}, n={m['n_scenarios']}, "
                f"CI=[{m['ci_lower']:.4f}, {m['ci_upper']:.4f}], p_perm_bh={m['p_perm_bh']:.2e}"
            )

    print("\nDone. category_variation_strength_independent.csv updated for all models.")


if __name__ == "__main__":
    main()
