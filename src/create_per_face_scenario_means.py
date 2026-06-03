#!/usr/bin/env python3
"""For each model, write a CSV with per-face-per-scenario mean p(option_a).

Columns per output row:
  face_folder, age, gender, ethnicity, body_type, scenario,
  base_p_option_a,
  <category>_mean_p   — one column per variation category,
                        = mean p(option_a) across all variations in that
                          category applied to this face in this scenario.

Sources (inside output/evaluation/<model>/):
  base_faces_probability_scores.csv  — base face scores
  paired_deltas.csv                  — per-variation scores (variation_score)

Output: output/evaluation/<model>/per_face_scenario_means.csv

Usage:
  python3 src/create_per_face_scenario_means.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EVALUATION_ROOT = Path("output/evaluation")
MODELS = ["gemma3", "gemma4", "internvl", "llava_next", "pixtral", "qwen3"]

# Fixed category order for columns
VARIATION_CATEGORIES = [
    "accessories",
    "eyewear",
    "facial_hair_male",
    "fashion_style",
    "hair_color",
    "hair_length",
    "hair_style",
    "lip_makeup_female",
    "makeup_female",
    "piercings",
    "skin_irregularities",
    "tattoos",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_base_scores(path: Path) -> dict[tuple[str, str], dict]:
    """Return {(face_folder, scenario): row_dict} from base_faces_probability_scores."""
    result: dict[tuple[str, str], dict] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = (row["face_folder"], row["scenario"])
            result[key] = {
                "face_folder":   row["face_folder"],
                "age":           row["age"],
                "gender":        row["gender"],
                "ethnicity":     row["ethnicity"],
                "body_type":     row["body_type"],
                "scenario":      row["scenario"],
                "base_p_option_a": row["p_option_a"],
            }
    return result


def _load_variation_means(path: Path) -> dict[tuple[str, str, str], float]:
    """Return {(face_folder, scenario, category): mean_variation_p} from paired_deltas."""
    # Accumulate: (face_folder, scenario, category) → [variation_score, ...]
    acc: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            cat = row["variation_name"].partition(":")[0]
            key = (row["face_folder"], row["scenario"], cat)
            try:
                acc[key].append(float(row["variation_score"]))
            except (ValueError, KeyError):
                pass
    return {k: sum(v) / len(v) for k, v in acc.items()}


def _process_model(model: str) -> None:
    model_dir = EVALUATION_ROOT / model
    base_path  = model_dir / "base_faces_probability_scores.csv"
    delta_path = model_dir / "paired_deltas.csv"

    if not base_path.exists() or not delta_path.exists():
        print(f"  [{model}] Missing files — skipped.")
        return

    print(f"  [{model}] Loading base scores …")
    base_scores = _load_base_scores(base_path)

    print(f"  [{model}] Loading variation scores …")
    var_means = _load_variation_means(delta_path)

    # Build output rows
    fieldnames = (
        ["face_folder", "age", "gender", "ethnicity", "body_type",
         "scenario", "base_p_option_a"]
        + [f"{cat}_mean_p" for cat in VARIATION_CATEGORIES]
    )

    out_path = model_dir / "per_face_scenario_means.csv"
    n_rows = 0
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (face, scenario), base_row in sorted(base_scores.items()):
            out: dict = dict(base_row)
            for cat in VARIATION_CATEGORIES:
                val = var_means.get((face, scenario, cat))
                out[f"{cat}_mean_p"] = f"{val:.6f}" if val is not None else ""
            writer.writerow(out)
            n_rows += 1

    print(f"  [{model}] Wrote {n_rows} rows → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    for model in MODELS:
        _process_model(model)
    print("Done.")


if __name__ == "__main__":
    main()
