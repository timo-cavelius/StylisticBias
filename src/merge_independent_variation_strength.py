#!/usr/bin/env python3
"""Merge all model-level category_variation_strength_independent.csv files.

Output:
  output/evaluation/model_comparison_20260503_041215/
  category_variation_strength_independent_all_models.csv
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "output" / "evaluation"
OUT_PATH = (
    EVAL_DIR
    / "model_comparison_20260503_041215"
    / "category_variation_strength_independent_all_models.csv"
)

MODELS = ["gemma3", "gemma4", "internvl", "llava_next", "pixtral", "qwen3"]


def main() -> None:
    rows: list[dict[str, str]] = []

    for model in MODELS:
        path = EVAL_DIR / model / "category_variation_strength_independent.csv"
        if not path.exists():
            print(f"[warn] missing: {path}")
            continue

        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append(
                    {
                        "model": model,
                        "category_type": row.get("category_type", ""),
                        "variation_strength": row.get("variation_strength", ""),
                        "n_scenarios": row.get("n_scenarios", ""),
                        "n_faces": row.get("n_faces", ""),
                        "ci_lower_95": row.get("ci_lower_95", ""),
                        "ci_upper_95": row.get("ci_upper_95", ""),
                        "perm_p": row.get("perm_p", ""),
                        "perm_p_bh": row.get("perm_p_bh", ""),
                        "test_method": row.get("test_method", ""),
                    }
                )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "category_type",
                "variation_strength",
                "n_scenarios",
                "n_faces",
                "ci_lower_95",
                "ci_upper_95",
                "perm_p",
                "perm_p_bh",
                "test_method",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved: {OUT_PATH}")
    print(f"rows: {len(rows)}")


if __name__ == "__main__":
    main()
