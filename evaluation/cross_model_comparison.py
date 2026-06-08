#!/usr/bin/env python3
"""Cross-model comparison visualization utilities.

This module collects model-level summaries and draws a compact comparison strip
chart showing median absolute |Δ| per variation category for each model.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


EVAL_ROOT = Path("output/evaluation")
OUT_DIR  = Path("output/evaluation/eval_charts")


def _model_dirs(root: Path) -> List[Path]:
    return [p for p in sorted(root.iterdir()) if p.is_dir() and (p / "variation_impact_summary.csv").exists()]


def load_per_model_medians(model_dir: Path) -> Dict[str, float]:
    med = {}
    with (model_dir / "variation_impact_summary.csv").open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            med[r["variation_name"]] = abs(float(r["mean_delta"]))
    return med


def plot_comparison(out_path: Path) -> None:
    models = _model_dirs(EVAL_ROOT)
    if not models:
        raise FileNotFoundError("No model evaluation dirs found")

    # collect set of top-k variations across models
    per_model = {m.name: load_per_model_medians(m) for m in models}
    all_vars = sorted({v for d in per_model.values() for v in d.keys()}, key=lambda s: s)

    # Build matrix: models x variables
    mat = np.zeros((len(models), len(all_vars)), dtype=float)
    for i, m in enumerate(models):
        for j, v in enumerate(all_vars):
            mat[i, j] = per_model[m.name].get(v, 0.0)

    # plot heatmap-like strip
    fig, ax = plt.subplots(figsize=(12, len(models) * 0.35 + 2.8))
    im = ax.imshow(mat, aspect="auto", cmap="viridis_r")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([m.name for m in models])
    ax.set_xticks(range(len(all_vars)))
    ax.set_xticklabels([v.split(":", 1)[-1].strip() for v in all_vars], rotation=45, ha="right")
    fig.colorbar(im, ax=ax, label="|mean Δ|")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    plot_comparison(OUT_DIR / "cross_model_comparison.png")


if __name__ == "__main__":
    main()
