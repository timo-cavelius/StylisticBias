#!/usr/bin/env python3
"""Plot per-model spread of variation shift magnitudes.

Loads `paired_deltas.csv` per model and plots box/violin summaries side-by-side
to compare dispersion between models.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np


EVAL_ROOT = Path("output/evaluation")
OUT_DIR = Path("output/evaluation/eval_charts")


def _model_dirs(root: Path) -> List[Path]:
    return [p for p in sorted(root.iterdir()) if p.is_dir() and (p / "paired_deltas.csv").exists()]


def _load_abs_deltas(model_dir: Path) -> List[float]:
    vals = []
    with (model_dir / "paired_deltas.csv").open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            vals.append(abs(float(r.get("delta", 0.0))))
    return vals


def main() -> None:
    models = _model_dirs(EVAL_ROOT)
    if not models:
        raise FileNotFoundError("No models found")
    data = [np.array(_load_abs_deltas(m)) for m in models]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.violinplot(data, showextrema=False)
    ax.set_xticks(range(1, len(models) + 1))
    ax.set_xticklabels([m.name for m in models], rotation=45, ha="right")
    out = OUT_DIR / "variation_shift_model_spread.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Plot per-model spread of variation shift magnitudes.

Loads `paired_deltas.csv` per model and plots box/violin summaries side-by-side
to compare dispersion between models.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np


EVAL_ROOT = Path("output/evaluation")
OUT_DIR = Path("output/evaluation/eval_charts")


def _model_dirs(root: Path) -> List[Path]:
    return [p for p in sorted(root.iterdir()) if p.is_dir() and (p / "paired_deltas.csv").exists()]


def _load_abs_deltas(model_dir: Path) -> List[float]:
    vals = []
    with (model_dir / "paired_deltas.csv").open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            vals.append(abs(float(r.get("delta", 0.0))))
    return vals


def main() -> None:
    models = _model_dirs(EVAL_ROOT)
    if not models:
        raise FileNotFoundError("No models found")
    data = [np.array(_load_abs_deltas(m)) for m in models]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.violinplot(data, showextrema=False)
    ax.set_xticks(range(1, len(models) + 1))
    ax.set_xticklabels([m.name for m in models], rotation=45, ha="right")
    out = OUT_DIR / "variation_shift_model_spread.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
