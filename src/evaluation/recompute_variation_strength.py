"""Recompute category_variation_strength.csv for all models using only the 25 official scenarios.

Moved from root `src/recompute_variation_strength.py`.
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats


ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "output" / "evaluation"

CATEGORY_VALUES = {
    "age": ["young adult", "middle-aged adult", "elderly"],
    "gender": ["male", "female"],
    "ethnicity": ["Asian", "African", "European", "Middle Eastern", "Latino"],
    "body_type": ["normal", "obese", "thin"],
}

OFFICIAL_SCENARIOS = set(range(1, 26))


def _mean(values):
    if not values:
        return None
    return sum(values) / len(values)


def _std(values):
    if len(values) < 2:
        return None
    m = _mean(values)
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


def _bh_correction(p_values):
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


def _compute_variation_strength(rows, n_bootstrap=500, rng=None):
    """rows: list of dicts with keys scenario, age, gender, ethnicity, body_type, p_option_a."""
    if rng is None:
        rng = np.random.default_rng(42)

    # face_data[category_type][scenario][category_value] = [p_option_a, ...]
    face_data = {cat: defaultdict(lambda: defaultdict(list)) for cat in CATEGORY_VALUES}

    for row in rows:
        scenario = int(row["scenario"])
        if scenario not in OFFICIAL_SCENARIOS:
            continue
        p = row.get("p_option_a")
        if p is None:
            continue
        for cat, allowed in CATEGORY_VALUES.items():
            val = row.get(cat)
            if val in allowed:
                face_data[cat][scenario][val].append(float(p))

    result = {}
    raw_p_values = {}

    for cat, scen_map in face_data.items():
        scenario_stds = []
        per_scenario = {}

        for scen in sorted(scen_map):
            val_faces = scen_map[scen]
            if len(val_faces) < 2:
                continue
            means = [_mean(v) for v in val_faces.values() if _mean(v) is not None]
            if len(means) < 2:
                continue
            s = _std(means)
            if s is not None:
                per_scenario[scen] = s
                scenario_stds.append(s)

        avg = _mean(scenario_stds)

        # Bootstrap CI
        ci_lower = ci_upper = None
        if scenario_stds:
            boot = []
            for _ in range(n_bootstrap):
                boot_stds = []
                for scen, val_faces in scen_map.items():
                    if len(val_faces) < 2:
                        continue
                    boot_means = []
                    for pvals in val_faces.values():
                        sampled = rng.choice(pvals, size=len(pvals), replace=True)
                        boot_means.append(float(np.mean(sampled)))
                    if len(boot_means) >= 2:
                        s = _std(boot_means)
                        if s is not None:
                            boot_stds.append(s)
                if boot_stds:
                    v = _mean(boot_stds)
                    if v is not None:
                        boot.append(v)
            if boot:
                ci_lower = float(np.percentile(boot, 2.5))
                ci_upper = float(np.percentile(boot, 97.5))

        # Wilcoxon signed-rank vs zero (one-sided)
        wilcoxon_p = None
        if len(scenario_stds) >= 2:
            try:
                _, wilcoxon_p = scipy_stats.wilcoxon(
                    scenario_stds, zero_method="wilcox", alternative="greater"
                )
                wilcoxon_p = float(wilcoxon_p)
            except Exception:
                wilcoxon_p = None

        if wilcoxon_p is not None:
            raw_p_values[cat] = wilcoxon_p

        result[cat] = {
            "variation_strength": avg,
            "n_scenarios": len(scenario_stds),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "wilcoxon_p": wilcoxon_p,
        }

    # BH correction within model across categories
    cats_with_p = [c for c in result if raw_p_values.get(c) is not None]
    if cats_with_p:
        pvals = [raw_p_values[c] for c in cats_with_p]
        adj = _bh_correction(pvals)
        for c, a in zip(cats_with_p, adj):
            result[c]["wilcoxon_p_bh"] = a
    for cat in result:
        if "wilcoxon_p_bh" not in result[cat]:
            result[cat]["wilcoxon_p_bh"] = None

    return result


def _write_csv(path, header, rows):
    with open(path, "w") as f:
        f.write(",".join(str(h) for h in header) + "\n")
        for row in rows:
            f.write(",".join("" if v is None else str(v) for v in row) + "\n")


def main():
    models = [d.name for d in EVAL_DIR.iterdir() if d.is_dir() and d.name != "model_comparison_20260503_041215"]
    models.sort()

    print(f"Found models: {models}")

    for model in models:
        prob_file = EVAL_DIR / model / "base_faces_probability_scores.json"
        if not prob_file.exists():
            print(f"  [{model}] SKIP — no probability scores file")
            continue

        with open(prob_file) as f:
            rows = json.load(f)

        print(f"  [{model}] {len(rows)} rows loaded, recomputing...")
        result = _compute_variation_strength(rows)

        out_rows = []
        for cat in sorted(result):
            m = result[cat]
            out_rows.append([
                cat,
                m["variation_strength"],
                m["n_scenarios"],
                m["ci_lower"],
                m["ci_upper"],
                m["wilcoxon_p"],
                m["wilcoxon_p_bh"],
            ])

        out_path = EVAL_DIR / model / "category_variation_strength.csv"
        _write_csv(
            out_path,
            ["category_type", "variation_strength", "n_scenarios",
             "ci_lower_95", "ci_upper_95", "wilcoxon_p", "wilcoxon_p_bh"],
            out_rows,
        )
        for cat, m in sorted(result.items()):
            print(f"    {cat}: strength={m['variation_strength']:.4f}, n={m['n_scenarios']}, "
                  f"CI=[{m['ci_lower']:.4f}, {m['ci_upper']:.4f}], p_bh={m['wilcoxon_p_bh']:.2e}")

    print("\nDone. category_variation_strength.csv updated for all models.")


if __name__ == "__main__":
    main()
