"""Re-test significance for Table 3 and the big detailed table using independent observations.

Table 3: face-level aggregation (n ≈ 130–320 per cell), with bootstrap CIs and Cohen's d.
  - Reviewer recommendation: face is the experimental unit for appearance-variation effects.
  - Each base face contributes one mean Δ per cell; Wilcoxon runs on those face means.

Big detailed table: scenario-level aggregation (n ≤ 25 per cell).
  - More conservative; tests generalizability across decision contexts.

BH correction applied across all cells within each table.
"""

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "output" / "evaluation"
OUT_DIR = ROOT / "output" / "evaluation" / "model_comparison_20260503_041215"

MODELS = ["gemma3", "gemma4", "internvl", "llava_next", "pixtral", "qwen3"]
OFFICIAL_SCENARIOS = set(range(1, 26))

# Table 3 category groupings
TABLE3_CATEGORIES = {
    "Fashion":         lambda v: v.startswith("fashion_style:"),
    "Facial hair":     lambda v: v.startswith("facial_hair_male:"),
    "Eyewear":         lambda v: v.startswith("eyewear:"),
    "Makeup & lips":   lambda v: v.startswith("makeup_female:") or v.startswith("lip_makeup_female:"),
    "Tattoos":         lambda v: v.startswith("tattoos:"),
    "Hair style":      lambda v: v.startswith("hair_style:"),
    "Skin irreg.":     lambda v: v.startswith("skin_irregularities:"),
    "Hair len./color": lambda v: v.startswith("hair_length:") or v.startswith("hair_color:"),
    "Accessories":     lambda v: v.startswith("accessories:"),
    "Piercings":       lambda v: v.startswith("piercings:"),
}

# Big table: demographic dimension → column values
DEMO_DIMS = {
    "Age":    ("age",       ["young adult", "middle-aged adult", "elderly"]),
    "Gender": ("gender",    ["male", "female"]),
    "Ethn.":  ("ethnicity", ["Asian", "African", "European", "Middle Eastern", "Latino"]),
    "Body":   ("body_index",["thin", "normal", "obese"]),
}



def bh_correct(p_values):
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adj = [1.0] * n
    prev = 1.0
    for rank, (i, p) in enumerate(reversed(indexed), 1):
        a = min(p * n / (n - rank + 1), prev)
        prev = a
        adj[i] = a
    return adj


def wilcoxon_vs_zero(vals):
    """One-sample Wilcoxon signed-rank vs zero (two-sided)."""
    vals = [v for v in vals if not math.isnan(v)]
    if len(vals) < 3:
        return float("nan")
    try:
        _, p = scipy_stats.wilcoxon(vals, zero_method="wilcox", alternative="two-sided")
        return float(p)
    except Exception:
        return float("nan")


def cohens_d(vals):
    """Cohen's d for one-sample test vs zero: d = mean / SD."""
    vals = [v for v in vals if not math.isnan(v)]
    if len(vals) < 2:
        return float("nan")
    m = np.mean(vals)
    sd = np.std(vals, ddof=1)
    if sd == 0:
        return float("nan")
    return float(m / sd)


def bootstrap_ci(vals, n_boot=1000, seed=42):
    """95% bootstrap CI of the mean via percentile method."""
    vals = [v for v in vals if not math.isnan(v)]
    if not vals:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    arr = np.asarray(vals, dtype=float)
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    boot_means = [float(np.mean(rng.choice(arr, size=arr.size, replace=True)))
                  for _ in range(n_boot)]
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def bootstrap_unweighted_mean(subgroup_lists, n_boot=1000, seed=42):
    """Bootstrap 95% CI for the unweighted mean of subgroup means.

    Resamples face means within each subgroup independently, then takes the
    unweighted average across subgroups — matching how the reported mean is
    computed so that the CI actually brackets it.
    """
    rng = np.random.default_rng(seed)
    arrays = [np.asarray(fm, dtype=float) for fm in subgroup_lists if fm]
    if not arrays:
        return float("nan"), float("nan")
    boot = []
    for _ in range(n_boot):
        sub_means = [float(np.mean(rng.choice(a, size=len(a), replace=True))) for a in arrays]
        boot.append(float(np.mean(sub_means)))
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def load_all_deltas():
    """Load paired_deltas.csv for all models, filter to official scenarios."""
    rows = []
    for model in MODELS:
        path = EVAL_DIR / model / "paired_deltas.csv"
        if not path.exists():
            print(f"  WARNING: {path} not found")
            continue
        with open(path) as f:
            for row in csv.DictReader(f):
                scen = int(row["scenario"])
                if scen not in OFFICIAL_SCENARIOS:
                    continue
                row["model"] = model
                rows.append(row)
    print(f"Loaded {len(rows):,} observations across {len(MODELS)} models")
    return rows


def aggregate_to_scenarios(rows, variation_filter, demo_col, demo_val=None):
    """Return list of per-scenario mean deltas for a given cell."""
    scen_deltas = defaultdict(list)
    for row in rows:
        if not variation_filter(row["variation_name"]):
            continue
        if demo_val is not None and row.get(demo_col) != demo_val:
            continue
        try:
            delta = float(row["delta"])
        except (ValueError, TypeError):
            continue
        scen_deltas[int(row["scenario"])].append(delta)
    return [np.mean(v) for v in scen_deltas.values() if v]


def aggregate_to_faces(rows, variation_filter, demo_col, demo_val=None):
    """Return list of per-face mean deltas for a given cell.

    Each unique base face contributes one value: its mean Δ across all
    scenarios, models, and matching variations. This is the face-level
    independent unit recommended for Table 3 (reviewer comment).
    """
    face_deltas = defaultdict(list)
    for row in rows:
        if not variation_filter(row["variation_name"]):
            continue
        if demo_val is not None and row.get(demo_col) != demo_val:
            continue
        face = (row.get("face_folder") or "").strip()
        if not face:
            continue
        try:
            delta = float(row["delta"])
        except (ValueError, TypeError):
            continue
        face_deltas[face].append(delta)
    return [float(np.mean(v)) for v in face_deltas.values() if v]


# ── Table 3 ──────────────────────────────────────────────────────────────────

def check_table3(rows):
    """Face-level aggregation for Table 3 (reviewer recommendation).

    Reported mean : unweighted average of per-subgroup face means — each
        demographic subgroup (e.g. young adult, middle-aged, elderly for Age)
        contributes equally regardless of its size, so the column values
        differ across Age / Gender / Ethn. / Body.

    Significance   : Wilcoxon signed-rank on the pooled face means (all faces
        with a label in this dimension), i.e. face-level as recommended.
        n_faces ≈ 130–340 per cell.

    Cohen's d / bootstrap CI also computed on the pooled face means.
    BH correction applied across all 40 cells simultaneously.
    """
    print("\n" + "=" * 70)
    print("TABLE 3  –  Face-level Wilcoxon, unweighted subgroup mean")
    print("=" * 70)

    # cells: (category, demo_label, reported_mean, pooled_face_means, subgroup_lists)
    cells = []
    for cat_label, vfilter in TABLE3_CATEGORIES.items():
        for demo_label, (demo_col, demo_vals) in DEMO_DIMS.items():
            subgroup_lists: list[list[float]] = []
            pooled: list[float] = []
            for demo_val in demo_vals:
                fm = aggregate_to_faces(rows, vfilter, demo_col, demo_val)
                if fm:
                    subgroup_lists.append(fm)
                    pooled.extend(fm)
            subgroup_means = [float(np.mean(s)) for s in subgroup_lists]
            reported = float(np.mean(subgroup_means)) if subgroup_means else float("nan")
            cells.append((cat_label, demo_label, reported, pooled, subgroup_lists))

    # BH correction on face-level Wilcoxon p-values
    raw_ps = [wilcoxon_vs_zero(c[3]) for c in cells]
    adj_ps = bh_correct([p if not math.isnan(p) else 1.0 for p in raw_ps])

    csv_rows = []
    not_sig = []
    for (cat, demo, reported, pooled, subgroup_lists), p_raw, p_adj in zip(cells, raw_ps, adj_ps):
        n_faces = len(pooled)
        ci_lo, ci_hi = bootstrap_unweighted_mean(subgroup_lists)
        d = cohens_d(pooled)
        sig = "***" if p_adj < 0.001 else ("**" if p_adj < 0.01 else ("*" if p_adj < 0.05 else "ns"))
        csv_rows.append({
            "category": cat,
            "demographic": demo,
            "n_faces": n_faces,
            "mean_delta_unweighted": round(reported, 5),
            "ci_lower_boot": round(ci_lo, 5) if not math.isnan(ci_lo) else float("nan"),
            "ci_upper_boot": round(ci_hi, 5) if not math.isnan(ci_hi) else float("nan"),
            "cohens_d": round(d, 4) if not math.isnan(d) else float("nan"),
            "p_raw": float(f"{p_raw:.6e}"),
            "p_adj_bh": float(f"{p_adj:.6e}"),
            "sig": sig,
        })
        if sig == "ns":
            not_sig.append((cat, demo))

    # Save CSV
    out_path = OUT_DIR / "table3_significance_independent.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nSaved → {out_path}")

    print(f"\n{'Category':<20} {'Demo':<10} {'n_faces':>8} {'mean_Δ':>8} {'Cohen_d':>8} "
          f"{'p_raw':>10} {'p_adj':>10} {'sig':>5}")
    print("-" * 86)
    for r in csv_rows:
        flag = " ←" if r["sig"] == "ns" else ""
        d_str = f"{r['cohens_d']:>8.3f}" if not math.isnan(float(r["cohens_d"])) else "     nan"
        print(f"{r['category']:<20} {r['demographic']:<10} {r['n_faces']:>8} "
              f"{r['mean_delta_unweighted']:>8.4f} {d_str} {r['p_raw']:>10.2e} "
              f"{r['p_adj_bh']:>10.2e} {r['sig']:>5}{flag}")

    print(f"\nNot significant: {len(not_sig)}/40 cells")
    return not_sig


# ── Big detailed table ────────────────────────────────────────────────────────

def check_big_table(rows):
    """Face-level aggregation for the detailed variation table.

    Unit of independence = base face. Each face contributes one mean Δ for a
    given variation × demographic value cell (averaged across all its scenarios
    and models). Matching Table 3's approach per reviewer recommendation.
    """
    print("\n" + "=" * 70)
    print("BIG TABLE  –  Face-level independent test (n = faces per cell)")
    print("=" * 70)

    all_variations = sorted(set(r["variation_name"] for r in rows))

    cells = []  # (var_name, demo_label, demo_val, face_means)
    for var in all_variations:
        vfilter = lambda vn, v=var: vn == v
        for demo_label, (demo_col, demo_vals) in DEMO_DIMS.items():
            for demo_val in demo_vals:
                face_means = aggregate_to_faces(rows, vfilter, demo_col, demo_val)
                cells.append((var, demo_label, demo_val, face_means))

    # BH correction across all cells simultaneously
    raw_ps = [wilcoxon_vs_zero(c[3]) for c in cells]
    adj_ps = bh_correct([p if not math.isnan(p) else 1.0 for p in raw_ps])

    csv_rows = []
    not_sig = []
    for (var, demo, val, face_means), p_raw, p_adj in zip(cells, raw_ps, adj_ps):
        if not face_means:
            continue
        n = len(face_means)
        mean = float(np.mean(face_means))
        ci_lo, ci_hi = bootstrap_ci(face_means)
        d = cohens_d(face_means)
        sig = "***" if p_adj < 0.001 else ("**" if p_adj < 0.01 else ("*" if p_adj < 0.05 else "ns"))
        csv_rows.append({
            "variation": var,
            "demographic_dimension": demo,
            "demographic_value": val,
            "n_faces": n,
            "mean_delta": round(mean, 5),
            "ci_lower_boot": round(ci_lo, 5) if not math.isnan(ci_lo) else float("nan"),
            "ci_upper_boot": round(ci_hi, 5) if not math.isnan(ci_hi) else float("nan"),
            "cohens_d": round(d, 4) if not math.isnan(d) else float("nan"),
            "p_raw": float(f"{p_raw:.6e}"),
            "p_adj_bh": float(f"{p_adj:.6e}"),
            "sig": sig,
        })
        if sig == "ns":
            not_sig.append((var, demo, val))

    # Save CSV
    out_path = OUT_DIR / "big_table_significance_independent.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved → {out_path}")
    print(f"Not significant: {len(not_sig)}/{len(csv_rows)} cells")

    return not_sig


def main():
    rows = load_all_deltas()

    not_sig_t3 = check_table3(rows)
    not_sig_big = check_big_table(rows)

    print(f"\nDone. CSVs saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
