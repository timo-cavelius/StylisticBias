#!/usr/bin/env python3
"""Formal interaction tests for Tables 4, 5, and 6.

Table 4: Fashion style × Age group interaction
  - Does the fashion effect differ across age groups (Young / Middle-aged / Elderly)?

Table 5: Facial tattoo × Demographic group interaction
  - Does the facial tattoo effect differ between Young vs Elderly, Male vs Female, Thin vs Obese?

Table 6: Fashion style × Body type interaction
  - Does the fashion effect differ across body types (Thin / Normal / Obese)?

Statistical approach:
  - Aggregate to face level (n = unique faces, one Δ per face per variation)
  - Tables 4 & 6: mixed-effects model (random intercept per face) with LRT for interaction
  - Table 5: pairwise Mann-Whitney U tests between demographic subgroups
  - Permutation interaction test as robustness check
  - BH correction within each table
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "output" / "evaluation"
OUT_DIR = ROOT / "output" / "evaluation" / "model_comparison_20260503_041215"

MODELS = ["gemma3", "gemma4", "internvl", "llava_next", "pixtral", "qwen3"]
OFFICIAL_SCENARIOS = set(range(1, 26))
N_PERMS = 2000

# ── Variation name maps (display → data) ──────────────────────────────────────
TABLE4_STYLES = {
    "Prof./Business":  "fashion_style:Professional / Business formal",
    "Formal/Evening":  "fashion_style:Formal / Evening wear",
    "Smart casual":    "fashion_style:Smart casual",
    "Vintage/Retro":   "fashion_style:Vintage / Retro",
    "Casual":          "fashion_style:Casual",
    "Streetwear":      "fashion_style:Streetwear",
}
TABLE6_STYLES = {
    "Prof./Business":  "fashion_style:Professional / Business formal",
    "Formal/Evening":  "fashion_style:Formal / Evening wear",
    "Smart casual":    "fashion_style:Smart casual",
    "Vintage/Retro":   "fashion_style:Vintage / Retro",
    "Worn/Distressed": "fashion_style:Worn / Distressed clothing",
}
TABLE5_VAR = "tattoos:Facial tattoo"


def _bh_correct(p_values: list[float]) -> list[float]:
    n = len(p_values)
    if n == 0:
        return []
    ordered = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [1.0] * n
    prev = 1.0
    for rank, (idx, p) in enumerate(reversed(ordered), 1):
        adj = min(p * n / (n - rank + 1), prev)
        prev = adj
        adjusted[idx] = adj
    return adjusted


def _sig_marker(p: float) -> str:
    if math.isnan(p):
        return "ns"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for two independent groups (pooled SD)."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled_sd = math.sqrt(((len(a) - 1) * np.var(a, ddof=1) +
                           (len(b) - 1) * np.var(b, ddof=1)) /
                          (len(a) + len(b) - 2))
    if pooled_sd == 0:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / pooled_sd)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_rows() -> list[dict]:
    rows = []
    for model in MODELS:
        path = EVAL_DIR / model / "paired_deltas.csv"
        if not path.exists():
            continue
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    scen = int(row["scenario"])
                    delta = float(row["delta"])
                except (ValueError, TypeError):
                    continue
                if scen not in OFFICIAL_SCENARIOS:
                    continue
                rows.append({
                    "face": (row.get("face_folder") or "").strip(),
                    "scenario": scen,
                    "variation": (row.get("variation_name") or "").strip(),
                    "delta": delta,
                    "age":    (row.get("age")        or "").strip().lower(),
                    "gender": (row.get("gender")     or "").strip().lower(),
                    "body":   (row.get("body_index") or "").strip().lower(),
                })
    print(f"Loaded {len(rows):,} rows across {len(MODELS)} models")
    return rows


def face_level_df(rows: list[dict],
                  var_filter: set[str],
                  demo_col: str,
                  demo_vals: list[str],
                  var_labels: dict[str, str]) -> pd.DataFrame:
    """Aggregate to face × variation level, keeping one Δ per face per variation.

    Each face has one demographic value (e.g., age=young adult); that face's
    Δ values are averaged across all its scenarios and models for each variation.
    Returns a DataFrame with columns: face, style_label, demo_value, delta.
    """
    label_map = {v: k for k, v in var_labels.items()}  # data_name → display_name
    cell: dict[tuple[str, str, str], list[float]] = defaultdict(list)

    for row in rows:
        if row["variation"] not in var_filter:
            continue
        demo = row.get(demo_col, "")
        if demo not in demo_vals:
            continue
        face = row["face"]
        if not face:
            continue
        var_label = label_map.get(row["variation"], row["variation"])
        cell[(face, var_label, demo)].append(row["delta"])

    records = []
    for (face, var_label, demo), deltas in cell.items():
        records.append({
            "face": face,
            "style": var_label,
            "demo": demo,
            "delta": float(np.mean(deltas)),
        })
    return pd.DataFrame(records)


# ── Interaction test via LRT ───────────────────────────────────────────────────

def ols_interaction_ftest(df: pd.DataFrame,
                          groupby_col: str = "demo") -> dict:
    """Test style × demographic interaction using OLS F-test.

    Full model:   delta ~ C(style) * C(demo)
    Reduced model: delta ~ C(style) + C(demo)

    F-test comparing the two models. Observations are face×style pairs (face-level
    aggregated, one Δ per face per style). Note: OLS assumes independence across
    rows; within-face correlation is handled separately by the permutation test,
    which should be treated as the primary inference.

    Also reports partial eta-squared for the interaction term.
    """
    df = df.dropna(subset=["delta", "style", "demo", "face"]).copy()
    if len(df) < 20:
        return {"F_stat": float("nan"), "df_num": 0, "df_denom": 0,
                "p_ols": float("nan"), "partial_eta2": float("nan"),
                "n_faces": 0, "n_obs": 0}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            full    = smf.ols(f"delta ~ C(style) * C({groupby_col})", data=df).fit()
            reduced = smf.ols(f"delta ~ C(style) + C({groupby_col})", data=df).fit()
        except Exception as e:
            return {"F_stat": float("nan"), "df_num": 0, "df_denom": 0,
                    "p_ols": float("nan"), "partial_eta2": float("nan"),
                    "n_faces": df["face"].nunique(), "n_obs": len(df), "error": str(e)}

    # Extra df from interaction terms
    df_num = int(reduced.df_resid - full.df_resid)
    df_denom = int(full.df_resid)
    if df_num <= 0 or df_denom <= 0:
        return {"F_stat": float("nan"), "df_num": df_num, "df_denom": df_denom,
                "p_ols": float("nan"), "partial_eta2": float("nan"),
                "n_faces": df["face"].nunique(), "n_obs": len(df)}

    ss_interaction = reduced.ssr - full.ssr
    ms_interaction = ss_interaction / df_num
    ms_error       = full.ssr / df_denom
    F_stat = ms_interaction / ms_error if ms_error > 0 else float("nan")
    p_ols  = float(1.0 - scipy_stats.f.cdf(F_stat, df_num, df_denom))

    # Partial eta-squared: SS_interaction / (SS_interaction + SS_residual_full)
    partial_eta2 = ss_interaction / (ss_interaction + full.ssr) if (ss_interaction + full.ssr) > 0 else float("nan")

    return {
        "F_stat": round(F_stat, 4),
        "df_num": df_num,
        "df_denom": df_denom,
        "p_ols": p_ols,
        "partial_eta2": round(partial_eta2, 4),
        "n_faces": int(df["face"].nunique()),
        "n_obs": len(df),
    }


# ── Permutation interaction test ──────────────────────────────────────────────

def permutation_interaction(df: pd.DataFrame,
                            demo_col: str = "demo",
                            n_perms: int = N_PERMS,
                            seed: int = 42) -> dict:
    """Permutation test for style × demographic interaction.

    Shuffle demographic labels across faces (preserving within-face structure).
    Test statistic: variance of per-style group-mean differences (measures interaction spread).
    """
    df = df.dropna(subset=["delta", "style", "demo", "face"]).copy()
    if df["face"].nunique() < 6 or df[demo_col].nunique() < 2:
        return {"perm_stat": float("nan"), "p_perm": float("nan")}

    rng = np.random.default_rng(seed)

    # Face → demographic mapping (each face has one label)
    face_demo = df.drop_duplicates("face").set_index("face")[demo_col]
    face_list = face_demo.index.to_numpy()
    demo_labels = face_demo.to_numpy()

    def interaction_stat(face_demo_map: dict) -> float:
        """Variance of (per-group mean per style) across groups, averaged across styles."""
        stats_list = []
        for style, sdf in df.groupby("style"):
            group_means = {}
            for face, gdf in sdf.groupby("face"):
                grp = face_demo_map.get(face)
                if grp is not None:
                    group_means.setdefault(grp, []).append(gdf["delta"].mean())
            means = [np.mean(v) for v in group_means.values() if v]
            if len(means) >= 2:
                stats_list.append(float(np.var(means)))
        return float(np.mean(stats_list)) if stats_list else float("nan")

    obs_map = face_demo.to_dict()
    obs_stat = interaction_stat(obs_map)
    if math.isnan(obs_stat):
        return {"perm_stat": float("nan"), "p_perm": float("nan")}

    ge_count = 0
    n_valid = 0
    for _ in range(n_perms):
        perm_labels = rng.permutation(demo_labels)
        perm_map = dict(zip(face_list, perm_labels))
        perm_stat = interaction_stat(perm_map)
        if math.isnan(perm_stat):
            continue
        n_valid += 1
        if perm_stat >= obs_stat - 1e-12:
            ge_count += 1

    p_perm = float((ge_count + 1) / (n_valid + 1)) if n_valid > 0 else float("nan")
    return {"perm_stat": round(obs_stat, 6), "p_perm": p_perm}


# ── Table 4: Fashion × Age ────────────────────────────────────────────────────

def run_table4(rows: list[dict]) -> list[dict]:
    print("\n" + "=" * 70)
    print("TABLE 4  –  Fashion style × Age group interaction")
    print("=" * 70)

    demo_vals = ["young adult", "middle-aged adult", "elderly"]
    df = face_level_df(rows, set(TABLE4_STYLES.values()), "age", demo_vals, TABLE4_STYLES)
    print(f"  {df['face'].nunique()} unique faces, {len(df)} face×style observations")

    # 1. Overall interaction F-test (OLS)
    lrt = ols_interaction_ftest(df, groupby_col="demo")
    print(f"\n  OLS interaction F-test (style × age):  "
          f"F({lrt['df_num']},{lrt['df_denom']})={lrt['F_stat']:.3f},  "
          f"p={lrt['p_ols']:.4e},  η²p={lrt['partial_eta2']:.4f}  "
          f"({_sig_marker(lrt['p_ols'])})")
    print(f"  (Note: OLS ignores within-face correlation; use permutation p for primary inference)")

    # 2. Permutation test (primary test)
    perm = permutation_interaction(df, demo_col="demo")
    print(f"  Permutation test ({N_PERMS} perms, primary): stat={perm['perm_stat']:.5f},  "
          f"p={perm['p_perm']:.4f}  ({_sig_marker(perm['p_perm'])})")

    # 3. Per-style pairwise (Young vs Elderly) — most extreme comparison per Table 4
    print(f"\n  Per-style: Young vs Elderly (Mann-Whitney U, face level)")
    print(f"  {'Style':<22} {'n_young':>7} {'n_elderly':>9} "
          f"{'mean_young':>10} {'mean_elderly':>12} {'Cohen d':>8} {'p_raw':>10}")
    print(f"  {'-'*80}")

    style_results = []
    raw_ps = []
    for style_label in TABLE4_STYLES:
        young   = df[(df["style"] == style_label) & (df["demo"] == "young adult")]["delta"].to_numpy()
        elderly = df[(df["style"] == style_label) & (df["demo"] == "elderly")]["delta"].to_numpy()
        if len(young) < 3 or len(elderly) < 3:
            continue
        stat, p = scipy_stats.mannwhitneyu(young, elderly, alternative="two-sided")
        d = _cohens_d(young, elderly)
        style_results.append({
            "style": style_label,
            "n_young": len(young), "n_elderly": len(elderly),
            "mean_young": float(np.mean(young)), "mean_elderly": float(np.mean(elderly)),
            "cohens_d": round(d, 4), "p_raw": float(p),
        })
        raw_ps.append(float(p))

    adj_ps = _bh_correct(raw_ps)
    csv_rows = []
    for r, p_adj in zip(style_results, adj_ps):
        r["p_adj_bh"] = round(p_adj, 6)
        r["sig"] = _sig_marker(p_adj)
        flag = " ←ns" if r["sig"] == "ns" else ""
        print(f"  {r['style']:<22} {r['n_young']:>7} {r['n_elderly']:>9} "
              f"{r['mean_young']:>+10.4f} {r['mean_elderly']:>+12.4f} "
              f"{r['cohens_d']:>+8.3f} {r['p_raw']:>10.3e}{flag}")
        csv_rows.append({
            "table": "Table4", "test": "Young_vs_Elderly",
            "style_or_dimension": r["style"],
            "group_A": "young adult", "group_B": "elderly",
            "n_A": r["n_young"], "n_B": r["n_elderly"],
            "mean_A": round(r["mean_young"], 5), "mean_B": round(r["mean_elderly"], 5),
            "cohens_d": r["cohens_d"],
            "p_raw": round(r["p_raw"], 6), "p_adj_bh": r["p_adj_bh"], "sig": r["sig"],
            "overall_ols_F": lrt["F_stat"],
            "overall_ols_p": round(lrt["p_ols"], 6) if not math.isnan(lrt["p_ols"]) else "nan",
            "partial_eta2": lrt["partial_eta2"],
            "overall_perm_p": round(perm["p_perm"], 4) if not math.isnan(perm["p_perm"]) else "nan",
        })

    # Also add 3-group Kruskal-Wallis per style
    print(f"\n  Per-style: Kruskal-Wallis across all 3 age groups")
    print(f"  {'Style':<22} {'H':>7} {'p_raw':>10}")
    kw_rows = []
    kw_ps = []
    for style_label in TABLE4_STYLES:
        groups = [df[(df["style"] == style_label) & (df["demo"] == d)]["delta"].to_numpy()
                  for d in demo_vals]
        groups = [g for g in groups if len(g) >= 3]
        if len(groups) < 2:
            continue
        h, p = scipy_stats.kruskal(*groups)
        kw_rows.append({"style": style_label, "H": round(h, 4), "p_raw": float(p)})
        kw_ps.append(float(p))
    kw_adj = _bh_correct(kw_ps)
    for r, p_adj in zip(kw_rows, kw_adj):
        sig = _sig_marker(p_adj)
        print(f"  {r['style']:<22} {r['H']:>7.3f} {r['p_raw']:>10.3e}  adj={p_adj:.3e}  {sig}")
        csv_rows.append({
            "table": "Table4", "test": "Kruskal_Wallis_3groups",
            "style_or_dimension": r["style"],
            "group_A": "young adult", "group_B": "middle-aged + elderly",
            "n_A": "", "n_B": "",
            "mean_A": "", "mean_B": "",
            "cohens_d": "",
            "p_raw": round(r["p_raw"], 6), "p_adj_bh": round(p_adj, 6), "sig": sig,
            "overall_ols_p": round(lrt["p_ols"], 6) if not math.isnan(lrt["p_ols"]) else "nan",
            "overall_perm_p": round(perm["p_perm"], 4) if not math.isnan(perm["p_perm"]) else "nan",
        })

    return csv_rows


# ── Table 5: Facial tattoo × Demographic ─────────────────────────────────────

def run_table5(rows: list[dict]) -> list[dict]:
    print("\n" + "=" * 70)
    print("TABLE 5  –  Facial tattoo × Demographic group interaction")
    print("=" * 70)

    comparisons = [
        ("age",  "young adult", "elderly",  "Age: Young vs Elderly"),
        ("gender", "male",      "female",   "Gender: Male vs Female"),
        ("body",  "thin",       "obese",    "Body: Thin vs Obese"),
    ]

    csv_rows = []
    raw_ps = []
    results_buffer = []

    for demo_col, val_a, val_b, label in comparisons:
        # Face-level aggregation for each group
        face_deltas_a: dict[str, list[float]] = defaultdict(list)
        face_deltas_b: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            if row["variation"] != TABLE5_VAR:
                continue
            face = row["face"]
            if not face:
                continue
            demo = row.get(demo_col, "")
            if demo == val_a:
                face_deltas_a[face].append(row["delta"])
            elif demo == val_b:
                face_deltas_b[face].append(row["delta"])

        arr_a = np.array([np.mean(v) for v in face_deltas_a.values() if v])
        arr_b = np.array([np.mean(v) for v in face_deltas_b.values() if v])

        if len(arr_a) < 3 or len(arr_b) < 3:
            continue

        stat, p = scipy_stats.mannwhitneyu(arr_a, arr_b, alternative="two-sided")
        d = _cohens_d(arr_a, arr_b)
        raw_ps.append(float(p))
        results_buffer.append((label, val_a, val_b, arr_a, arr_b, p, d, demo_col))

    adj_ps = _bh_correct(raw_ps)
    print(f"\n  {'Comparison':<28} {'n_A':>5} {'n_B':>5} {'mean_A':>8} {'mean_B':>8} "
          f"{'d':>7} {'p_raw':>10} {'p_adj':>10} {'sig':>5}")
    print(f"  {'-'*90}")

    for (label, val_a, val_b, arr_a, arr_b, p_raw, d, demo_col), p_adj in zip(results_buffer, adj_ps):
        sig = _sig_marker(p_adj)
        flag = " ←ns" if sig == "ns" else ""
        print(f"  {label:<28} {len(arr_a):>5} {len(arr_b):>5} "
              f"{np.mean(arr_a):>+8.4f} {np.mean(arr_b):>+8.4f} "
              f"{d:>+7.3f} {p_raw:>10.3e} {p_adj:>10.3e} {sig:>5}{flag}")
        csv_rows.append({
            "table": "Table5", "test": "MannWhitneyU",
            "style_or_dimension": label,
            "group_A": val_a, "group_B": val_b,
            "n_A": len(arr_a), "n_B": len(arr_b),
            "mean_A": round(float(np.mean(arr_a)), 5),
            "mean_B": round(float(np.mean(arr_b)), 5),
            "cohens_d": round(d, 4),
            "p_raw": round(p_raw, 6), "p_adj_bh": round(p_adj, 6), "sig": sig,
            "overall_ols_p": "", "overall_perm_p": "",
        })

    return csv_rows


# ── Table 6: Fashion × Body type ─────────────────────────────────────────────

def run_table6(rows: list[dict]) -> list[dict]:
    print("\n" + "=" * 70)
    print("TABLE 6  –  Fashion style × Body type interaction")
    print("=" * 70)

    demo_vals = ["thin", "normal", "obese"]
    df = face_level_df(rows, set(TABLE6_STYLES.values()), "body", demo_vals, TABLE6_STYLES)
    print(f"  {df['face'].nunique()} unique faces, {len(df)} face×style observations")

    # 1. Overall interaction F-test (OLS)
    lrt = ols_interaction_ftest(df, groupby_col="demo")
    print(f"\n  OLS interaction F-test (style × body):  "
          f"F({lrt['df_num']},{lrt['df_denom']})={lrt['F_stat']:.3f},  "
          f"p={lrt['p_ols']:.4e},  η²p={lrt['partial_eta2']:.4f}  "
          f"({_sig_marker(lrt['p_ols'])})")
    print(f"  (Note: OLS ignores within-face correlation; use permutation p for primary inference)")

    # 2. Permutation test (primary test)
    perm = permutation_interaction(df, demo_col="demo")
    print(f"  Permutation test ({N_PERMS} perms, primary): stat={perm['perm_stat']:.5f},  "
          f"p={perm['p_perm']:.4f}  ({_sig_marker(perm['p_perm'])})")

    # 3. Per-style: Thin vs Obese (most extreme comparison from Table 6)
    print(f"\n  Per-style: Thin vs Obese (Mann-Whitney U, face level)")
    print(f"  {'Style':<22} {'n_thin':>6} {'n_obese':>7} "
          f"{'mean_thin':>9} {'mean_obese':>10} {'Cohen d':>8} {'p_raw':>10}")
    print(f"  {'-'*80}")

    style_results = []
    raw_ps = []
    for style_label in TABLE6_STYLES:
        thin  = df[(df["style"] == style_label) & (df["demo"] == "thin")]["delta"].to_numpy()
        obese = df[(df["style"] == style_label) & (df["demo"] == "obese")]["delta"].to_numpy()
        if len(thin) < 3 or len(obese) < 3:
            continue
        stat, p = scipy_stats.mannwhitneyu(thin, obese, alternative="two-sided")
        d = _cohens_d(thin, obese)
        style_results.append({
            "style": style_label,
            "n_thin": len(thin), "n_obese": len(obese),
            "mean_thin": float(np.mean(thin)), "mean_obese": float(np.mean(obese)),
            "cohens_d": round(d, 4), "p_raw": float(p),
        })
        raw_ps.append(float(p))

    adj_ps = _bh_correct(raw_ps)
    csv_rows = []
    for r, p_adj in zip(style_results, adj_ps):
        r["p_adj_bh"] = round(p_adj, 6)
        r["sig"] = _sig_marker(p_adj)
        flag = " ←ns" if r["sig"] == "ns" else ""
        print(f"  {r['style']:<22} {r['n_thin']:>6} {r['n_obese']:>7} "
              f"{r['mean_thin']:>+9.4f} {r['mean_obese']:>+10.4f} "
              f"{r['cohens_d']:>+8.3f} {r['p_raw']:>10.3e}{flag}")
        csv_rows.append({
            "table": "Table6", "test": "Thin_vs_Obese",
            "style_or_dimension": r["style"],
            "group_A": "thin", "group_B": "obese",
            "n_A": r["n_thin"], "n_B": r["n_obese"],
            "mean_A": round(r["mean_thin"], 5), "mean_B": round(r["mean_obese"], 5),
            "cohens_d": r["cohens_d"],
            "p_raw": round(r["p_raw"], 6), "p_adj_bh": r["p_adj_bh"], "sig": r["sig"],
            "overall_ols_F": lrt["F_stat"],
            "overall_ols_p": round(lrt["p_ols"], 6) if not math.isnan(lrt["p_ols"]) else "nan",
            "partial_eta2": lrt["partial_eta2"],
            "overall_perm_p": round(perm["p_perm"], 4) if not math.isnan(perm["p_perm"]) else "nan",
        })

    # 3-group Kruskal-Wallis per style
    print(f"\n  Per-style: Kruskal-Wallis across Thin / Normal / Obese")
    print(f"  {'Style':<22} {'H':>7} {'p_raw':>10}")
    kw_rows = []
    kw_ps = []
    for style_label in TABLE6_STYLES:
        groups = [df[(df["style"] == style_label) & (df["demo"] == d)]["delta"].to_numpy()
                  for d in demo_vals]
        groups = [g for g in groups if len(g) >= 3]
        if len(groups) < 2:
            continue
        h, p = scipy_stats.kruskal(*groups)
        kw_rows.append({"style": style_label, "H": round(h, 4), "p_raw": float(p)})
        kw_ps.append(float(p))
    kw_adj = _bh_correct(kw_ps)
    for r, p_adj in zip(kw_rows, kw_adj):
        sig = _sig_marker(p_adj)
        print(f"  {r['style']:<22} {r['H']:>7.3f} {r['p_raw']:>10.3e}  adj={p_adj:.3e}  {sig}")
        csv_rows.append({
            "table": "Table6", "test": "Kruskal_Wallis_3groups",
            "style_or_dimension": r["style"],
            "group_A": "thin", "group_B": "normal + obese",
            "n_A": "", "n_B": "",
            "mean_A": "", "mean_B": "",
            "cohens_d": "",
            "p_raw": round(r["p_raw"], 6), "p_adj_bh": round(p_adj, 6), "sig": sig,
            "overall_ols_p": round(lrt["p_ols"], 6) if not math.isnan(lrt["p_ols"]) else "nan",
            "overall_perm_p": round(perm["p_perm"], 4) if not math.isnan(perm["p_perm"]) else "nan",
        })

    return csv_rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rows = load_rows()

    all_rows = []
    all_rows += run_table4(rows)
    all_rows += run_table5(rows)
    all_rows += run_table6(rows)

    out_path = OUT_DIR / "interaction_tests.csv"
    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
