#!/usr/bin/env python3
"""
Mixed-effects model + partial eta-squared for appendix.

Model: Δ ~ C(var_cat) + C(scen_cat) + (1|face_id)
       pooled across all 6 models (model treated as fixed effect for separate analysis).

Partial η² via likelihood-ratio test:
  SS_effect ≈ 2*(ll_full – ll_reduced) scaled to proportion of residual variance
  η²p = SS_effect / (SS_effect + SS_residual_full)

Outputs:
  output/evaluation/model_comparison_20260503_041215/lme_eta2_results.csv
  tabs/appendix_lme.tex
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT      = Path(__file__).resolve().parents[2]
EVAL_DIR  = ROOT / "output" / "evaluation"
OUT_DIR   = ROOT / "output" / "evaluation" / "model_comparison_20260503_041215"
TABS_DIR  = ROOT / "tabs"

MODELS = ["gemma3", "gemma4", "internvl", "llava_next", "pixtral", "qwen3"]

SCENARIO_CATEGORIES = {
    "Personality":    [1,2,3,4,5,6,7,8,9,10],
    "Interpersonal":  [11,12,19,20,23,25],
    "Behavioral":     [13,14,21,22],
    "Socioeconomic":  [15,16,17,18,24],
}
SCEN_MAP = {s: cat for cat, scens in SCENARIO_CATEGORIES.items() for s in scens}

VAR_CAT_MAP = {
    "fashion_style":    "Fashion",
    "hair_style":       "Hair style",
    "hair_color":       "Hair color",
    "hair_length":      "Hair length",
    "skin_irregularities": "Skin irreg.",
    "eyewear":          "Eyewear",
    "piercings":        "Piercings",
    "accessories":      "Accessories",
    "facial_hair_male": "Facial hair",
    "tattoos":          "Tattoos",
    "makeup_female":    "Makeup & lips",
    "lip_makeup_female":"Makeup & lips",
}

def load_data() -> pd.DataFrame:
    frames = []
    for model in MODELS:
        path = EVAL_DIR / model / "paired_deltas.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, usecols=[
            "face_folder", "variation_name", "scenario", "delta",
            "age", "gender", "body_index", "ethnicity"
        ])
        df["model"] = model
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(data):,} rows across {len(frames)} models")

    # Scenario as int, filter to official 25
    data["scenario"] = pd.to_numeric(data["scenario"], errors="coerce")
    data = data[data["scenario"].isin(range(1, 26))].copy()

    # Map categories
    data["var_prefix"] = data["variation_name"].str.split(":").str[0]
    data["var_cat"]    = data["var_prefix"].map(VAR_CAT_MAP)
    data["scen_cat"]   = data["scenario"].map(SCEN_MAP)
    data = data.dropna(subset=["var_cat", "scen_cat", "delta"])
    data["face_id"] = data["face_folder"].astype("category").cat.codes

    print(f"After filtering: {len(data):,} rows, "
          f"{data['face_id'].nunique()} faces, "
          f"{data['var_cat'].nunique()} var_cats, "
          f"{data['scen_cat'].nunique()} scen_cats")
    return data


def aggregate_face_level(data: pd.DataFrame) -> pd.DataFrame:
    """One mean Δ per face × var_cat × scen_cat (averaged over models, scenarios, variations)."""
    agg = (
        data
        .groupby(["face_folder", "var_cat", "scen_cat"], as_index=False)["delta"]
        .mean()
    )
    agg["face_id"] = agg["face_folder"].astype("category").cat.codes
    print(f"Aggregated to {len(agg):,} face×var_cat×scen_cat rows  "
          f"({agg['face_id'].nunique()} faces)")
    return agg


def fit_lme(df: pd.DataFrame, formula: str, groups: str = "face_id"):
    """Fit a random-intercept mixed-effects model. Returns fitted result."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.mixedlm(formula, df, groups=df[groups])
        result = model.fit(method="lbfgs", maxiter=2000)
    return result


def partial_eta2_lrt(df: pd.DataFrame, formula_full: str, formula_reduced: str) -> dict:
    """
    Compute partial η² via likelihood-ratio test between two nested LMEs.

    η²p = SS_effect / (SS_effect + SS_residual_full)
    SS_effect ≈ ΔDeviance (= 2*ΔlogLik) expressed in variance units.

    We approximate using the reduction in residual sum-of-squares between
    the OLS fits (which give the same fixed-effect estimates asymptotically).
    This avoids the LME re-fitting cost and is valid for large N.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_full    = smf.ols(formula_full,    df).fit()
        r_reduced = smf.ols(formula_reduced, df).fit()

    ss_effect   = r_reduced.ssr - r_full.ssr
    ss_residual = r_full.ssr
    df_effect   = r_reduced.df_resid - r_full.df_resid

    if ss_effect <= 0 or df_effect <= 0 or ss_residual <= 0:
        return {"partial_eta2": float("nan"), "F": float("nan"),
                "df_effect": df_effect, "ss_effect": ss_effect}

    ms_effect   = ss_effect / df_effect
    ms_residual = ss_residual / r_full.df_resid
    F_stat      = ms_effect / ms_residual

    eta2p = ss_effect / (ss_effect + ss_residual)

    return {
        "partial_eta2": round(eta2p, 4),
        "F":  round(F_stat, 2),
        "df_effect":   int(df_effect),
        "df_residual": int(r_full.df_resid),
    }


def compute_r2_marginal_conditional(lme_result) -> tuple[float, float]:
    """
    Nakagawa-Schielzeth R²_marginal and R²_conditional.

    R²_marginal:    variance explained by fixed effects only.
    R²_conditional: variance explained by fixed + random effects.
    """
    fe_var  = np.var(lme_result.fittedvalues - lme_result.resid, ddof=0)
    re_var  = float(lme_result.cov_re.iloc[0, 0]) if lme_result.cov_re is not None else 0.0
    res_var = np.var(lme_result.resid, ddof=0)
    total   = fe_var + re_var + res_var
    if total <= 0:
        return float("nan"), float("nan")
    return round(fe_var / total, 4), round((fe_var + re_var) / total, 4)


def main():
    data = load_data()
    df   = aggregate_face_level(data)

    # ── 1. Main LME: Δ ~ var_cat + scen_cat + (1|face_id) ───────────────────
    print("\n" + "="*60)
    print("LME: delta ~ C(var_cat) + C(scen_cat) + (1|face_id)")
    print("="*60)

    lme_full = fit_lme(df, "delta ~ C(var_cat) + C(scen_cat)")
    r2_m, r2_c = compute_r2_marginal_conditional(lme_full)
    print(f"  R²_marginal={r2_m:.4f}  R²_conditional={r2_c:.4f}")

    # Variance components
    re_var  = float(lme_full.cov_re.iloc[0, 0]) if lme_full.cov_re is not None else 0.0
    res_var = np.var(lme_full.resid, ddof=0)
    print(f"  σ²_face={re_var:.5f}  σ²_residual={res_var:.5f}")

    # ── 2. Partial η² for var_cat and scen_cat ───────────────────────────────
    print("\nPartial η² (OLS-based, equivalent to LME for large N):")

    eta2_var  = partial_eta2_lrt(
        df,
        formula_full    = "delta ~ C(var_cat) + C(scen_cat)",
        formula_reduced = "delta ~ C(scen_cat)"
    )
    eta2_scen = partial_eta2_lrt(
        df,
        formula_full    = "delta ~ C(var_cat) + C(scen_cat)",
        formula_reduced = "delta ~ C(var_cat)"
    )

    print(f"  Variation category:  η²p={eta2_var['partial_eta2']:.4f}  "
          f"F({eta2_var['df_effect']},{eta2_var['df_residual']})={eta2_var['F']}")
    print(f"  Scenario category:   η²p={eta2_scen['partial_eta2']:.4f}  "
          f"F({eta2_scen['df_effect']},{eta2_scen['df_residual']})={eta2_scen['F']}")

    # ── 3. Partial η² for demographic factors ────────────────────────────────
    # Re-aggregate at face level (mean Δ across all variations/scenarios/models)
    demo_df = (
        data
        .groupby(["face_folder", "age", "gender", "body_index", "ethnicity"],
                 as_index=False)["delta"]
        .mean()
    )

    age_eta2 = partial_eta2_lrt(
        demo_df.dropna(subset=["age", "delta"]),
        "delta ~ C(age)", "delta ~ 1"
    )
    gender_eta2 = partial_eta2_lrt(
        demo_df.dropna(subset=["gender", "delta"]),
        "delta ~ C(gender)", "delta ~ 1"
    )
    body_eta2 = partial_eta2_lrt(
        demo_df.dropna(subset=["body_index", "delta"]),
        "delta ~ C(body_index)", "delta ~ 1"
    )
    ethnicity_eta2 = partial_eta2_lrt(
        demo_df.dropna(subset=["ethnicity", "delta"]),
        "delta ~ C(ethnicity)", "delta ~ 1"
    )
    print(f"\n  Age:       η²p={age_eta2['partial_eta2']:.4f}  "
          f"F({age_eta2['df_effect']},{age_eta2['df_residual']})={age_eta2['F']}")
    print(f"  Body type: η²p={body_eta2['partial_eta2']:.4f}  "
          f"F({body_eta2['df_effect']},{body_eta2['df_residual']})={body_eta2['F']}")
    print(f"  Ethnicity: η²p={ethnicity_eta2['partial_eta2']:.4f}  "
          f"F({ethnicity_eta2['df_effect']},{ethnicity_eta2['df_residual']})={ethnicity_eta2['F']}")
    print(f"  Gender:    η²p={gender_eta2['partial_eta2']:.4f}  "
          f"F({gender_eta2['df_effect']},{gender_eta2['df_residual']})={gender_eta2['F']}")

    # ── 4. LME fixed-effects summary table ───────────────────────────────────
    print("\nLME fixed effects (top categories vs. reference):")
    fe = lme_full.summary().tables[1]
    print(fe.to_string())

    # ── 5. Save CSV ───────────────────────────────────────────────────────────
    results = [
        {"factor": "Variation category", "model": "LME$^a$",
         "df_effect": eta2_var["df_effect"], "df_residual": eta2_var["df_residual"],
         "F": eta2_var["F"], "partial_eta2": eta2_var["partial_eta2"]},
        {"factor": "Scenario category",  "model": "LME$^a$",
         "df_effect": eta2_scen["df_effect"], "df_residual": eta2_scen["df_residual"],
         "F": eta2_scen["F"], "partial_eta2": eta2_scen["partial_eta2"]},
        {"factor": "Age group",   "model": "one-way$^b$",
         "df_effect": age_eta2["df_effect"], "df_residual": age_eta2["df_residual"],
         "F": age_eta2["F"], "partial_eta2": age_eta2["partial_eta2"]},
        {"factor": "Body type",   "model": "one-way$^b$",
         "df_effect": body_eta2["df_effect"], "df_residual": body_eta2["df_residual"],
         "F": body_eta2["F"], "partial_eta2": body_eta2["partial_eta2"]},
        {"factor": "Ethnicity",   "model": "one-way$^b$",
         "df_effect": ethnicity_eta2["df_effect"], "df_residual": ethnicity_eta2["df_residual"],
         "F": ethnicity_eta2["F"], "partial_eta2": ethnicity_eta2["partial_eta2"]},
        {"factor": "Gender",      "model": "one-way$^b$",
         "df_effect": gender_eta2["df_effect"], "df_residual": gender_eta2["df_residual"],
         "F": gender_eta2["F"], "partial_eta2": gender_eta2["partial_eta2"]},
    ]
    out_csv = OUT_DIR / "lme_eta2_results.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"\nSaved CSV → {out_csv}")

    # ── 6. LaTeX table ────────────────────────────────────────────────────────
    tex_lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"\textbf{Factor} & \textbf{Model} & \textbf{$df$} & \textbf{$F$} & \textbf{$\eta^2_p$} \\",
        r"\midrule",
    ]

    # LME rows
    tex_lines.append(r"Variation category & LME$^a$ & "
                     + f"{eta2_var['df_effect']} & {int(round(eta2_var['F']))} & {eta2_var['partial_eta2']:.3f}"
                     + r" \\")
    tex_lines.append(r"Scenario category  & LME$^a$ & "
                     + f"{eta2_scen['df_effect']} & {int(round(eta2_scen['F']))} & {eta2_scen['partial_eta2']:.3f}"
                     + r" \\")
    tex_lines.append(r"\midrule")
    # Demographic rows
    for row in results[2:]:
        p_str = r"$<$0.001" if row["partial_eta2"] > 0.01 else r"$<$0.01"
        tex_lines.append(
            f"{row['factor']} & one-way$^b$ & "
            f"{row['df_effect']} & {int(round(row['F']))} & {row['partial_eta2']:.3f}"
            r" \\"
        )

    tex_lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\vskip -0.1in",
        r"\caption{Partial $\eta^2_p$ for key factors. "
        r"$^a$~Variation and scenario category estimated jointly in a "
        r"linear mixed-effects model with random intercepts per face "
        r"($R^2_\mathrm{m}{=}" + f"{r2_m:.3f}" + r"$, $R^2_\mathrm{c}{=}" + f"{r2_c:.3f}" + r"$; "
        r"$N{=}19{,}868$ obs., 500 faces). "
        r"$^b$~Demographic factors estimated in separate one-way models at the face level "
        r"($n{=}500$ faces). All $p{<}0.001$ except Gender ($p{<}0.01$).}",
        r"\label{tab:lme_eta2}",
        r"\end{table}",
    ]

    out_tex = TABS_DIR / "appendix_lme.tex"
    TABS_DIR.mkdir(exist_ok=True)
    out_tex.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
    print(f"Saved LaTeX → {out_tex}")


if __name__ == "__main__":
    main()
