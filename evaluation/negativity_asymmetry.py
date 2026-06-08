#!/usr/bin/env python3
"""Formal test of negativity asymmetry: |Δ_negative| > |Δ_positive|.

For each pre-defined positive/negative variant pair (same attribute domain),
we compute face-level mean |Δ| for both variants and test whether the
negative-cue magnitude exceeds the positive-cue magnitude.

Test unit : base face (n ≈ 130–334 per pair, all models pooled)
Test       : Wilcoxon signed-rank on |Δ_neg| − |Δ_pos| > 0 per face
             (one-sided, BH-corrected across pairs)
Ratio      : median |Δ_neg| / median |Δ_pos|  (reported per pair and per model)

Pairs defined on domain knowledge (not data-driven):
  Within fashion_style:
    1. Formal / Evening wear  (+) vs Worn / Distressed clothing (-)
    2. Professional / Business formal (+) vs Worn / Distressed clothing (-)
    3. Smart casual (+) vs Streetwear (-)
  Within hair_style:
    4. Slicked back (+) vs Messy (-)
  Cross-domain appearance cues (female faces):
    5. makeup_female:Heavy (+) vs skin_irregularities:Acne (-)

Outputs:
    output/evaluation/eval_charts/negativity_asymmetry_pairs.csv
    output/evaluation/eval_charts/negativity_asymmetry_per_model.csv
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

ROOT    = Path(__file__).resolve().parents[2]
EVAL    = ROOT / "output" / "evaluation"
OUT_DIR = ROOT / "output" / "evaluation" / "eval_charts"
MODELS  = ["gemma3", "gemma4", "internvl", "llava_next", "pixtral", "qwen3"]
MODEL_DISPLAY = {
    "gemma3": "Gemma-3", "gemma4": "Gemma-4", "internvl": "InternVL3",
    "llava_next": "LLaVA-v1.6", "pixtral": "Pixtral", "qwen3": "Qwen3-VL",
}
OFFICIAL_SCENARIOS = set(range(1, 26))

# ---------------------------------------------------------------------------
# Pairs: (label, positive_variation, negative_variation, face_filter)
# face_filter: None = all faces, "female" = female only, "male" = male only
# ---------------------------------------------------------------------------
PAIRS = [
    ("Formal vs Worn",
     "fashion_style:Formal / Evening wear",
     "fashion_style:Worn / Distressed clothing",
     None),
    ("Business vs Worn",
     "fashion_style:Professional / Business formal",
     "fashion_style:Worn / Distressed clothing",
     None),
    ("Smart Casual vs Streetwear",
     "fashion_style:Smart casual",
     "fashion_style:Streetwear",
     None),
    ("Hair: Slicked vs Messy",
     "hair_style:Slicked back",
     "hair_style:Messy",
     None),
    ("Makeup vs Acne (female)",
     "makeup_female:Heavy",
     "skin_irregularities:Acne",
     "female"),
]


def bh_correct(pvals: list[float]) -> list[float]:
    n = len(pvals)
    if n == 0:
        return []
    indexed = sorted(enumerate(pvals), key=lambda x: x[1])
    adj = [1.0] * n
    prev = 1.0
    for rank, (i, p) in enumerate(reversed(indexed), 1):
        a = min(p * n / (n - rank + 1), prev)
        prev = a
        adj[i] = a
    return adj


def load_rows() -> list[dict]:
    rows = []
    for model in MODELS:
        path = EVAL / model / "paired_deltas.csv"
        if not path.exists():
            continue
        with open(path) as f:
            for row in csv.DictReader(f):
                try:
                    scen = int(row["scenario"])
                    float(row["delta"])
                except (ValueError, TypeError):
                    continue
                if scen not in OFFICIAL_SCENARIOS:
                    continue
                row["model"] = model
                rows.append(row)
    print(f"Loaded {len(rows):,} rows across {len(MODELS)} models")
    return rows


def face_means_for_variant(
    rows: list[dict],
    variation: str,
    face_filter: str | None,
) -> dict[str, float]:
    """Return {face_folder: mean_delta} for the given variation."""
    face_deltas: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["variation_name"] != variation:
            continue
        if face_filter == "female" and row.get("gender") != "female":
            continue
        if face_filter == "male" and row.get("gender") != "male":
            continue
        face = (row.get("face_folder") or "").strip()
        if not face:
            continue
        try:
            face_deltas[face].append(float(row["delta"]))
        except (ValueError, TypeError):
            continue
    return {f: float(np.mean(v)) for f, v in face_deltas.items() if v}


def face_means_for_variant_per_model(
    rows: list[dict],
    variation: str,
    face_filter: str | None,
    model: str,
) -> dict[str, float]:
    """Return {face_folder: mean_delta} for given variation and model."""
    face_deltas: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["model"] != model:
            continue
        if row["variation_name"] != variation:
            continue
        if face_filter == "female" and row.get("gender") != "female":
            continue
        if face_filter == "male" and row.get("gender") != "male":
            continue
        face = (row.get("face_folder") or "").strip()
        if not face:
            continue
        try:
            face_deltas[face].append(float(row["delta"]))
        except (ValueError, TypeError):
            continue
    return {f: float(np.mean(v)) for f, v in face_deltas.items() if v}


def paired_test(
    pos_means: dict[str, float],
    neg_means: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (abs_pos, abs_neg, diff) arrays for faces present in both."""
    common = sorted(set(pos_means) & set(neg_means))
    abs_pos = np.array([abs(pos_means[f]) for f in common])
    abs_neg = np.array([abs(neg_means[f]) for f in common])
    return abs_pos, abs_neg, abs_neg - abs_pos


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()

    # ── Pooled across all models ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("NEGATIVITY ASYMMETRY  –  pooled face-level test (all models)")
    print("=" * 70)

    pair_results = []
    for label, pos_var, neg_var, filt in PAIRS:
        pos_m = face_means_for_variant(rows, pos_var, filt)
        neg_m = face_means_for_variant(rows, neg_var, filt)
        abs_pos, abs_neg, diff = paired_test(pos_m, neg_m)
        n = len(diff)
        if n < 3:
            p_raw = float("nan")
        else:
            _, p_raw = scipy_stats.wilcoxon(
                diff, zero_method="wilcox", alternative="greater"
            )
            p_raw = float(p_raw)
        med_pos  = float(np.median(abs_pos))
        med_neg  = float(np.median(abs_neg))
        ratio    = med_neg / med_pos if med_pos > 0 else float("nan")
        pair_results.append((label, pos_var, neg_var, n, med_pos, med_neg, ratio, p_raw))

    raw_ps = [r[7] if not math.isnan(r[7]) else 1.0 for r in pair_results]
    adj_ps = bh_correct(raw_ps)

    csv_rows_pairs = []
    print(f"\n{'Pair':<28} {'n':>5} {'med|Δ+|':>9} {'med|Δ-|':>9} "
          f"{'ratio':>7} {'p_raw':>10} {'p_adj':>10} {'sig':>5}")
    print("-" * 90)
    for (label, pv, nv, n, mp, mn, ratio, p_raw), p_adj in zip(pair_results, adj_ps):
        sig = ("***" if p_adj < 0.001 else "**" if p_adj < 0.01
               else "*" if p_adj < 0.05 else "ns")
        print(f"{label:<28} {n:>5} {mp:>9.4f} {mn:>9.4f} {ratio:>7.3f} "
              f"{p_raw:>10.2e} {p_adj:>10.2e} {sig:>5}")
        csv_rows_pairs.append({
            "pair": label,
            "positive_variant": pv,
            "negative_variant": nv,
            "n_faces": n,
            "median_abs_pos": round(mp, 5),
            "median_abs_neg": round(mn, 5),
            "ratio_neg_pos": round(ratio, 4),
            "p_raw_wilcoxon_greater": f"{p_raw:.4e}",
            "p_adj_bh": f"{p_adj:.4e}",
            "sig": sig,
        })

    out_pairs = OUT_DIR / "negativity_asymmetry_pairs.csv"
    with open(out_pairs, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows_pairs[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows_pairs)
    print(f"\nSaved → {out_pairs}")

    # ── Per-model ratios ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PER-MODEL RATIOS  –  median |Δ_neg| / median |Δ_pos|")
    print("=" * 70)

    header = ["pair"] + [MODEL_DISPLAY[m] for m in MODELS] + ["wilcoxon_p", "sig"]
    csv_rows_model = []

    for label, pos_var, neg_var, filt in PAIRS:
        model_ratios = []
        for model in MODELS:
            pm = face_means_for_variant_per_model(rows, pos_var, filt, model)
            nm = face_means_for_variant_per_model(rows, neg_var, filt, model)
            abs_p, abs_n, _ = paired_test(pm, nm)
            mp = float(np.median(abs_p)) if len(abs_p) else float("nan")
            mn = float(np.median(abs_n)) if len(abs_n) else float("nan")
            r = mn / mp if mp > 0 else float("nan")
            model_ratios.append(r)

        valid = [r for r in model_ratios if not math.isnan(r)]
        if len(valid) >= 3:
            log_r = [math.log(r) for r in valid if r > 0]
            if len(log_r) >= 3:
                _, p = scipy_stats.wilcoxon(
                    [x - 0 for x in log_r],
                    zero_method="wilcox", alternative="greater"
                )
                p = float(p)
            else:
                p = float("nan")
        else:
            p = float("nan")
        sig = ("***" if p < 0.001 else "**" if p < 0.01
               else "*" if p < 0.05 else "ns")

        ratio_strs = [f"{r:.3f}" if not math.isnan(r) else "—" for r in model_ratios]
        print(f"\n{label}  (Wilcoxon p={p:.3f}  {sig})")
        for m, rs in zip(MODELS, ratio_strs):
            print(f"  {MODEL_DISPLAY[m]:<14}: {rs}")

        row = {"pair": label}
        for m, r in zip(MODELS, model_ratios):
            row[MODEL_DISPLAY[m]] = round(r, 3) if not math.isnan(r) else "—"
        row["wilcoxon_p"] = f"{p:.4e}" if not math.isnan(p) else "—"
        row["sig"] = sig
        csv_rows_model.append(row)

    out_model = OUT_DIR / "negativity_asymmetry_per_model.csv"
    with open(out_model, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(csv_rows_model)
    print(f"\nSaved → {out_model}")


if __name__ == "__main__":
    main()
