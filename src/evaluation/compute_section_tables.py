#!/usr/bin/env python3
"""Compute values for the four tables in sec:results_finegrained.

- tab:overview   – SBS per category × demographic dimension
- tab:age_gradient – SBS per fashion style × age group
- tab:tattoo_flip  – Facial tattoo SBS per selected demographic subgroup
- tab:body_compensation – SBS per fashion style × body type

All values use model-averaged per-face means; significance via face-level
Wilcoxon signed-rank with BH FDR correction per table.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.stats import wilcoxon

EVALUATION_ROOT = Path("output/evaluation")
MODELS = ["gemma3", "gemma4", "internvl", "llava_next", "pixtral", "qwen3"]

FEMALE_ONLY = {"makeup_female", "lip_makeup_female"}
MALE_ONLY   = {"facial_hair_male"}

CATEGORY_MAP: dict[str, str] = {
    "fashion_style":        "Fashion",
    "facial_hair_male":     "Facial hair",
    "eyewear":              "Eyewear",
    "makeup_female":        "Makeup & lips",
    "lip_makeup_female":    "Makeup & lips",
    "tattoos":              "Tattoos",
    "hair_style":           "Hair style",
    "skin_irregularities":  "Skin irreg.",
    "hair_color":           "Hair len./color",
    "hair_length":          "Hair len./color",
    "accessories":          "Accessories",
    "piercings":            "Piercings",
}

CATEGORY_ORDER = [
    "Fashion", "Facial hair", "Eyewear", "Makeup & lips",
    "Tattoos", "Hair style", "Skin irreg.", "Hair len./color",
    "Accessories", "Piercings",
]

DIM_GROUPS: dict[str, list[str]] = {
    "Age":    ["young adult", "middle-aged adult", "elderly"],
    "Gender": ["male", "female"],
    "Ethn.":  ["Asian", "African", "European", "Middle Eastern", "Latino"],
    "Body":   ["thin", "normal", "obese"],
}

DIM_COL: dict[str, str] = {
    "Age":    "age",
    "Gender": "gender",
    "Ethn.":  "ethnicity",
    "Body":   "body_index",
}


def load_all_deltas() -> list[dict]:
    rows = []
    for model in MODELS:
        path = EVALUATION_ROOT / model / "paired_deltas.csv"
        if not path.exists():
            print(f"  Missing: {path}")
            continue
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cat_key = row["variation_name"].split(":", 1)[0].strip().lower()
                gender = row["gender"]
                # skip gender-specific variations for wrong gender
                if cat_key in FEMALE_ONLY and gender != "female":
                    continue
                if cat_key in MALE_ONLY and gender != "male":
                    continue
                row["cat_key"] = cat_key
                row["category"] = CATEGORY_MAP.get(cat_key, "Other")
                row["delta"] = float(row["delta"])
                row["model"] = model
                rows.append(row)
    return rows


def face_means(rows: list[dict],
               filter_fn=None,
               group_key_fn=None) -> dict:
    """
    Compute per-face mean delta for each group defined by group_key_fn.
    Returns dict: group_key → array of per-face means (one value per face×model).

    If group_key_fn is None, returns a single group under key 'all'.
    filter_fn: optional function(row) -> bool to pre-filter rows.
    """
    if filter_fn:
        rows = [r for r in rows if filter_fn(r)]

    # acc[group][model+face] = list of deltas
    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for row in rows:
        key = group_key_fn(row) if group_key_fn else "all"
        if key is None:
            continue
        face_id = row["model"] + "||" + row["face_folder"]
        acc[key][face_id].append(row["delta"])

    return {
        k: np.array([np.mean(v) for v in face_dict.values()])
        for k, face_dict in acc.items()
    }


def bh_correct(pvals: list[float]) -> list[float]:
    n = len(pvals)
    if n == 0:
        return []
    order = np.argsort(pvals)
    corrected = np.ones(n)
    for rank, idx in enumerate(order):
        corrected[idx] = pvals[idx] * n / (rank + 1)
    # Enforce monotonicity (BH): corrected[i] = min(corrected[i:]) going from largest rank
    for i in range(n - 2, -1, -1):
        corrected[order[i]] = min(corrected[order[i]], corrected[order[i + 1]])
    return corrected.clip(0, 1).tolist()


def wilcoxon_p(arr: np.ndarray) -> float:
    arr = arr[np.isfinite(arr)]
    if len(arr) < 5 or np.all(arr == 0):
        return 1.0
    try:
        _, p = wilcoxon(arr, alternative="two-sided")
        return float(p)
    except Exception:
        return 1.0


def compute_overview(rows: list[dict]):
    print("\n=== tab:overview ===")
    print(f"{'Category':<20} {'Age':>8} {'Gender':>8} {'Ethn.':>8} {'Body':>8}")

    all_cells: list[tuple] = []  # (cat, dim, mean, arr)

    for cat in CATEGORY_ORDER:
        cat_rows = [r for r in rows if r["category"] == cat]
        row_cells = {}
        for dim, groups in DIM_GROUPS.items():
            col = DIM_COL[dim]
            # per-subgroup per-face means, then average across subgroups
            subgroup_means = []
            for grp in groups:
                grp_rows = [r for r in cat_rows if r[col] == grp]
                if not grp_rows:
                    continue
                per_face: dict[str, list[float]] = defaultdict(list)
                for r in grp_rows:
                    fid = r["model"] + "||" + r["face_folder"]
                    per_face[fid].append(r["delta"])
                face_arr = np.array([np.mean(v) for v in per_face.values()])
                subgroup_means.append(np.mean(face_arr))
            dim_mean = np.mean(subgroup_means) if subgroup_means else float("nan")
            row_cells[dim] = dim_mean

        vals = [row_cells.get(d, float("nan")) for d in ["Age", "Gender", "Ethn.", "Body"]]
        print(f"{cat:<20} {vals[0]:>+8.3f} {vals[1]:>+8.3f} {vals[2]:>+8.3f} {vals[3]:>+8.3f}")

    # Overall average row
    print()
    for dim in ["Age", "Gender", "Ethn.", "Body"]:
        col = DIM_COL[dim]
        all_means = []
        for grp in DIM_GROUPS[dim]:
            grp_rows = [r for r in rows if r[col] == grp]
            per_face: dict[str, list[float]] = defaultdict(list)
            for r in grp_rows:
                fid = r["model"] + "||" + r["face_folder"]
                per_face[fid].append(r["delta"])
            face_arr = np.array([np.mean(v) for v in per_face.values()])
            all_means.append(np.mean(face_arr))
        print(f"  Average {dim}: {np.mean(all_means):+.3f}")

    # Now compute significance (Wilcoxon per cell, BH across all cells)
    print("\nSignificance (BH across 40 cells):")
    cells_info: list[tuple] = []
    for cat in CATEGORY_ORDER:
        cat_rows = [r for r in rows if r["category"] == cat]
        for dim, groups in DIM_GROUPS.items():
            col = DIM_COL[dim]
            # Pool all per-face means across subgroups for significance test
            all_face_means: dict[str, list[float]] = defaultdict(list)
            for grp in groups:
                grp_rows = [r for r in cat_rows if r[col] == grp]
                for r in grp_rows:
                    fid = r["model"] + "||" + r["face_folder"]
                    all_face_means[fid].append(r["delta"])
            face_arr = np.array([np.mean(v) for v in all_face_means.values()])
            mean_val = np.mean(face_arr)
            p_raw = wilcoxon_p(face_arr)
            cells_info.append((cat, dim, mean_val, p_raw))

    p_raws = [c[3] for c in cells_info]
    p_bh = bh_correct(p_raws)

    print(f"\n{'Category':<20} {'Dim':<8} {'Mean':>8} {'p_raw':>10} {'p_BH':>10} sig")
    for i, (cat, dim, mean_val, p_raw) in enumerate(cells_info):
        pb = p_bh[i]
        sig = "***" if pb < 0.001 else ("*" if pb < 0.05 else "ns")
        print(f"{cat:<20} {dim:<8} {mean_val:>+8.3f} {p_raw:>10.4f} {pb:>10.4f} {sig}")


def compute_age_gradient(rows: list[dict]):
    print("\n=== tab:age_gradient ===")

    fashion_rows = [r for r in rows if r["cat_key"] == "fashion_style"]

    # Get unique fashion styles (sorted)
    styles_of_interest = [
        "Professional / Business formal",
        "Formal / Evening wear",
        "Smart casual",
        "Vintage / Retro",
        "Casual",
        "Streetwear",
        "Functional / outdoor wear",
        "Sporty / Athletic wear",
        "Worn / Distressed clothing",
    ]

    age_groups = ["young adult", "middle-aged adult", "elderly"]

    cells_info = []
    print(f"\n{'Style':<30} {'Young':>8} {'Middle':>8} {'Elderly':>8} {'Y-E':>8}")

    for style in styles_of_interest:
        style_rows = [r for r in fashion_rows
                      if r["variation_name"].split(":", 1)[1].strip().lower() == style.lower()]
        if not style_rows:
            # try partial match
            style_rows = [r for r in fashion_rows
                          if style.lower() in r["variation_name"].lower()]
        if not style_rows:
            print(f"  WARNING: No rows for style '{style}'")
            continue

        age_means = {}
        for ag in age_groups:
            ag_rows = [r for r in style_rows if r["age"] == ag]
            per_face: dict[str, list[float]] = defaultdict(list)
            for r in ag_rows:
                fid = r["model"] + "||" + r["face_folder"]
                per_face[fid].append(r["delta"])
            face_arr = np.array([np.mean(v) for v in per_face.values()])
            age_means[ag] = np.mean(face_arr)
            cells_info.append((style, ag, face_arr))

        ya = age_means.get("young adult", float("nan"))
        ma = age_means.get("middle adult", float("nan"))
        el = age_means.get("elderly", float("nan"))
        gap = el - ya if np.isfinite(el) and np.isfinite(ya) else float("nan")
        print(f"{style:<30} {ya:>+8.3f} {ma:>+8.3f} {el:>+8.3f} {gap:>+8.3f}")

    # Significance
    print("\nSignificance per cell:")
    p_raws = [wilcoxon_p(arr) for _, _, arr in cells_info]
    p_bh = bh_correct(p_raws)
    for i, (style, ag, arr) in enumerate(cells_info):
        pb = p_bh[i]
        sig = "***" if pb < 0.001 else ("*" if pb < 0.05 else "ns")
        print(f"  {style:<35} {ag:<15} mean={np.mean(arr):>+.3f} p_BH={pb:.4f} {sig}")


def compute_tattoo_flip(rows: list[dict]):
    print("\n=== tab:tattoo_flip ===")

    tattoo_rows = [r for r in rows if r["cat_key"] == "tattoos"]
    print(f"  Total tattoo rows: {len(tattoo_rows)}")

    groups_of_interest = [
        ("Age",    "young adult",      "age"),
        ("Age",    "middle-aged adult","age"),
        ("Age",    "elderly",          "age"),
        ("Gender", "male",             "gender"),
        ("Gender", "female",           "gender"),
        ("Body",   "thin",             "body_index"),
        ("Body",   "normal",           "body_index"),
        ("Body",   "obese",            "body_index"),
    ]

    cells_info = []
    print(f"\n{'Dim':<8} {'Group':<15} {'Mean':>8}")
    for dim, grp, col in groups_of_interest:
        grp_rows = [r for r in tattoo_rows if r[col] == grp]
        per_face: dict[str, list[float]] = defaultdict(list)
        for r in grp_rows:
            fid = r["model"] + "||" + r["face_folder"]
            per_face[fid].append(r["delta"])
        face_arr = np.array([np.mean(v) for v in per_face.values()])
        mean_val = np.mean(face_arr)
        print(f"{dim:<8} {grp:<15} {mean_val:>+8.3f}  (n_faces={len(face_arr)})")
        cells_info.append((dim, grp, mean_val, face_arr))

    p_raws = [wilcoxon_p(arr) for _, _, _, arr in cells_info]
    p_bh = bh_correct(p_raws)
    print("\nSignificance:")
    for i, (dim, grp, mean_val, arr) in enumerate(cells_info):
        pb = p_bh[i]
        sig = "***" if pb < 0.001 else ("*" if pb < 0.05 else "ns")
        print(f"  {dim:<8} {grp:<15} mean={mean_val:>+.3f} p_BH={pb:.4f} {sig}")


def compute_body_compensation(rows: list[dict]):
    print("\n=== tab:body_compensation ===")

    fashion_rows = [r for r in rows if r["cat_key"] == "fashion_style"]

    styles_of_interest = [
        "Professional / Business formal",
        "Formal / Evening wear",
        "Smart casual",
        "Vintage / Retro",
        "Worn / Distressed clothing",
    ]
    body_groups = ["thin", "normal", "obese"]

    cells_info = []
    print(f"\n{'Style':<35} {'Thin':>8} {'Normal':>8} {'Obese':>8}")
    for style in styles_of_interest:
        style_rows = [r for r in fashion_rows
                      if style.lower() in r["variation_name"].lower()]
        if not style_rows:
            print(f"  WARNING: No rows for '{style}'")
            continue

        body_means = {}
        for bg in body_groups:
            bg_rows = [r for r in style_rows if r["body_index"] == bg]
            per_face: dict[str, list[float]] = defaultdict(list)
            for r in bg_rows:
                fid = r["model"] + "||" + r["face_folder"]
                per_face[fid].append(r["delta"])
            face_arr = np.array([np.mean(v) for v in per_face.values()])
            body_means[bg] = np.mean(face_arr)
            cells_info.append((style, bg, face_arr))

        th = body_means.get("thin", float("nan"))
        no = body_means.get("normal", float("nan"))
        ob = body_means.get("obese", float("nan"))
        print(f"{style:<35} {th:>+8.3f} {no:>+8.3f} {ob:>+8.3f}")

        # Print compensation %
        if np.isfinite(th) and np.isfinite(ob) and th > 0:
            print(f"  Ob/Th ratio: {ob/th:.2f}x  ({(ob/th-1)*100:.0f}% more for obese)")

    # Body compensation for Worn/Distressed
    print("\nSignificance:")
    p_raws = [wilcoxon_p(arr) for _, _, arr in cells_info]
    p_bh = bh_correct(p_raws)
    for i, (style, bg, arr) in enumerate(cells_info):
        pb = p_bh[i]
        sig = "***" if pb < 0.001 else ("*" if pb < 0.05 else "ns")
        print(f"  {style:<35} {bg:<8} mean={np.mean(arr):>+.3f} p_BH={pb:.4f} {sig}")


def main():
    print("Loading data...")
    rows = load_all_deltas()
    print(f"Total rows loaded: {len(rows)}")

    # Check unique ages
    ages = sorted(set(r["age"] for r in rows))
    print(f"Age values: {ages}")

    compute_overview(rows)
    compute_age_gradient(rows)
    compute_tattoo_flip(rows)
    compute_body_compensation(rows)


if __name__ == "__main__":
    main()
