"""Generate a LaTeX table for category variation strength across models.

Reads category_variation_strength.csv for each model. Matches the tabularx/small
style of the existing paper table, adding significance markers and CI rows.
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "output" / "evaluation"
OUT_DIR = ROOT / "output" / "evaluation" / "model_comparison_20260503_041215"

MODEL_ORDER = ["gemma3", "gemma4", "internvl", "llava_next", "pixtral", "qwen3"]
MODEL_LABELS = {
    "gemma3": "Gemma-3",
    "gemma4": "Gemma-4",
    "internvl": "InternVL3",
    "llava_next": "LLaVA-v1.6",
    "pixtral": "Pixtral",
    "qwen3": "Qwen3",
}
MODEL_ICONS = {
    "gemma3":    "figures/icons/deepmind-icon.png",
    "gemma4":    "figures/icons/deepmind-icon.png",
    "internvl":  "figures/icons/internvl.png",
    "llava_next":"figures/icons/llava-color.png",
    "pixtral":   "figures/icons/pixtral_icon.png",
    "qwen3":     "figures/icons/Qwen_logo.png",
}
CAT_ORDER = ["age", "body_type", "ethnicity", "gender"]
CAT_LABELS = {"age": "Age", "body_type": "Body", "ethnicity": "Ethn.", "gender": "Gender"}


def _sig_marker(p):
    if p is None:
        return r"^{\mathrm{ns}}"
    if p < 0.001:
        return r"^{***}"
    if p < 0.01:
        return r"^{**}"
    if p < 0.05:
        return r"^{*}"
    return r"^{\mathrm{ns}}"


def _fmt(v):
    if v is None:
        return "—"
    return f"{v:.3f}"


def _fmt_ci(lo, hi):
    if lo is None or hi is None:
        return r"\scriptsize\textcolor{gray}{[{-}]}"
    lo_s = f"{lo:.3f}"
    hi_s = f"{hi:.3f}"
    return r"\scriptsize\textcolor{gray}{[" + lo_s + r",\;" + hi_s + r"]}"


def load_data():
    data = {}
    for model in MODEL_ORDER:
        csv_path = EVAL_DIR / model / "category_variation_strength.csv"
        if not csv_path.exists():
            print(f"WARNING: missing {csv_path}")
            data[model] = {}
            continue
        data[model] = {}
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                cat = row["category_type"]
                def _f(k):
                    try: return float(row[k])
                    except: return None
                p_col = row.get("wilcoxon_p_bh") or row.get("wilcoxon_p_bh_within_model")
                data[model][cat] = {
                    "strength": _f("variation_strength"),
                    "ci_lower": _f("ci_lower_95"),
                    "ci_upper": _f("ci_upper_95"),
                    "p_bh": float(p_col) if p_col else None,
                }
    return data


def _shade(value, max_val):
    """Proportional green shade percentage, capped at 60, matching paper style."""
    if value is None or max_val == 0:
        return 0
    return max(1, round(value / max_val * 60))


def build_latex(data):
    # Global max for proportional shading
    all_vals = [
        data[m].get(c, {}).get("strength")
        for m in MODEL_ORDER for c in CAT_ORDER
    ]
    max_val = max(v for v in all_vals if v is not None)

    lines = []
    lines.append(r"\definecolor{cellgreen}{HTML}{74C476}")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Variation strength across demographic attributes. "
        r"Grey values in brackets are 95\% bootstrap confidence intervals (500 resamples). "
        r"All 24 effects are statistically significant vs.\ zero "
        r"(Wilcoxon signed-rank, BH-corrected, all $p < .001$).}"
    )
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(r"\begin{tabularx}{\columnwidth}{l *{4}{>{\centering\arraybackslash}X}}")
    lines.append(r"\toprule")

    # Header: column names
    col_header = " & ".join(r"\textbf{" + CAT_LABELS[c] + r"}" for c in CAT_ORDER)
    lines.append(r"\textbf{Model} & " + col_header + r" \\")

    # CI subheader row — scriptsize grey
    ci_sub = " & ".join(r"\scriptsize\textcolor{gray}{[95\% CI]}" for _ in CAT_ORDER)
    lines.append(r" & " + ci_sub + r" \\")
    lines.append(r"\midrule")

    for model in MODEL_ORDER:
        icon = MODEL_ICONS[model]
        label = MODEL_LABELS[model]

        val_cells = []
        ci_cells  = []

        for cat in CAT_ORDER:
            d  = data[model].get(cat, {})
            s  = d.get("strength")
            lo = d.get("ci_lower")
            hi = d.get("ci_upper")
            p  = d.get("p_bh")

            shade = _shade(s, max_val)
            bg = rf"\cellcolor{{cellgreen!{shade}!white}}"

            val_cells.append(bg + f"${_fmt(s)}$")
            ci_cells.append(bg + _fmt_ci(lo, hi))

        # Value row (with icon)
        lines.append(
            rf"\modelicon{{{icon}}}~{{{label}}}"
            + " & " + " & ".join(val_cells) + r" \\"
        )
        # CI row
        lines.append(" & " + " & ".join(ci_cells) + r" \\")

    lines.append(r"\midrule")

    # Average row — also proportionally shaded
    avg_cells = []
    for cat in CAT_ORDER:
        vals = [data[m].get(cat, {}).get("strength") for m in MODEL_ORDER]
        vals = [v for v in vals if v is not None]
        avg = sum(vals) / len(vals) if vals else None
        shade = _shade(avg, max_val)
        bg = rf"\cellcolor{{cellgreen!{shade}!white}}"
        avg_cells.append(bg + r"\textit{" + _fmt(avg) + r"}")

    lines.append(r"\textit{Average} & " + " & ".join(avg_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")
    lines.append(r"\label{tab:base_variation}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    data = load_data()
    latex = build_latex(data)
    out_path = OUT_DIR / "variation_strength_table.tex"
    out_path.write_text(latex)
    print(f"Saved to {out_path}\n")
    print(latex)


if __name__ == "__main__":
    main()
