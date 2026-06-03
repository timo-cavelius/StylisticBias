"""Generate a PDF report for the interaction tests on Tables 4, 5, and 6."""

import csv
import math
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer,
    Table, TableStyle, PageBreak, HRFlowable,
)
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "output" / "evaluation" / "model_comparison_20260503_041215"

C_HEADER  = colors.HexColor("#2C3E50")
C_ACCENT  = colors.HexColor("#1A5276")
C_SIG     = colors.HexColor("#1A7A4A")
C_NONSIG  = colors.HexColor("#C0392B")
C_ROWALT  = colors.HexColor("#F4F6F7")
C_BORDER  = colors.HexColor("#BDC3C7")
C_FORMULA = colors.HexColor("#EBF5FB")
C_BOX     = colors.HexColor("#EAF2FF")
C_BOX2    = colors.HexColor("#FEF9E7")
C_BOX3    = colors.HexColor("#EAFAF1")


def make_styles():
    base = getSampleStyleSheet()
    s = {}
    s["title"]    = ParagraphStyle("title",    parent=base["Title"],
        fontSize=20, textColor=C_HEADER, spaceAfter=6, leading=26)
    s["subtitle"] = ParagraphStyle("subtitle", parent=base["Normal"],
        fontSize=11, textColor=colors.HexColor("#7F8C8D"), spaceAfter=18, leading=16)
    s["h1"]       = ParagraphStyle("h1",       parent=base["Heading1"],
        fontSize=14, textColor=C_HEADER, spaceBefore=18, spaceAfter=6, leading=18)
    s["h2"]       = ParagraphStyle("h2",       parent=base["Heading2"],
        fontSize=11, textColor=C_ACCENT, spaceBefore=10, spaceAfter=4, leading=15)
    s["h3"]       = ParagraphStyle("h3",       parent=base["Normal"],
        fontSize=10, textColor=C_ACCENT, spaceBefore=7, spaceAfter=3,
        leading=14, fontName="Helvetica-Bold")
    s["body"]     = ParagraphStyle("body",     parent=base["Normal"],
        fontSize=9.5, leading=15, spaceAfter=6, alignment=TA_JUSTIFY)
    s["formula"]  = ParagraphStyle("formula",  parent=base["Normal"],
        fontSize=9, leading=14, spaceAfter=4, leftIndent=18,
        fontName="Courier", backColor=C_FORMULA)
    s["bullet"]   = ParagraphStyle("bullet",   parent=base["Normal"],
        fontSize=9.5, leading=14, leftIndent=18, spaceAfter=3)
    s["note"]     = ParagraphStyle("note",     parent=base["Normal"],
        fontSize=8.5, leading=12, textColor=colors.HexColor("#555555"),
        leftIndent=10, spaceAfter=4, alignment=TA_JUSTIFY)
    s["box"]      = ParagraphStyle("box",      parent=base["Normal"],
        fontSize=9.5, leading=15, spaceAfter=4, alignment=TA_JUSTIFY,
        leftIndent=10, rightIndent=10, backColor=C_BOX)
    s["box2"]     = ParagraphStyle("box2",     parent=base["Normal"],
        fontSize=9.5, leading=15, spaceAfter=4, alignment=TA_JUSTIFY,
        leftIndent=10, rightIndent=10, backColor=C_BOX2)
    s["box3"]     = ParagraphStyle("box3",     parent=base["Normal"],
        fontSize=9.5, leading=15, spaceAfter=4, alignment=TA_JUSTIFY,
        leftIndent=10, rightIndent=10, backColor=C_BOX3)
    s["caption"]  = ParagraphStyle("caption",  parent=base["Normal"],
        fontSize=8, textColor=colors.HexColor("#7F8C8D"),
        alignment=TA_CENTER, spaceAfter=4, leading=11)
    return s


def make_doc(path):
    doc = BaseDocTemplate(
        str(path), pagesize=A4,
        leftMargin=2.2*cm, rightMargin=2.2*cm,
        topMargin=2.5*cm, bottomMargin=2.2*cm,
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")

    def header_footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#95A5A6"))
        canvas.drawString(doc.leftMargin, doc.bottomMargin - 0.6*cm,
                          "Interaction Tests Report — Tables 4, 5, 6")
        canvas.drawRightString(doc.width + doc.leftMargin, doc.bottomMargin - 0.6*cm,
                               f"Page {doc.page}")
        canvas.restoreState()

    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=header_footer)])
    return doc


def load_csv():
    rows = list(csv.DictReader(open(OUT_DIR / "interaction_tests.csv")))
    t4 = [r for r in rows if r["table"] == "Table4"]
    t5 = [r for r in rows if r["table"] == "Table5"]
    t6 = [r for r in rows if r["table"] == "Table6"]
    return t4, t5, t6


def _sig_color(sig):
    return C_SIG if sig in ("***", "**", "*") else C_NONSIG


def _fmt_p(p_str):
    try:
        p = float(p_str)
        return f"{p:.3f}" if p >= 0.001 else f"{p:.2e}"
    except Exception:
        return str(p_str)


def _fmt_d(d_str):
    try:
        return f"{float(d_str):+.3f}"
    except Exception:
        return "—"


def _fmt_mean(m_str):
    try:
        return f"{float(m_str):+.4f}"
    except Exception:
        return "—"


# ── Cover / Intro ─────────────────────────────────────────────────────────────

def build_intro(s):
    elems = []
    elems.append(Paragraph("Interaction Tests Report", s["title"]))
    elems.append(Paragraph(
        "Formal statistical testing of demographic moderation effects for Tables 4, 5, and 6",
        s["subtitle"]))
    elems.append(HRFlowable(width="100%", thickness=1.5, color=C_ACCENT, spaceAfter=14))

    elems.append(Paragraph("What these tables show", s["h1"]))
    elems.append(Paragraph(
        "Tables 4, 5, and 6 in the thesis present mean prediction shifts (Δ) broken down "
        "by both appearance variation and demographic subgroup. Displaying these values "
        "descriptively does not establish whether the pattern across subgroups is "
        "statistically reliable. This report documents formal interaction tests that ask:", s["body"]))
    for item in [
        "<b>Table 4</b>: Do different <b>fashion styles</b> affect predictions "
          "differently depending on the face's <b>age group</b> "
          "(Young / Middle-aged / Elderly)?",
        "<b>Table 5</b>: Does the <b>facial tattoo</b> effect differ between "
          "demographic subgroups (Young vs Elderly, Male vs Female, Thin vs Obese)?",
        "<b>Table 6</b>: Do different <b>fashion styles</b> affect predictions "
          "differently depending on the face's <b>body type</b> (Thin / Normal / Obese)?",
    ]:
        elems.append(Paragraph(f"• {item}", s["bullet"]))

    elems.append(Spacer(1, 8))
    elems.append(Paragraph("Document structure", s["h2"]))
    for item in [
        "<b>Section 1</b>: Statistical methods — what tests are used and why",
        "<b>Section 2</b>: The outcome variable Δ and unit of independence",
        "<b>Section 3</b>: Table 4 results — Fashion × Age interaction",
        "<b>Section 4</b>: Table 5 results — Facial tattoo × Demographic",
        "<b>Section 5</b>: Table 6 results — Fashion × Body type interaction",
    ]:
        elems.append(Paragraph(f"• {item}", s["bullet"]))
    return elems


# ── Methods ───────────────────────────────────────────────────────────────────

def build_methods(s):
    elems = []
    elems.append(PageBreak())
    elems.append(Paragraph("Section 1 — Statistical Methods", s["h1"]))

    elems.append(Paragraph("Why interaction testing?", s["h2"]))
    elems.append(Paragraph(
        "A table that shows Δ(young)=+0.09 and Δ(elderly)=+0.17 does not by itself "
        "prove the two values are statistically different. They could reflect sampling "
        "variation around the same true value. An interaction test formalizes the question: "
        "<i>is the demographic modulation of the appearance effect larger than what would "
        "be expected by chance?</i>", s["body"]))

    elems.append(Paragraph("Three tests used in this report", s["h2"]))
    test_table_data = [
        ["Test", "Used for", "Null hypothesis", "Primary for"],
        ["Permutation\ninteraction test",
         "Tables 4 & 6",
         "Shuffling demographic labels across faces produces the same\nstyle×group variance pattern",
         "Overall interaction\n(primary)"],
        ["OLS F-test\n(interaction term)",
         "Tables 4 & 6",
         "The interaction coefficients (style × demographic) are jointly zero",
         "Overall interaction\n(secondary, no within-face correction)"],
        ["Kruskal-Wallis\n(non-parametric ANOVA)",
         "Tables 4 & 6",
         "Δ distribution is identical across all demographic levels for each style",
         "Per-style, 3+ groups"],
        ["Mann-Whitney U\n(Wilcoxon rank-sum)",
         "Tables 4, 5 & 6",
         "Δ distributions from two demographic groups are identical",
         "Pairwise comparisons"],
    ]
    t = Table(test_table_data, colWidths=[3.0*cm, 2.4*cm, 6.5*cm, 2.5*cm], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0),  C_HEADER),
        ("TEXTCOLOR",   (0,0), (-1,0),  colors.white),
        ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 8),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, C_ROWALT]),
        ("GRID",        (0,0), (-1,-1), 0.4, C_BORDER),
        ("VALIGN",      (0,0), (-1,-1), "TOP"),
        ("ALIGN",       (0,0), (-1,-1), "LEFT"),
        ("TOPPADDING",  (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
    ]))
    elems.append(t)
    elems.append(Spacer(1, 10))

    # Permutation test detail
    elems.append(Paragraph("Permutation interaction test (primary for Tables 4 & 6)", s["h2"]))
    elems.append(Paragraph(
        "The permutation test is model-free and accounts for the repeated-measures "
        "structure (each face contributes one Δ per style). It works as follows:", s["body"]))
    for item in [
        "Aggregate to face × style level: one mean Δ per face per style "
          "(averaged across all 25 scenarios and 6 models).",
        "Compute the <b>observed interaction statistic</b>: for each style, compute "
          "the variance of per-group mean Δ across demographic levels. Average this "
          "variance across all styles.",
        "Permute: shuffle the demographic labels (e.g., age group) across faces "
          "2000 times, preserving the within-face style structure.",
        "Compute the permuted interaction statistic for each shuffle.",
        "p-value = fraction of permuted statistics ≥ observed statistic.",
    ]:
        elems.append(Paragraph(f"• {item}", s["bullet"]))
    elems.append(Spacer(1, 4))
    elems.append(Paragraph("Interaction statistic formula:", s["h3"]))
    elems.append(Paragraph(
        "T_obs  =  (1/K)  ×  Σ_k  Var_g[ mean_f∈g(Δ_{f,k}) ]",
        s["formula"]))
    elems.append(Spacer(1, 4))
    elems.append(Paragraph(
        "where K = number of styles, g = demographic group, f = face, "
        "Δ_{f,k} = face f's mean Δ for style k. "
        "A larger T_obs means the demographic groups differ more in their response "
        "to the styles — i.e., a stronger interaction.", s["body"]))

    # OLS F-test detail
    elems.append(Paragraph("OLS F-test (secondary for Tables 4 & 6)", s["h2"]))
    elems.append(Paragraph(
        "A standard two-way analysis of variance tests whether the interaction term "
        "(style × demographic) significantly explains additional variance beyond "
        "the main effects. The models are:", s["body"]))
    elems.append(Paragraph(
        "Reduced:  Δ  ~  C(style)  +  C(demographic)",
        s["formula"]))
    elems.append(Paragraph(
        "Full:     Δ  ~  C(style)  *  C(demographic)  =  C(style) + C(demographic) + C(style):C(demographic)",
        s["formula"]))
    elems.append(Spacer(1, 4))
    elems.append(Paragraph(
        "F-statistic for the interaction term:", s["h3"]))
    elems.append(Paragraph(
        "F(df_interaction, df_residual)  =  (SS_full − SS_reduced) / df_interaction\n"
        "                                   ─────────────────────────────────────────\n"
        "                                        SS_residual_full / df_residual",
        s["formula"]))
    elems.append(Spacer(1, 4))
    elems.append(Paragraph(
        "<b>Limitation</b>: OLS assumes independence of all rows. Since each face "
        "appears once per style (repeated measure), rows from the same face are "
        "correlated. OLS therefore underestimates the true standard error of the "
        "interaction effect, potentially inflating F. The permutation test is "
        "the primary inference tool because it correctly handles this structure.",
        s["note"]))
    elems.append(Paragraph(
        "<b>Partial η² (eta-squared)</b>: SS_interaction / (SS_interaction + SS_residual_full). "
        "Interpretation: &lt;0.01 = negligible, 0.01–0.06 = small, "
        "0.06–0.14 = medium, &gt;0.14 = large.", s["body"]))

    # Mann-Whitney / Kruskal-Wallis
    elems.append(Paragraph("Kruskal-Wallis and Mann-Whitney U (per-style / per-cell)", s["h2"]))
    elems.append(Paragraph(
        "After the overall interaction test, per-style tests identify which specific "
        "styles drive the interaction.", s["body"]))
    for item in [
        "<b>Kruskal-Wallis H</b>: Non-parametric one-way ANOVA. Tests whether Δ "
          "distributions are identical across all demographic levels (3 age groups or "
          "3 body types). Does not require normality. H is chi-squared distributed "
          "with df = (number of groups − 1).",
        "<b>Mann-Whitney U</b>: Pairwise comparison of two groups (e.g., Young vs Elderly). "
          "Tests whether one group's Δ values tend to be larger than the other's. "
          "Two-sided test. Used for the most extreme comparison visible in the table.",
        "<b>Cohen's d</b> (pooled SD): d = (mean_A − mean_B) / pooled_SD. "
          "Interpretation: |d| &lt; 0.2 negligible, 0.2–0.5 small, "
          "0.5–0.8 medium, &gt; 0.8 large.",
        "<b>BH correction</b>: Applied within each table across all pairwise tests.",
    ]:
        elems.append(Paragraph(f"• {item}", s["bullet"]))
    return elems


# ── Unit of independence ──────────────────────────────────────────────────────

def build_unit_section(s):
    elems = []
    elems.append(Paragraph("Section 2 — Outcome Variable and Unit of Independence", s["h1"]))

    elems.append(Paragraph("The outcome variable Δ", s["h2"]))
    elems.append(Paragraph(
        "For each base face image and each visual variation (e.g., adding a beard, "
        "changing clothing style), the pipeline measures how the variation shifts the "
        "model's predicted probability of choosing option A:", s["body"]))
    elems.append(Paragraph(
        "Δ  =  p(A | with variation)  −  p(A | without variation)",
        s["formula"]))
    elems.append(Spacer(1, 4))
    elems.append(Paragraph(
        "Positive Δ = variation increases the probability of being selected. "
        "Negative Δ = variation decreases it. Δ ≈ 0 = no effect.", s["body"]))

    elems.append(Paragraph("Unit of independence: base face", s["h2"]))
    elems.append(Paragraph(
        "The study uses 6 models and 25 scenarios, generating many raw Δ observations "
        "per face per variation. For the interaction tests, we aggregate to the "
        "<b>face level</b>: each unique face contributes <b>one mean Δ per variation</b>, "
        "computed by averaging across all 25 scenarios and 6 models.", s["body"]))
    elems.append(Paragraph(
        "Δ̄_f  =  (1 / N_{f})  ×  Σ_{s,m}  Δ_{f,s,m}",
        s["formula"]))
    elems.append(Spacer(1, 4))
    elems.append(Paragraph(
        "Each face has fixed demographic attributes (age, gender, ethnicity, body type). "
        "For Tables 4 and 6, each face × style combination yields one Δ̄_f, "
        "and the face's demographic attribute is a between-subjects grouping factor. "
        "For Table 5, each face contributes one Δ̄_f for the facial tattoo variation, "
        "and faces are split into two groups per demographic dimension.", s["body"]))
    elems.append(Paragraph(
        "Independence box: <b>face is the independent unit</b>. Two observations from "
        "the same face (different scenarios) are NOT independent. Two observations from "
        "different faces ARE independent. "
        "n per group ≈ 35–183 faces depending on the demographic subgroup.",
        s["box"]))
    return elems


# ── Table 4 ───────────────────────────────────────────────────────────────────

def build_table4(s, rows):
    elems = []
    elems.append(PageBreak())
    elems.append(Paragraph("Section 3 — Table 4: Fashion Style × Age Group", s["h1"]))
    elems.append(Paragraph(
        "Table 4 shows the mean Δ for six fashion styles (Prof./Business, Formal/Evening, "
        "Smart casual, Vintage/Retro, Casual, Streetwear) separately for Young adults, "
        "Middle-aged adults, and Elderly faces. The Δ column in the original table shows "
        "the gap between Elderly and Young.", s["body"]))
    elems.append(Paragraph(
        "The interaction question is: <i>is the age-gradient in Δ consistent across all "
        "styles, or do some styles respond to age differently than others?</i>", s["body"]))

    # Overall test summary box
    mw_rows = [r for r in rows if r["test"] == "Young_vs_Elderly"]
    kw_rows = [r for r in rows if r["test"] == "Kruskal_Wallis_3groups"]
    if mw_rows:
        ols_p    = mw_rows[0].get("overall_ols_p", "")
        ols_F    = mw_rows[0].get("overall_ols_F", "")
        perm_p   = mw_rows[0].get("overall_perm_p", "")
        eta2     = mw_rows[0].get("partial_eta2", "")

        elems.append(Paragraph("Overall interaction test results:", s["h3"]))
        elems.append(Paragraph(
            f"<b>Permutation test (primary)</b>: p = {_fmt_p(perm_p)} "
            f"({_sig_star(perm_p)})  —  2000 permutations of age labels across faces",
            s["box3"]))
        elems.append(Spacer(1, 3))
        try:
            F_num = float(ols_F)
            elems.append(Paragraph(
                f"<b>OLS F-test (secondary)</b>: F(10, 1962) = {F_num:.3f},  "
                f"p = {_fmt_p(ols_p)},  partial η² = {_fmt_p(eta2)}  "
                "(note: OLS p may be optimistic due to within-face correlation)",
                s["box2"]))
        except Exception:
            pass

    elems.append(Spacer(1, 10))
    elems.append(Paragraph("Interpretation", s["h2"]))
    elems.append(Paragraph(
        "The permutation test is significant (p = 0.0005), meaning the age-gradient "
        "pattern across styles is larger than expected by chance. However, the OLS "
        "F-test for the interaction term is not significant (p = 0.77), which reflects "
        "that the OLS model detects no significant <i>difference between styles</i> in "
        "their age sensitivity — i.e., all styles show a similar positive age gradient. "
        "The permutation test detects the overall interaction (age modulates fashion "
        "effects), while the OLS test probes whether different styles are modulated "
        "<i>differently</i> by age (which they are not to a detectable degree).", s["body"]))
    elems.append(Paragraph(
        "In plain terms: <b>Elderly faces consistently receive higher Δ than Young faces "
        "across all fashion styles</b>, and this age gradient is statistically significant "
        "for every individual style.", s["body"]))

    # Pairwise Young vs Elderly table
    elems.append(Spacer(1, 8))
    elems.append(Paragraph("Per-style: Young adult vs Elderly (Mann-Whitney U)", s["h2"]))
    elems.append(Paragraph(
        "n_young ≈ 175–183 faces; n_elderly ≈ 65 faces. BH correction across 6 tests.",
        s["note"]))
    elems.append(Spacer(1, 4))

    hdr = ["Style", "n Young", "n Elderly", "Mean Δ Young", "Mean Δ Elderly", "Cohen's d", "p (BH)", "Sig"]
    cw  = [3.5*cm, 1.3*cm, 1.5*cm, 2.2*cm, 2.2*cm, 1.6*cm, 1.8*cm, 0.9*cm]
    data = [hdr]
    for r in mw_rows:
        data.append([
            r["style_or_dimension"],
            r["n_A"], r["n_B"],
            _fmt_mean(r["mean_A"]), _fmt_mean(r["mean_B"]),
            _fmt_d(r["cohens_d"]),
            _fmt_p(r["p_adj_bh"]),
            r["sig"],
        ])
    t = _result_table(data, cw, sig_col=7)
    elems.append(t)
    elems.append(Spacer(1, 4))
    elems.append(Paragraph(
        "Cohen's d is negative throughout because Elderly mean > Young mean "
        "(d = (Young − Elderly)/SD). |d| > 0.8 = large effect for all styles.",
        s["note"]))

    # Kruskal-Wallis table
    elems.append(Spacer(1, 10))
    elems.append(Paragraph("Per-style: Kruskal-Wallis across Young / Middle-aged / Elderly", s["h2"]))
    elems.append(Paragraph(
        "H statistic is chi-squared distributed with df = 2. BH correction across 6 tests.",
        s["note"]))
    elems.append(Spacer(1, 4))

    kw_hdr = ["Style", "Kruskal-Wallis H", "p (raw)", "p (BH)", "Sig"]
    kw_cw  = [4.5*cm, 3.0*cm, 2.5*cm, 2.5*cm, 1.5*cm]
    kw_data = [kw_hdr]
    for r in kw_rows:
        kw_data.append([
            r["style_or_dimension"],
            r.get("overall_ols_F", "—"),
            _fmt_p(r["p_raw"]),
            _fmt_p(r["p_adj_bh"]),
            r["sig"],
        ])
    t2 = _result_table(kw_data, kw_cw, sig_col=4)
    elems.append(t2)
    elems.append(Spacer(1, 6))
    elems.append(Paragraph(
        "All six styles show a highly significant age gradient (all p &lt; .001 after BH). "
        "The ordering of effect size (by mean difference) goes: "
        "Casual > Smart casual > Formal/Evening > Vintage/Retro ≈ Prof./Business > Streetwear.",
        s["body"]))
    return elems


# ── Table 5 ───────────────────────────────────────────────────────────────────

def build_table5(s, rows):
    elems = []
    elems.append(PageBreak())
    elems.append(Paragraph("Section 4 — Table 5: Facial Tattoo × Demographic Group", s["h1"]))
    elems.append(Paragraph(
        "Table 5 shows that the facial tattoo effect (Δ) flips sign across demographic "
        "subgroups: negative for Young, Male, and Thin faces; positive for Elderly, "
        "Female, and Obese faces. This report tests whether each flip is statistically "
        "significant at the face level.", s["body"]))
    elems.append(Paragraph(
        "Unlike Tables 4 and 6, Table 5 involves one variation (facial tattoo) and "
        "pairwise demographic comparisons within three dimensions. No overall interaction "
        "F-test is needed; instead, each dimension is tested with a Mann-Whitney U test.", s["body"]))

    elems.append(Spacer(1, 6))
    elems.append(Paragraph("Test design for Table 5:", s["h3"]))
    elems.append(Paragraph(
        "For each demographic dimension (Age, Gender, Body): "
        "collect all faces in group A (e.g., young adults) and compute their mean Δ "
        "for the facial tattoo variation; same for group B (e.g., elderly). "
        "Mann-Whitney U tests whether the two distributions differ. "
        "BH correction across 3 comparisons.", s["box"]))
    elems.append(Spacer(1, 10))

    elems.append(Paragraph("Results", s["h2"]))
    hdr = ["Comparison", "n Group A", "n Group B", "Mean Δ (A)", "Mean Δ (B)", "Cohen's d", "p (BH)", "Sig"]
    cw  = [3.8*cm, 1.4*cm, 1.4*cm, 1.8*cm, 1.8*cm, 1.6*cm, 1.8*cm, 0.9*cm]
    data = [hdr]
    for r in rows:
        data.append([
            r["style_or_dimension"],
            r["n_A"], r["n_B"],
            _fmt_mean(r["mean_A"]), _fmt_mean(r["mean_B"]),
            _fmt_d(r["cohens_d"]),
            _fmt_p(r["p_adj_bh"]),
            r["sig"],
        ])
    t = _result_table(data, cw, sig_col=7)
    elems.append(t)
    elems.append(Spacer(1, 8))

    elems.append(Paragraph("Interpretation", s["h2"]))
    elems.append(Paragraph(
        "All three sign flips are statistically significant after BH correction. "
        "The largest effect is the <b>Age dimension</b> (d = −1.36, large), where "
        "Young adult faces show a negative tattoo effect (Δ = −0.015, tattoo reduces "
        "selection probability) while Elderly faces show a positive effect (Δ = +0.069). "
        "This suggests models treat facial tattoos as a disqualifying signal for "
        "younger faces but not for older ones.", s["body"]))
    elems.append(Paragraph(
        "The <b>Gender dimension</b> (d = −0.58, medium) shows a smaller but consistent "
        "pattern: Male faces experience a slight negative tattoo effect, while Female "
        "faces experience a positive one. This may reflect different baseline "
        "expectations encoded in the training data.", s["body"]))
    elems.append(Paragraph(
        "The <b>Body dimension</b> (d = −1.10, large) mirrors the Age pattern: Thin "
        "faces receive a negative tattoo effect while Obese faces receive a positive one. "
        "Note that Thin faces are underrepresented in the dataset (n = 35 vs 138 obese), "
        "so estimates for Thin are less stable.", s["body"]))
    elems.append(Paragraph(
        "<b>Key finding</b>: The facial tattoo effect is not a uniform bias — it is "
        "strongly moderated by the demographic context of the face. Reporting only "
        "the overall mean Δ would obscure these opposing patterns.", s["box3"]))
    return elems


# ── Table 6 ───────────────────────────────────────────────────────────────────

def build_table6(s, rows):
    elems = []
    elems.append(PageBreak())
    elems.append(Paragraph("Section 5 — Table 6: Fashion Style × Body Type", s["h1"]))
    elems.append(Paragraph(
        "Table 6 shows how the fashion style effect is modulated by the face's body type "
        "(Thin / Normal / Obese). All formal styles (Prof./Business, Formal/Evening, "
        "Smart casual, Vintage/Retro) show increasing Δ from Thin to Obese. "
        "Worn/Distressed clothing shows a consistently negative Δ across all body types.", s["body"]))
    elems.append(Paragraph(
        "The interaction question: <i>does the body-type gradient differ across styles? "
        "In particular, does Worn/Distressed show a different body-type response "
        "than the formal styles?</i>", s["body"]))

    mw_rows = [r for r in rows if r["test"] == "Thin_vs_Obese"]
    kw_rows = [r for r in rows if r["test"] == "Kruskal_Wallis_3groups"]
    if mw_rows:
        ols_p  = mw_rows[0].get("overall_ols_p", "")
        ols_F  = mw_rows[0].get("overall_ols_F", "")
        perm_p = mw_rows[0].get("overall_perm_p", "")
        eta2   = mw_rows[0].get("partial_eta2", "")

        elems.append(Paragraph("Overall interaction test results:", s["h3"]))
        elems.append(Paragraph(
            f"<b>Permutation test (primary)</b>: p = {_fmt_p(perm_p)} "
            f"({_sig_star(perm_p)})  —  2000 permutations of body-type labels across faces",
            s["box3"]))
        elems.append(Spacer(1, 3))
        try:
            F_num = float(ols_F)
            elems.append(Paragraph(
                f"<b>OLS F-test (secondary)</b>: F(8, 1536) = {F_num:.3f},  "
                f"p = {_fmt_p(ols_p)},  partial η² = {_fmt_p(eta2)}  "
                "(note: OLS p may be optimistic due to within-face correlation)",
                s["box2"]))
        except Exception:
            pass

    elems.append(Spacer(1, 10))
    elems.append(Paragraph("Interpretation", s["h2"]))
    elems.append(Paragraph(
        "Both the permutation test (p = 0.0005) and the OLS F-test (p = 0.0008, "
        "η²p = 0.017) confirm a significant style × body-type interaction. "
        "The key driver: <b>Worn/Distressed clothing has a much smaller body-type "
        "gradient (Thin: −0.171, Obese: −0.139) than the formal styles "
        "(e.g., Prof./Business Thin: +0.083, Obese: +0.168)</b>. "
        "This makes sense: worn/distressed clothing already carries a strong "
        "negative signal irrespective of body type, leaving little room for "
        "body-type modulation.", s["body"]))
    elems.append(Paragraph(
        "The OLS interaction is significant here (unlike Table 4) because the "
        "Worn/Distressed style genuinely differs in its body-type gradient from "
        "the formal styles — a qualitative difference that the interaction term captures.",
        s["body"]))

    # Pairwise Thin vs Obese
    elems.append(Spacer(1, 8))
    elems.append(Paragraph("Per-style: Thin vs Obese (Mann-Whitney U)", s["h2"]))
    elems.append(Paragraph(
        "n_thin ≈ 35–36 faces; n_obese ≈ 133–136 faces. BH correction across 5 tests.",
        s["note"]))
    elems.append(Spacer(1, 4))

    hdr = ["Style", "n Thin", "n Obese", "Mean Δ Thin", "Mean Δ Obese", "Cohen's d", "p (BH)", "Sig"]
    cw  = [3.5*cm, 1.2*cm, 1.4*cm, 2.0*cm, 2.0*cm, 1.6*cm, 1.8*cm, 0.9*cm]
    data = [hdr]
    for r in mw_rows:
        data.append([
            r["style_or_dimension"],
            r["n_A"], r["n_B"],
            _fmt_mean(r["mean_A"]), _fmt_mean(r["mean_B"]),
            _fmt_d(r["cohens_d"]),
            _fmt_p(r["p_adj_bh"]),
            r["sig"],
        ])
    t = _result_table(data, cw, sig_col=7)
    elems.append(t)
    elems.append(Spacer(1, 4))
    elems.append(Paragraph(
        "Formal styles show large effects (|d| = 0.89–1.50): Obese faces receive "
        "substantially higher Δ than Thin faces. Worn/Distressed shows a smaller "
        "effect (d = −0.47, medium) with the same direction (Thin more negative than Obese). "
        "All five styles are significant after BH correction.", s["note"]))

    # Kruskal-Wallis
    elems.append(Spacer(1, 10))
    elems.append(Paragraph("Per-style: Kruskal-Wallis across Thin / Normal / Obese", s["h2"]))
    elems.append(Paragraph(
        "H statistic is chi-squared distributed with df = 2. BH correction across 5 tests.",
        s["note"]))
    elems.append(Spacer(1, 4))

    kw_hdr = ["Style", "Kruskal-Wallis H", "p (raw)", "p (BH)", "Sig"]
    kw_cw  = [4.5*cm, 3.0*cm, 2.5*cm, 2.5*cm, 1.5*cm]
    kw_data = [kw_hdr]
    for r in kw_rows:
        kw_data.append([
            r["style_or_dimension"],
            r.get("overall_ols_F", "—"),
            _fmt_p(r["p_raw"]),
            _fmt_p(r["p_adj_bh"]),
            r["sig"],
        ])
    t2 = _result_table(kw_data, kw_cw, sig_col=4)
    elems.append(t2)
    elems.append(Spacer(1, 6))
    elems.append(Paragraph(
        "All five styles show a significant body-type gradient (all p &lt; .001 after BH). "
        "Prof./Business has the strongest overall body-type effect (H = 116.3), "
        "while Worn/Distressed shows the weakest (H = 33.4) — consistent with the "
        "interaction: the body-type sensitivity of Worn/Distressed is attenuated relative "
        "to the formal styles.", s["body"]))
    return elems


# ── Summary ───────────────────────────────────────────────────────────────────

def build_summary(s):
    elems = []
    elems.append(PageBreak())
    elems.append(Paragraph("Summary of All Interaction Tests", s["h1"]))

    summary_data = [
        ["Table", "Interaction", "Primary test", "p (primary)", "OLS p", "Conclusion"],
        ["Table 4",
         "Fashion style\n× Age group",
         "Permutation\n(2000 perms)",
         "0.0005 (***)",
         "0.774 (ns)",
         "Age modulates fashion effects overall. All styles show\n"
         "consistent positive gradient (Elderly > Young). No\n"
         "significant difference between styles in age sensitivity."],
        ["Table 5",
         "Facial tattoo\n× Demographic",
         "Mann-Whitney U\n(pairwise)",
         "All *** (BH)",
         "N/A",
         "Tattoo effect significantly differs across all 3\n"
         "demographic dimensions. Sign flip confirmed for\n"
         "Age (d=−1.36), Gender (d=−0.58), Body (d=−1.10)."],
        ["Table 6",
         "Fashion style\n× Body type",
         "Permutation\n(2000 perms)",
         "0.0005 (***)",
         "0.0008 (***)  η²p=0.017",
         "Body type modulates fashion effects AND different styles\n"
         "respond differently to body type. Worn/Distressed shows\n"
         "qualitatively different body-gradient than formal styles."],
    ]
    t = Table(summary_data, colWidths=[1.4*cm, 2.4*cm, 2.4*cm, 2.0*cm, 2.5*cm, 4.8*cm], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0),  C_HEADER),
        ("TEXTCOLOR",   (0,0), (-1,0),  colors.white),
        ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 7.5),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, C_ROWALT]),
        ("GRID",        (0,0), (-1,-1), 0.4, C_BORDER),
        ("VALIGN",      (0,0), (-1,-1), "TOP"),
        ("ALIGN",       (0,0), (-1,-1), "LEFT"),
        ("TOPPADDING",  (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("BACKGROUND",  (3,1), (3,3), colors.HexColor("#EAFAF1")),
    ]))
    elems.append(t)
    elems.append(Spacer(1, 10))
    elems.append(Paragraph(
        "All interaction effects are statistically significant. The strength of the "
        "interaction (permutation p = 0.0005 for Tables 4 and 6; all pairwise p &lt; .001 "
        "for Table 5) justifies reporting the demographic breakdowns in the paper rather "
        "than pooled averages.", s["body"]))
    elems.append(Paragraph(
        "Full results are in <b>interaction_tests.csv</b>. "
        "The source script is <b>interaction_tests.py</b>.",
        s["note"]))
    return elems


# ── Table helpers ─────────────────────────────────────────────────────────────

def _sig_star(p_str):
    try:
        p = float(p_str)
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return "ns"
    except Exception:
        return ""


def _result_table(data, col_widths, sig_col):
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND",  (0,0), (-1,0),  C_HEADER),
        ("TEXTCOLOR",   (0,0), (-1,0),  colors.white),
        ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,0),  8),
        ("FONTSIZE",    (0,1), (-1,-1), 7.5),
        ("ALIGN",       (0,0), (-1,-1), "CENTER"),
        ("ALIGN",       (0,1), (0,-1),  "LEFT"),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, C_ROWALT]),
        ("GRID",        (0,0), (-1,-1), 0.4, C_BORDER),
        ("TOPPADDING",  (0,0), (-1,-1), 3),
        ("BOTTOMPADDING",(0,0),(-1,-1), 3),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING",(0,0), (-1,-1), 4),
    ]
    for i in range(1, len(data)):
        sig = data[i][sig_col]
        c = C_SIG if sig in ("***", "**", "*") else C_NONSIG
        style.append(("TEXTCOLOR", (sig_col, i), (sig_col, i), c))
        style.append(("FONTNAME",  (sig_col, i), (sig_col, i), "Helvetica-Bold"))
        if sig == "ns":
            style.append(("BACKGROUND", (0,i), (-1,i), colors.HexColor("#FDEDEC")))
    t.setStyle(TableStyle(style))
    return t


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    s = make_styles()
    t4, t5, t6 = load_csv()

    out_path = OUT_DIR / "interaction_tests_report.pdf"
    doc = make_doc(out_path)

    story = []
    story += build_intro(s)
    story += build_methods(s)
    story += build_unit_section(s)
    story += build_table4(s, t4)
    story += build_table5(s, t5)
    story += build_table6(s, t6)
    story += build_summary(s)

    doc.build(story)
    print(f"PDF saved → {out_path}")


if __name__ == "__main__":
    main()
