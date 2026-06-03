"""Generate a PDF report explaining the significance analysis for both tables."""

import csv
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer,
    Table, TableStyle, PageBreak, HRFlowable, KeepTogether,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "output" / "evaluation" / "model_comparison_20260503_041215"

# ── Colors ───────────────────────────────────────────────────────────────────
C_HEADER   = colors.HexColor("#2C3E50")
C_ACCENT   = colors.HexColor("#2980B9")
C_SIG      = colors.HexColor("#1A7A4A")
C_NONSIG   = colors.HexColor("#C0392B")
C_LIGHT    = colors.HexColor("#EBF5FB")
C_ROWALT   = colors.HexColor("#F8F9FA")
C_BORDER   = colors.HexColor("#BDC3C7")
C_FORMULA  = colors.HexColor("#F4F6F7")
C_BOX      = colors.HexColor("#D5E8F4")
C_BOX2     = colors.HexColor("#FEF9E7")


def make_styles():
    base = getSampleStyleSheet()
    s = {}

    s["title"] = ParagraphStyle("title", parent=base["Title"],
        fontSize=20, textColor=C_HEADER, spaceAfter=6, leading=26)
    s["subtitle"] = ParagraphStyle("subtitle", parent=base["Normal"],
        fontSize=11, textColor=colors.HexColor("#7F8C8D"),
        spaceAfter=18, leading=16)
    s["h1"] = ParagraphStyle("h1", parent=base["Heading1"],
        fontSize=14, textColor=C_HEADER, spaceBefore=18, spaceAfter=6,
        borderPad=0, leading=18)
    s["h2"] = ParagraphStyle("h2", parent=base["Heading2"],
        fontSize=11, textColor=C_ACCENT, spaceBefore=12, spaceAfter=4, leading=15)
    s["h3"] = ParagraphStyle("h3", parent=base["Heading3"],
        fontSize=10, textColor=colors.HexColor("#1A5276"),
        spaceBefore=8, spaceAfter=3, leading=14, fontName="Helvetica-Bold")
    s["body"] = ParagraphStyle("body", parent=base["Normal"],
        fontSize=9.5, leading=15, spaceAfter=6, alignment=TA_JUSTIFY)
    s["formula"] = ParagraphStyle("formula", parent=base["Normal"],
        fontSize=9, leading=14, spaceAfter=4, leftIndent=18,
        fontName="Courier", backColor=C_FORMULA)
    s["caption"] = ParagraphStyle("caption", parent=base["Normal"],
        fontSize=8, textColor=colors.HexColor("#7F8C8D"),
        alignment=TA_CENTER, spaceAfter=4, leading=11)
    s["bullet"] = ParagraphStyle("bullet", parent=base["Normal"],
        fontSize=9.5, leading=14, leftIndent=18, spaceAfter=3,
        bulletIndent=4)
    s["note"] = ParagraphStyle("note", parent=base["Normal"],
        fontSize=8.5, leading=12, textColor=colors.HexColor("#555555"),
        leftIndent=10, spaceAfter=4, alignment=TA_JUSTIFY)
    s["box"] = ParagraphStyle("box", parent=base["Normal"],
        fontSize=9.5, leading=15, spaceAfter=4, alignment=TA_JUSTIFY,
        leftIndent=10, rightIndent=10, backColor=C_BOX)
    s["box2"] = ParagraphStyle("box2", parent=base["Normal"],
        fontSize=9.5, leading=15, spaceAfter=4, alignment=TA_JUSTIFY,
        leftIndent=10, rightIndent=10, backColor=C_BOX2)
    return s


def sig_color(sig):
    if sig in ("***", "**", "*"):
        return C_SIG
    return C_NONSIG


def load_table3():
    path = OUT_DIR / "table3_significance_independent.csv"
    return list(csv.DictReader(open(path)))


def load_big_table():
    path = OUT_DIR / "big_table_significance_independent.csv"
    return list(csv.DictReader(open(path)))


# ── Page template ─────────────────────────────────────────────────────────────

def make_doc(path):
    doc = BaseDocTemplate(
        str(path), pagesize=A4,
        leftMargin=2.2*cm, rightMargin=2.2*cm,
        topMargin=2.5*cm, bottomMargin=2.2*cm,
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin,
                  doc.width, doc.height, id="main")

    def header_footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#95A5A6"))
        canvas.drawString(doc.leftMargin, doc.bottomMargin - 0.6*cm,
                          "Significance Analysis Report — Independent Observations")
        canvas.drawRightString(doc.width + doc.leftMargin, doc.bottomMargin - 0.6*cm,
                               f"Page {doc.page}")
        canvas.restoreState()

    doc.addPageTemplates([PageTemplate(id="main", frames=[frame],
                                       onPage=header_footer)])
    return doc


# ── Content builders ──────────────────────────────────────────────────────────

def build_intro(s):
    elems = []
    elems.append(Paragraph("Significance Analysis Report", s["title"]))
    elems.append(Paragraph(
        "Statistical re-evaluation using independent observations (scenario-level aggregation)",
        s["subtitle"]))
    elems.append(HRFlowable(width="100%", thickness=1.5, color=C_ACCENT, spaceAfter=14))

    elems.append(Paragraph("Overview", s["h1"]))
    elems.append(Paragraph(
        "This report documents the re-evaluation of statistical significance for two tables "
        "in the thesis. The original analysis pooled all individual face predictions across "
        "models and scenarios, producing very large sample sizes (n ≈ 42,000–381,000). "
        "This constitutes <b>pseudo-replication</b>: the same 25 evaluation scenarios are "
        "repeated across 6 models and ~100+ face images, so the effective degrees of freedom "
        "are far smaller than the raw observation count suggests.", s["body"]))
    elems.append(Paragraph(
        "The corrected analysis uses <b>face-level aggregation</b> for both tables, "
        "as recommended by the reviewer. Each base face contributes one mean Δ per cell "
        "(averaged across all matching scenarios and models). Wilcoxon signed-rank tests "
        "are run on the face means, with BH-corrected p-values, bootstrap CIs, "
        "and Cohen's d reported alongside.", s["body"]))

    elems.append(Paragraph("Document structure", s["h2"]))
    for item in [
        "<b>Section 1</b>: What the two tables measure — structure, variables, and notation",
        "<b>Section 2</b>: Unit of independence — why scenario is the correct choice",
        "<b>Section 3</b>: Full statistical pipeline — aggregation, Wilcoxon, BH correction",
        "<b>Section 4</b>: Table 3 results — 40 cells (10 categories × 4 demographics)",
        "<b>Section 5</b>: Detailed table results — 437 cells (variation × demographic value)",
    ]:
        elems.append(Paragraph(f"• {item}", s["bullet"]))
    return elems


def build_what_tables_measure(s):
    elems = []
    elems.append(PageBreak())
    elems.append(Paragraph("Section 1 — What the Two Tables Measure", s["h1"]))

    # ── Delta definition ──────────────────────────────────────────────────────
    elems.append(Paragraph("The outcome variable: prediction shift Δ", s["h2"]))
    elems.append(Paragraph(
        "The pipeline uses 6 multimodal language models (Gemma-3, Gemma-4, InternVL3, "
        "LLaVA-v1.6, Pixtral, Qwen3) to evaluate 25 decision scenarios. "
        "Each scenario presents the model with a pair of candidates (option A vs. option B) "
        "and asks it to choose. The model outputs a probability p(A) for choosing option A.", s["body"]))
    elems.append(Paragraph(
        "For each base face, the pipeline measures how visual <b>variations</b> (e.g., adding "
        "a beard, changing clothing) shift the model's preference relative to the unmodified face:", s["body"]))
    elems.append(Paragraph(
        "Δ  =  p(A | with variation)  −  p(A | without variation)",
        s["formula"]))
    elems.append(Spacer(1, 4))
    elems.append(Paragraph(
        "A positive Δ means the variation makes the model more likely to choose option A. "
        "A negative Δ means the variation reduces the probability of choosing A. "
        "Δ = 0 means the variation had no effect on the model's prediction.", s["body"]))
    elems.append(Paragraph(
        "The 25 scenarios span different social contexts (hiring, medical triage, lending, "
        "housing, etc.). For each scenario, Δ is computed for each combination of "
        "face × variation × model, yielding hundreds of thousands of raw observations.", s["body"]))

    # ── Table 3 ───────────────────────────────────────────────────────────────
    elems.append(Paragraph("Table 3 — Category-level overview (10 × 4 = 40 cells)", s["h2"]))
    elems.append(Paragraph(
        "Table 3 provides a high-level overview. It groups variations into 10 <b>appearance "
        "categories</b> and crosses them with 4 <b>demographic dimensions</b>. "
        "Each cell reports the average Δ aggregated over all variations within the category "
        "and over all demographic values within the dimension.", s["body"]))

    elems.append(Paragraph("Appearance categories (rows):", s["h3"]))
    categories = [
        ("Fashion",         "fashion_style:*",                   "Clothing style (casual, formal, worn, etc.)"),
        ("Facial hair",     "facial_hair_male:*",                "Beard and stubble types (male faces only)"),
        ("Eyewear",         "eyewear:*",                         "Glasses and sunglasses"),
        ("Makeup & lips",   "makeup_female:*, lip_makeup_female:*", "Makeup and lip color (female faces only)"),
        ("Tattoos",         "tattoos:*",                         "Presence and placement of tattoos"),
        ("Hair style",      "hair_style:*",                      "Hairstyle (curly, straight, afro, etc.)"),
        ("Skin irreg.",     "skin_irregularities:*",             "Acne, birthmarks, vitiligo, scarring"),
        ("Hair len./color", "hair_length:*, hair_color:*",       "Hair length and color combined"),
        ("Accessories",     "accessories:*, headwear:*",         "Jewelry, bags, hats, headwear"),
        ("Piercings",       "piercings:*",                       "Ear and facial piercings"),
    ]
    cat_data = [["Category", "Variation prefix(es)", "Description"]]
    for cat, prefix, desc in categories:
        cat_data.append([cat, prefix, desc])
    t = Table(cat_data, colWidths=[2.8*cm, 4.2*cm, 8.0*cm], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0),  C_HEADER),
        ("TEXTCOLOR",   (0,0), (-1,0),  colors.white),
        ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 8),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, C_ROWALT]),
        ("GRID",        (0,0), (-1,-1), 0.4, C_BORDER),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",  (0,0), (-1,-1), 3),
        ("BOTTOMPADDING",(0,0),(-1,-1), 3),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING",(0,0), (-1,-1), 4),
        ("ALIGN",       (0,0), (-1,-1), "LEFT"),
    ]))
    elems.append(t)
    elems.append(Spacer(1, 8))

    elems.append(Paragraph("Demographic dimensions (columns):", s["h3"]))
    demo_data = [
        ["Dimension", "Column", "Values included", "Role in Table 3"],
        ["Age",    "age",        "young adult, middle-aged adult, elderly",  "All 3 values pooled"],
        ["Gender", "gender",     "male, female",                             "Both values pooled"],
        ["Ethn.",  "ethnicity",  "Asian, African, European, Middle Eastern, Latino", "All 5 pooled"],
        ["Body",   "body_index", "thin, normal, obese",                      "All 3 values pooled"],
    ]
    t2 = Table(demo_data, colWidths=[1.8*cm, 2.2*cm, 6.5*cm, 3.5*cm], repeatRows=1)
    t2.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0),  C_HEADER),
        ("TEXTCOLOR",   (0,0), (-1,0),  colors.white),
        ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 8),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, C_ROWALT]),
        ("GRID",        (0,0), (-1,-1), 0.4, C_BORDER),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",  (0,0), (-1,-1), 3),
        ("BOTTOMPADDING",(0,0),(-1,-1), 3),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING",(0,0), (-1,-1), 4),
        ("ALIGN",       (0,0), (-1,-1), "LEFT"),
    ]))
    elems.append(t2)
    elems.append(Spacer(1, 6))
    elems.append(Paragraph(
        "Because Table 3 pools all demographic values within each dimension, each of the "
        "four columns tests the same hypothesis: <i>does this appearance category produce "
        "a non-zero average shift, regardless of which specific demographic value?</i> "
        "The four columns therefore share the same p-value per category row.",
        s["note"]))

    # ── Big table ─────────────────────────────────────────────────────────────
    elems.append(Paragraph(
        "Detailed Variation Table — Per-variation × demographic value (437 cells)", s["h2"]))
    elems.append(Paragraph(
        "The detailed table is more granular. It crosses each individual variation "
        "(e.g., <i>fashion_style: Worn/Distressed</i>) with each specific demographic "
        "value (e.g., Age = young adult). This produces:", s["body"]))
    elems.append(Paragraph(
        "cells  =  34 variations  ×  13 demographic values  =  437 tested cells",
        s["formula"]))
    elems.append(Spacer(1, 4))

    elems.append(Paragraph("Demographic values tested (13 total):", s["h3"]))
    dv_data = [
        ["Dimension", "Values (n)", "Values"],
        ["Age",       "3",          "young adult | middle-aged adult | elderly"],
        ["Gender",    "2",          "male | female"],
        ["Ethnicity", "5",          "Asian | African | European | Middle Eastern | Latino"],
        ["Body",      "3",          "thin | normal | obese"],
    ]
    t3 = Table(dv_data, colWidths=[2.5*cm, 1.5*cm, 10.0*cm], repeatRows=1)
    t3.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0),  C_HEADER),
        ("TEXTCOLOR",   (0,0), (-1,0),  colors.white),
        ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 8),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, C_ROWALT]),
        ("GRID",        (0,0), (-1,-1), 0.4, C_BORDER),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",  (0,0), (-1,-1), 3),
        ("BOTTOMPADDING",(0,0),(-1,-1), 3),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING",(0,0), (-1,-1), 4),
        ("ALIGN",       (0,0), (-1,-1), "LEFT"),
    ]))
    elems.append(t3)
    elems.append(Spacer(1, 6))
    elems.append(Paragraph(
        "Unlike Table 3 (which pools all demographic values), the detailed table compares "
        "each variation against a specific demographic subgroup. For example, a cell "
        "<i>Worn/Distressed × Age=young adult</i> tests whether adding worn clothing shifts "
        "predictions specifically when the face is labeled as young adult.", s["body"]))

    return elems


def build_independence_section(s):
    elems = []
    elems.append(Paragraph("Section 2 — Unit of Independence", s["h1"]))

    elems.append(Paragraph(
        "Both Table 3 and the detailed variation table use the <b>base face</b> as the "
        "unit of independence, following the reviewer's recommendation.", s["body"]))
    elems.append(Paragraph(
        "A base face is a unique face image with fixed demographic attributes "
        "(age, gender, ethnicity, body type). The study uses approximately 130–334 unique "
        "faces. Each face contributes exactly <b>one mean Δ per cell</b>, computed by "
        "averaging all its Δ values across scenarios, models, and matching variations. "
        "Wilcoxon signed-rank tests then run on these face means.", s["body"]))

    elems.append(Paragraph(
        "Independence box (both tables)", s["h3"]))
    elems.append(Paragraph(
        "<b>Both tables — Independent unit = base face</b>. Two observations from the same "
        "face (e.g., face F evaluated in scenario 7 and scenario 12) share the same "
        "underlying appearance and are therefore <i>not independent</i>. Each face "
        "contributes exactly one aggregated Δ per cell, giving n ≈ 30–334 independent "
        "observations depending on the variation and demographic filter.",
        s["box"]))
    elems.append(Spacer(1, 8))

    elems.append(Paragraph("Comparison of possible units", s["h2"]))
    comparison = [
        ["Unit", "n per cell", "Used for", "Rationale"],
        ["Individual prediction", "42,000–381,000", "None (excluded)",
         "Extreme pseudo-replication. Same scenario × face repeated across 6 models."],
        ["Scenario", "≤ 25", "None (excluded)",
         "Conservative but limited power. Tests generalizability across decision contexts. "
         "Not used here as reviewer recommended face-level for consistency with other tables."],
        ["Base face\n(both tables)", "30–334", "Table 3 + Detailed table",
         "Reviewer recommendation. Face is the experimental unit for appearance effects. "
         "Each face gets one mean Δ across all its scenarios and models. Consistent with "
         "Tables 1 and 2 which also use faces as the primary measurement unit."],
    ]
    t = Table(comparison, colWidths=[2.4*cm, 2.0*cm, 2.4*cm, 7.2*cm], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0),  C_HEADER),
        ("TEXTCOLOR",   (0,0), (-1,0),  colors.white),
        ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 8),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, C_ROWALT]),
        ("GRID",        (0,0), (-1,-1), 0.4, C_BORDER),
        ("VALIGN",      (0,0), (-1,-1), "TOP"),
        ("TOPPADDING",  (0,0), (-1,-1), 3),
        ("BOTTOMPADDING",(0,0),(-1,-1), 3),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING",(0,0), (-1,-1), 4),
        ("ALIGN",       (0,0), (-1,-1), "LEFT"),
        ("BACKGROUND",  (0,3), (-1,3),  colors.HexColor("#EAFAF1")),
    ]))
    elems.append(t)
    elems.append(Spacer(1, 8))

    elems.append(Paragraph("Power and interpretation", s["h2"]))
    elems.append(Paragraph(
        "With face-level aggregation (n ≈ 30–334 per cell), the Wilcoxon test has "
        "good statistical power. A significant result means the appearance-variation "
        "effect is consistent across faces with that demographic attribute. "
        "Cohen's d quantifies the practical magnitude independently of sample size. "
        "Bootstrap CIs show the uncertainty around the mean face-level effect.", s["body"]))

    elems.append(Paragraph("How both tables aggregate in practice", s["h2"]))
    for item in [
        "Filter rows to official scenarios 1–25.",
        "Filter to the target variation(s) and demographic value(s) for the cell.",
        "Group remaining rows by <b>face_folder</b> (unique base face).",
        "Average Δ within each face → one value per face.",
        "Run Wilcoxon signed-rank (two-sided, vs. zero) on the face means.",
        "Compute Cohen's d = mean / SD of face means.",
        "Compute 95% bootstrap CI from 1000 resamples of face means.",
        "Apply BH correction: across 40 cells for Table 3; across 437 cells for the detailed table.",
    ]:
        elems.append(Paragraph(f"• {item}", s["bullet"]))
    elems.append(Spacer(1, 4))

    return elems


def build_methods(s):
    elems = []
    elems.append(Paragraph("Section 3 — Statistical Pipeline", s["h1"]))

    elems.append(Paragraph("Step 1 — Filter to relevant rows", s["h2"]))
    elems.append(Paragraph(
        "For each cell being tested, rows from <b>paired_deltas.csv</b> (produced by the "
        "evaluation pipeline) are filtered to:", s["body"]))
    for item in [
        "Only official scenarios 1–25 (scenarios outside this range are excluded).",
        "Only the variation(s) belonging to the target category or specific variation name.",
        "Only the target demographic value (for the detailed table) or all values (for Table 3).",
    ]:
        elems.append(Paragraph(f"• {item}", s["bullet"]))
    elems.append(Spacer(1, 4))

    elems.append(Paragraph("paired_deltas.csv columns used:", s["h3"]))
    elems.append(Paragraph(
        "scenario (int 1–25)  |  face_folder (str)  |  variation_name (str)  |  "
        "delta (float)  |  age / gender / ethnicity / body_index (str)",
        s["formula"]))

    elems.append(Paragraph("Step 2a — Face-level aggregation (Table 3)", s["h2"]))
    elems.append(Paragraph(
        "After filtering, Δ values are grouped by <b>face_folder</b> (unique base face). "
        "Within each face, all scenario × model combinations contribute their Δ. "
        "These are averaged to produce one value per face:", s["body"]))
    elems.append(Paragraph(
        "Δ̄_f  =  (1 / N_f)  ×  Σ_{s,m}  Δ_{f,s,m}      for each face f",
        s["formula"]))
    elems.append(Spacer(1, 4))
    elems.append(Paragraph(
        "This produces a list of ~130–320 values: <b>[Δ̄_f1, Δ̄_f2, …, Δ̄_fN]</b>. "
        "Wilcoxon, Cohen's d, and bootstrap CIs are then computed on these face means.",
        s["body"]))
    elems.append(Paragraph("Pseudocode (aggregate_to_faces):", s["h3"]))
    elems.append(Paragraph(
        "face_deltas = defaultdict(list)\n"
        "for row in filtered_rows:\n"
        "    face_deltas[row['face_folder']].append(row['delta'])\n"
        "face_means = [mean(v) for v in face_deltas.values() if v]",
        s["formula"]))

    elems.append(Paragraph("Step 2b — Face-level aggregation (Detailed table)", s["h2"]))
    elems.append(Paragraph(
        "The detailed table uses the same face-level aggregation as Table 3. "
        "After filtering to a specific variation AND a specific demographic value, "
        "Δ values are grouped by <b>face_folder</b>. The demographic value filter "
        "selects only faces with that attribute (e.g., filtering by age='young adult' "
        "keeps only young adult faces). Each of those faces contributes one mean Δ:", s["body"]))
    elems.append(Paragraph(
        "Δ̄_f  =  (1 / N_f)  ×  Σ_{s,m}  Δ_{f,s,m}      for each face f with the target demographic",
        s["formula"]))
    elems.append(Spacer(1, 4))
    elems.append(Paragraph(
        "n varies by demographic value: e.g., ~66 faces for Middle Eastern, ~167 for Asian. "
        "Variations incompatible with a demographic (e.g., facial hair × female) produce "
        "zero face means and are skipped.",
        s["body"]))

    elems.append(Paragraph("Step 3 — Wilcoxon signed-rank test (two-sided, vs. zero)", s["h2"]))
    elems.append(Paragraph(
        "A non-parametric one-sample test is applied to the aggregated means "
        "(face means for Table 3; scenario means for the detailed table), "
        "testing whether the median shift differs from zero:", s["body"]))
    elems.append(Paragraph(
        "H₀: median(Δ̄) = 0      vs.      H₁: median(Δ̄) ≠ 0",
        s["formula"]))
    elems.append(Spacer(1, 4))
    elems.append(Paragraph(
        "The Wilcoxon signed-rank test is preferred over a t-test because normality "
        "cannot be assumed. Pairs where Δ̄ = 0 are discarded (zero_method='wilcox'). "
        "If fewer than 3 non-zero values remain, the cell is skipped.", s["body"]))

    elems.append(Paragraph("Step 3b — Effect size: Cohen's d (Table 3 only)", s["h2"]))
    elems.append(Paragraph(
        "For Table 3, Cohen's d quantifies the practical magnitude of the effect. "
        "For a one-sample test vs. zero, d is computed as the ratio of the mean to the "
        "standard deviation of the face means:", s["body"]))
    elems.append(Paragraph(
        "d  =  mean(Δ̄_f)  /  SD(Δ̄_f)      (one-sample Cohen's d vs. zero)",
        s["formula"]))
    elems.append(Spacer(1, 4))
    elems.append(Paragraph(
        "Interpretation: |d| &lt; 0.2 = negligible, 0.2–0.5 = small, "
        "0.5–0.8 = medium, &gt; 0.8 = large effect.", s["body"]))

    elems.append(Paragraph("Step 3c — Bootstrap CI (Table 3 only)", s["h2"]))
    elems.append(Paragraph(
        "A 95% percentile bootstrap CI is computed on the face means. "
        "1000 resamples are drawn with replacement from the face means, "
        "the mean is computed for each resample, and the 2.5th and 97.5th "
        "percentiles form the interval. This CI reflects uncertainty about "
        "the population mean across faces.", s["body"]))
    elems.append(Paragraph(
        "CI = [percentile(boot_means, 2.5),  percentile(boot_means, 97.5)]",
        s["formula"]))

    elems.append(Paragraph("Step 4 — Benjamini-Hochberg FDR correction", s["h2"]))
    elems.append(Paragraph(
        "To control the false discovery rate (FDR) across all cells tested simultaneously, "
        "BH correction is applied. Both tables use face means as input to BH. "
        "The scope of correction differs by table:", s["body"]))
    for item in [
        "<b>Table 3</b>: BH applied across all <b>40 cells</b> (10 categories × 4 demographics).",
        "<b>Detailed table</b>: BH applied across all <b>437 cells</b> (34 variations × 13 demographic values).",
    ]:
        elems.append(Paragraph(f"• {item}", s["bullet"]))
    elems.append(Spacer(1, 4))
    elems.append(Paragraph(
        "BH correction procedure:", s["h3"]))
    elems.append(Paragraph(
        "1. Sort raw p-values ascending: p_(1) ≤ p_(2) ≤ … ≤ p_(m)\n"
        "2. p_adj,(i) = min( p_(i) × m / i ,  p_adj,(i+1) )\n"
        "3. Apply step-down from largest to smallest rank",
        s["formula"]))
    elems.append(Spacer(1, 4))
    elems.append(Paragraph(
        "Significance thresholds after BH correction: "
        "<font color='#1A7A4A'>*** p &lt; .001 &nbsp;|&nbsp; ** p &lt; .01 &nbsp;|&nbsp; "
        "* p &lt; .05</font> &nbsp;|&nbsp; "
        "<font color='#C0392B'>ns = not significant (p ≥ .05)</font>.", s["body"]))

    elems.append(Paragraph("Why this prevents pseudo-replication", s["h2"]))
    elems.append(Paragraph(
        "The original analysis with n ≈ 381,000 observations treated every individual "
        "prediction as independent. In reality:", s["body"]))
    for bullet in [
        "The same 25 scenarios appear in all models — they are <i>not</i> independent.",
        "Faces within a scenario share the same prompt context.",
        "With n = 381,000, even a mean shift of Δ = 0.001 produces p &lt; .001, "
          "regardless of whether the effect is consistent across scenarios.",
        "The 95% confidence intervals in the original Table 3 already revealed the problem: "
          "9 of 40 cells had CIs crossing zero yet were marked ***.",
    ]:
        elems.append(Paragraph(f"• {bullet}", s["bullet"]))
    elems.append(Spacer(1, 6))
    elems.append(Paragraph(
        "Aggregating to scenario means (n ≤ 25) is the correct unit of analysis: "
        "it treats each scenario as one independent replication of the experiment, "
        "and significant results reflect effects that generalize across evaluation contexts.",
        s["body"]))
    return elems


def build_table3_section(s, rows):
    elems = []
    elems.append(PageBreak())
    elems.append(Paragraph("Section 4 — Table 3 Results", s["h1"]))
    elems.append(Paragraph(
        "Table 3 reports the average prediction shift Δ for each appearance category "
        "broken down by demographic dimension. Categories group related variations "
        "(e.g., Fashion = all fashion_style:* variations). Demographics pool all values "
        "within each dimension (e.g., Age pools young adult + middle-aged + elderly).",
        s["body"]))

    elems.append(Paragraph("Independence summary for Table 3:", s["h3"]))
    elems.append(Paragraph(
        "<b>Unit of independence: base face</b> (n ≈ 130–320 per cell)  |  "
        "BH correction: across all <b>40 cells</b>  |  "
        "Test: Wilcoxon signed-rank, two-sided, vs. zero  |  "
        "Effect size: Cohen's d  |  CIs: 1000-resample bootstrap",
        s["box"]))
    elems.append(Spacer(1, 8))

    # Summary counts
    n_sig = sum(1 for r in rows if r["sig"] != "ns")
    n_ns  = sum(1 for r in rows if r["sig"] == "ns")
    n_faces_example = rows[0]["n_faces"] if rows else "?"
    elems.append(Paragraph(
        f"<b>Result:</b> With face-level independent observations (n ≈ {n_faces_example} faces per cell), "
        f"<font color='#1A7A4A'><b>{n_sig} of 40 cells are significant</b></font> and "
        f"<font color='#C0392B'><b>{n_ns} are not significant</b></font> after BH correction.",
        s["body"]))
    elems.append(Paragraph(
        "Cohen's d interpretation: |d| &lt; 0.2 = negligible, 0.2–0.5 = small, "
        "0.5–0.8 = medium, &gt; 0.8 = large effect. "
        "Bootstrap CI = 95% percentile interval from 1000 resamples of face means.",
        s["note"]))
    elems.append(Spacer(1, 8))

    # Build table
    header = ["Category", "Demo", "n faces", "Mean Δ", "95% Boot CI", "Cohen's d", "p (raw)", "p (BH)", "Sig"]
    col_widths = [2.8*cm, 1.6*cm, 1.2*cm, 1.4*cm, 3.0*cm, 1.4*cm, 1.6*cm, 1.6*cm, 0.9*cm]

    data = [header]
    prev_cat = None
    for r in rows:
        cat_label = r["category"] if r["category"] != prev_cat else ""
        prev_cat = r["category"]
        try:
            lo = float(r["ci_lower_boot"])
            hi = float(r["ci_upper_boot"])
            ci_str = f"[{lo:+.3f}, {hi:+.3f}]"
        except Exception:
            ci_str = "—"
        try:
            d_val = float(r["cohens_d"])
            d_str = f"{d_val:+.3f}"
        except Exception:
            d_str = "—"
        p_raw = float(r["p_raw"])
        p_adj = float(r["p_adj_bh"])
        data.append([
            cat_label,
            r["demographic"],
            r["n_faces"],
            f"{float(r['mean_delta']):+.4f}",
            ci_str,
            d_str,
            f"{p_raw:.3f}" if p_raw >= 0.001 else f"{p_raw:.2e}",
            f"{p_adj:.3f}" if p_adj >= 0.001 else f"{p_adj:.2e}",
            r["sig"],
        ])

    t = Table(data, colWidths=col_widths, repeatRows=1)

    SIG_COL = 8  # index of "Sig" column (0-based)
    style = [
        ("BACKGROUND",  (0,0), (-1,0),  C_HEADER),
        ("TEXTCOLOR",   (0,0), (-1,0),  colors.white),
        ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,0),  7.5),
        ("FONTSIZE",    (0,1), (-1,-1), 7),
        ("ALIGN",       (0,0), (-1,-1), "CENTER"),
        ("ALIGN",       (0,1), (0,-1),  "LEFT"),
        ("ALIGN",       (1,1), (1,-1),  "LEFT"),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, C_ROWALT]),
        ("GRID",        (0,0), (-1,-1), 0.4, C_BORDER),
        ("TOPPADDING",  (0,0), (-1,-1), 3),
        ("BOTTOMPADDING",(0,0),(-1,-1), 3),
        ("LEFTPADDING", (0,0), (-1,-1), 3),
        ("RIGHTPADDING",(0,0), (-1,-1), 3),
    ]

    for i, r in enumerate(rows, start=1):
        if r["sig"] == "ns":
            style.append(("TEXTCOLOR", (SIG_COL,i), (SIG_COL,i), C_NONSIG))
            style.append(("FONTNAME",  (SIG_COL,i), (SIG_COL,i), "Helvetica-Bold"))
            style.append(("BACKGROUND",(0,i), (-1,i), colors.HexColor("#FDEDEC")))
        else:
            style.append(("TEXTCOLOR", (SIG_COL,i), (SIG_COL,i), C_SIG))
            style.append(("FONTNAME",  (SIG_COL,i), (SIG_COL,i), "Helvetica-Bold"))

    t.setStyle(TableStyle(style))
    elems.append(t)
    elems.append(Spacer(1, 6))
    elems.append(Paragraph(
        "Red-shaded rows are not significant after BH correction. "
        "All four demographic columns share the same p-value per category because "
        "the test pools all demographic values within each dimension. "
        "Cohen's d: |d| &lt; 0.2 negligible, 0.2–0.5 small, 0.5–0.8 medium, &gt;0.8 large.",
        s["note"]))

    elems.append(Spacer(1, 10))
    elems.append(Paragraph("Interpretation", s["h2"]))
    sig_cats  = sorted(set(r["category"] for r in rows if r["sig"] != "ns"))
    ns_cats   = sorted(set(r["category"] for r in rows if r["sig"] == "ns"))
    elems.append(Paragraph(
        f"<b>Significant categories</b> (survive independent testing): "
        f"<font color='#1A7A4A'>{', '.join(sig_cats) if sig_cats else 'none'}</font>. "
        "These categories show a consistent mean shift across the 25 evaluation scenarios.",
        s["body"]))
    elems.append(Paragraph(
        f"<b>Non-significant categories</b>: "
        f"<font color='#C0392B'>{', '.join(ns_cats) if ns_cats else 'none'}</font>. "
        "The observed shifts in these categories are not consistent enough across scenarios "
        "to conclude a reliable effect. They should not be presented as statistically "
        "significant in the paper.", s["body"]))
    return elems


def build_big_table_section(s, rows):
    elems = []
    elems.append(PageBreak())
    elems.append(Paragraph(
        "Section 5 — Detailed Table Results", s["h1"]))
    elems.append(Paragraph(
        "The detailed table tests each individual variation (e.g., <i>fashion_style: "
        "Worn/Distressed</i>) for each specific demographic group value "
        "(e.g., Age = young adult). This produces 437 cells across 34 variations "
        "and 13 demographic values.", s["body"]))

    elems.append(Paragraph("Independence summary for the detailed table:", s["h3"]))
    elems.append(Paragraph(
        "Unit of independence: <b>base face</b> (n = faces with matching demographic value)  |  "
        "BH correction: across all <b>437 cells simultaneously</b>  |  "
        "Test: Wilcoxon signed-rank, two-sided, vs. zero  |  "
        "Effect size: Cohen's d  |  CIs: 1000-resample bootstrap",
        s["box"]))
    elems.append(Spacer(1, 8))

    elems.append(Paragraph(
        "Key difference from Table 3: each cell now filters to one specific variation "
        "AND one specific demographic value. The n per cell equals the number of faces "
        "that carry that demographic attribute (e.g., all Asian faces, all young adult faces). "
        "For variations that only apply to certain face types (e.g., facial_hair_male "
        "only applies to male faces), incompatible demographic value combinations produce "
        "zero face means and are omitted from the results.", s["body"]))

    n_sig = sum(1 for r in rows if r["sig"] != "ns")
    n_ns  = sum(1 for r in rows if r["sig"] == "ns")
    elems.append(Paragraph(
        f"<b>Result:</b> "
        f"<font color='#1A7A4A'><b>{n_sig} of {len(rows)} cells are significant</b></font>, "
        f"<font color='#C0392B'><b>{n_ns} are not significant</b></font> after BH correction.",
        s["body"]))
    elems.append(Spacer(1, 8))

    # Summary table: not-significant cells per variation
    elems.append(Paragraph("Non-significant cells by variation", s["h2"]))

    var_stats = {}
    for r in rows:
        v = r["variation"]
        if v not in var_stats:
            var_stats[v] = {"total": 0, "ns": 0}
        var_stats[v]["total"] += 1
        if r["sig"] == "ns":
            var_stats[v]["ns"] += 1

    sum_header = ["Variation", "Sig", "ns", "Total", "% ns"]
    sum_col_w  = [5.5*cm, 1.2*cm, 1.2*cm, 1.2*cm, 1.5*cm]
    sum_data   = [sum_header]
    for v, st in sorted(var_stats.items(), key=lambda x: -x[1]["ns"]):
        pct = 100 * st["ns"] / st["total"]
        sum_data.append([
            v.replace("_", " ").replace(":", ": "),
            str(st["total"] - st["ns"]),
            str(st["ns"]),
            str(st["total"]),
            f"{pct:.0f}%",
        ])

    t_sum = Table(sum_data, colWidths=sum_col_w, repeatRows=1)
    sum_style = [
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
    for i, (v, st) in enumerate(
            sorted(var_stats.items(), key=lambda x: -x[1]["ns"]), start=1):
        pct = 100 * st["ns"] / st["total"]
        if pct == 100:
            sum_style.append(("BACKGROUND", (0,i), (-1,i), colors.HexColor("#FDEDEC")))
        elif pct == 0:
            sum_style.append(("BACKGROUND", (0,i), (-1,i), colors.HexColor("#EAFAF1")))
    t_sum.setStyle(TableStyle(sum_style))
    elems.append(t_sum)
    elems.append(Spacer(1, 4))
    elems.append(Paragraph(
        "Green = all cells significant. Red = all cells non-significant. "
        "Full per-cell results are in big_table_significance_independent.csv.",
        s["note"]))

    # Full per-cell table
    elems.append(PageBreak())
    elems.append(Paragraph("Full results — all cells", s["h2"]))
    elems.append(Paragraph(
        "Each row corresponds to one variation × demographic value combination. "
        "n = number of faces with that demographic attribute that had at least one "
        "valid Δ for this variation. Cells with n &lt; 3 are omitted.",
        s["note"]))
    elems.append(Spacer(1, 4))

    header = ["Variation", "Dimension", "Value", "n faces", "Mean Δ", "Cohen's d", "p (BH)", "Sig"]
    col_widths = [3.8*cm, 1.6*cm, 2.4*cm, 1.0*cm, 1.4*cm, 1.4*cm, 1.6*cm, 0.9*cm]
    data = [header]
    for r in rows:
        p_adj = float(r["p_adj_bh"])
        try:
            d_val = float(r["cohens_d"])
            d_str = f"{d_val:+.3f}"
        except Exception:
            d_str = "—"
        data.append([
            r["variation"].replace(":", ": ").replace("_", " "),
            r["demographic_dimension"],
            r["demographic_value"],
            r["n_faces"],
            f"{float(r['mean_delta']):+.4f}",
            d_str,
            f"{p_adj:.3f}" if p_adj >= 0.001 else f"{p_adj:.2e}",
            r["sig"],
        ])

    t = Table(data, colWidths=col_widths, repeatRows=1)
    SIG_COL = 7
    tstyle = [
        ("BACKGROUND",  (0,0), (-1,0),  C_HEADER),
        ("TEXTCOLOR",   (0,0), (-1,0),  colors.white),
        ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,0),  7),
        ("FONTSIZE",    (0,1), (-1,-1), 6.5),
        ("ALIGN",       (0,0), (-1,-1), "CENTER"),
        ("ALIGN",       (0,1), (1,-1),  "LEFT"),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, C_ROWALT]),
        ("GRID",        (0,0), (-1,-1), 0.3, C_BORDER),
        ("TOPPADDING",  (0,0), (-1,-1), 2),
        ("BOTTOMPADDING",(0,0),(-1,-1), 2),
        ("LEFTPADDING", (0,0), (-1,-1), 3),
        ("RIGHTPADDING",(0,0), (-1,-1), 3),
    ]
    for i, r in enumerate(rows, start=1):
        if r["sig"] == "ns":
            tstyle.append(("TEXTCOLOR", (SIG_COL,i), (SIG_COL,i), C_NONSIG))
            tstyle.append(("FONTNAME",  (SIG_COL,i), (SIG_COL,i), "Helvetica-Bold"))
        else:
            tstyle.append(("TEXTCOLOR", (SIG_COL,i), (SIG_COL,i), C_SIG))
            tstyle.append(("FONTNAME",  (SIG_COL,i), (SIG_COL,i), "Helvetica-Bold"))
    t.setStyle(TableStyle(tstyle))
    elems.append(t)
    return elems


def main():
    s = make_styles()
    t3_rows  = load_table3()
    big_rows = load_big_table()

    out_path = OUT_DIR / "significance_analysis_report.pdf"
    doc = make_doc(out_path)

    story = []
    story += build_intro(s)
    story += build_what_tables_measure(s)
    story += build_independence_section(s)
    story += build_methods(s)
    story += build_table3_section(s, t3_rows)
    story += build_big_table_section(s, big_rows)

    doc.build(story)
    print(f"PDF saved to {out_path}")


if __name__ == "__main__":
    main()
