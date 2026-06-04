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

ROOT = Path(__file__).resolve().parents[2]
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

