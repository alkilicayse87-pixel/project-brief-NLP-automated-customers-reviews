"""Builds docs/Project_Report.pdf from README.md (the README stays the source of truth)."""

import re
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image, ListFlowable, ListItem, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "Project_Report.pdf"
NAVY = colors.HexColor("#14213D")
TEAL = colors.HexColor("#1B998B")

styles = getSampleStyleSheet()
body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=10, leading=14, spaceAfter=6)
cell = ParagraphStyle("cell", parent=body, fontSize=8.5, leading=11, spaceAfter=0)
h1 = ParagraphStyle("h1", parent=styles["Heading1"], textColor=NAVY, fontSize=22, spaceAfter=4)
h2 = ParagraphStyle("h2", parent=styles["Heading2"], textColor=NAVY, fontSize=14, spaceBefore=12, spaceAfter=6)
code = ParagraphStyle("code", parent=styles["Code"], fontSize=7, leading=9, backColor=colors.HexColor("#F4F6F8"), borderPadding=4)


def inline(text):
    text = escape(text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<a href="\2" color="#1B998B">\1</a>', text)
    text = re.sub(r"(?<![\"=>])(https?://[^\s<)]+)", r'<a href="\1" color="#1B998B">\1</a>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r'<font face="Courier" size="9">\1</font>', text)
    return text


def table_flowable(rows):
    header = [c.strip() for c in rows[0].strip("|").split("|")]
    aligns = [("RIGHT" if c.strip().endswith(":") else "LEFT") for c in rows[1].strip("|").split("|")]
    data = [[Paragraph(f"<b>{inline(h)}</b>", cell) for h in header]]
    for r in rows[2:]:
        data.append([Paragraph(inline(c.strip()), cell) for c in r.strip("|").split("|")])
    t = Table(data, repeatRows=1, hAlign="LEFT")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EDF2")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B8C0CC")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]
    style += [("ALIGN", (i, 0), (i, -1), a) for i, a in enumerate(aligns)]
    t.setStyle(TableStyle(style))
    return [t, Spacer(1, 8)]


def list_flowable(items, ordered):
    return ListFlowable(
        [ListItem(Paragraph(inline(i), body), leftIndent=14) for i in items],
        bulletType="1" if ordered else "bullet",
        start=1 if ordered else "•",
        leftIndent=14,
    )


def build():
    lines = (ROOT / "README.md").read_text(encoding="utf-8").splitlines()
    story = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            block = []
            i += 1
            while not lines[i].startswith("```"):
                block.append(lines[i])
                i += 1
            text = "\n".join(block).replace("├── ", "|-- ").replace("└── ", "`-- ").replace("│   ", "|   ")
            story += [Preformatted(text, code), Spacer(1, 6)]
        elif line.startswith("# "):
            story += [Paragraph("Project Report", ParagraphStyle("kicker", parent=body, textColor=TEAL, fontSize=12, spaceAfter=0)),
                      Paragraph(inline(line[2:]), h1)]
        elif line.startswith("## "):
            story.append(Paragraph(inline(line[3:]), h2))
        elif line.startswith("!["):
            path = ROOT / re.search(r"\(([^)]+)\)", line).group(1)
            if path.exists():
                w, h = ImageReader(str(path)).getSize()
                width = 11 * cm
                story += [Image(str(path), width=width, height=width * h / w), Spacer(1, 6)]
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(lines[i])
                i += 1
            story += table_flowable(rows)
            continue
        elif re.match(r"\s*- ", line) or re.match(r"\d+\. ", line):
            ordered = bool(re.match(r"\d+\. ", line))
            items = []
            pat = r"\d+\. " if ordered else r"\s*- "
            while i < len(lines) and re.match(pat, lines[i]):
                items.append(re.sub(pat, "", lines[i], count=1))
                i += 1
            story += [list_flowable(items, ordered), Spacer(1, 4)]
            continue
        elif line.strip():
            story.append(Paragraph(inline(line), body))
        i += 1

    def footer(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(2 * cm, 1.2 * cm, "Project Report - NLP Automated Customer Reviews - Ayse A. Oed and Ivan Matteuzi")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, str(doc.page))

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm,
                            bottomMargin=2 * cm, title="Project Report - NLP Automated Customer Reviews",
                            author="Ayse A. Oed and Ivan Matteuzi")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print("saved", OUT)


if __name__ == "__main__":
    build()
