"""
make_docx.py — Convert all docs/*.md files into a single loom_docs.docx
Run from the docs/ directory:  python make_docx.py
Requires: pip install python-docx
"""

import os
import re
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

OUTPUT_DOC = "loom_docs.docx"

# Order index.md first, then alphabetical
MD_ORDER_FIRST = ["index.md"]

def sorted_md_files(directory):
    files = [f for f in os.listdir(directory) if f.endswith(".md")]
    ordered = [f for f in MD_ORDER_FIRST if f in files]
    rest = sorted(f for f in files if f not in MD_ORDER_FIRST)
    return ordered + rest


def ensure_code_style(doc):
    styles = doc.styles
    if "Code Block" not in styles:
        style = styles.add_style("Code Block", WD_STYLE_TYPE.PARAGRAPH)
        font = style.font
        font.name = "Courier New"
        font.size = Pt(9)
        font.color.rgb = RGBColor(0x1E, 0x1E, 0x1E)
        pf = style.paragraph_format
        pf.left_indent = Inches(0.3)
        pf.space_before = Pt(4)
        pf.space_after = Pt(4)
        # Light grey background via XML shading
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), "F0F0F0")
        style.element.pPr.append(shd)
    return styles["Code Block"]


def add_code_block(doc, code_style, text):
    for line in text.split("\n"):
        p = doc.add_paragraph(line, style=code_style)
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)


def add_horizontal_rule(doc):
    p = doc.add_paragraph()
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "AAAAAA")
    pBdr.append(bottom)
    pPr.append(pBdr)


def parse_inline(run_text):
    """Return list of (text, bold, italic, code) tuples parsed from inline markdown."""
    segments = []
    # Simple inline code: `...`
    pattern = re.compile(r"`([^`]+)`")
    last = 0
    for m in pattern.finditer(run_text):
        if m.start() > last:
            segments.append((run_text[last:m.start()], False, False, False))
        segments.append((m.group(1), False, False, True))
        last = m.end()
    if last < len(run_text):
        segments.append((run_text[last:], False, False, False))

    # Bold/italic pass (applied to non-code segments)
    result = []
    for text, b, i, c in segments:
        if c:
            result.append((text, b, i, c))
            continue
        # **bold**
        parts = re.split(r"\*\*(.+?)\*\*", text)
        for idx, part in enumerate(parts):
            if idx % 2 == 1:
                result.append((part, True, False, False))
            else:
                result.append((part, False, False, False))
    return result


def add_paragraph_with_inline(doc, text, style_name="Normal"):
    p = doc.add_paragraph(style=style_name)
    for segment, bold, italic, code in parse_inline(text):
        if not segment:
            continue
        run = p.add_run(segment)
        run.bold = bold
        run.italic = italic
        if code:
            run.font.name = "Courier New"
            run.font.size = Pt(9)
            run.font.color.rgb = RGBColor(0xC7, 0x25, 0x4F)
    return p


def parse_and_add_table(doc, lines):
    """Parse a markdown table and add it as a Word table."""
    rows = []
    for line in lines:
        line = line.strip()
        if re.match(r"^\|[-:| ]+\|$", line):
            continue  # separator row
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows.append(cells)
    if not rows:
        return
    col_count = max(len(r) for r in rows)
    table = doc.add_table(rows=len(rows), cols=col_count)
    table.style = "Table Grid"
    for r_idx, row in enumerate(rows):
        for c_idx, cell_text in enumerate(row):
            if c_idx >= col_count:
                break
            cell = table.cell(r_idx, c_idx)
            cell.text = ""
            p = cell.paragraphs[0]
            for segment, bold, italic, code in parse_inline(cell_text):
                if not segment:
                    continue
                run = p.add_run(segment)
                run.bold = bold or (r_idx == 0)  # header row bold
                if code:
                    run.font.name = "Courier New"
                    run.font.size = Pt(9)
    doc.add_paragraph()  # spacing after table


def process_md(doc, filepath, code_style):
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()

    in_code = False
    code_fence_lang = ""
    code_lines = []
    in_table = False
    table_lines = []
    i = 0

    while i < len(lines):
        line = lines[i].rstrip("\n")

        # ── Code fence ───────────────────────────────────────────────────
        if line.strip().startswith("```"):
            if not in_code:
                in_code = True
                code_fence_lang = line.strip()[3:].strip()
                code_lines = []
            else:
                add_code_block(doc, code_style, "\n".join(code_lines))
                in_code = False
                code_lines = []
            i += 1
            continue

        if in_code:
            code_lines.append(line)
            i += 1
            continue

        # ── Table detection ──────────────────────────────────────────────
        if line.strip().startswith("|"):
            if not in_table:
                in_table = True
                table_lines = []
            table_lines.append(line)
            i += 1
            continue
        else:
            if in_table:
                parse_and_add_table(doc, table_lines)
                in_table = False
                table_lines = []

        stripped = line.strip()

        # ── Horizontal rule ──────────────────────────────────────────────
        if re.match(r"^---+$", stripped):
            add_horizontal_rule(doc)
            i += 1
            continue

        # ── Headings ─────────────────────────────────────────────────────
        heading_match = re.match(r"^(#{1,6})\s+(.*)", stripped)
        if heading_match:
            level = len(heading_match.group(1))
            text = heading_match.group(2)
            # Strip markdown links from headings
            text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
            doc.add_heading(text, level=min(level, 4))
            i += 1
            continue

        # ── Blank line ───────────────────────────────────────────────────
        if not stripped:
            doc.add_paragraph()
            i += 1
            continue

        # ── Blockquote / note ────────────────────────────────────────────
        if stripped.startswith("> ") or stripped.startswith(">!"):
            text = re.sub(r"^>!?\s*\[!NOTE\]\s*", "", stripped)
            text = re.sub(r"^>\s*", "", text)
            p = add_paragraph_with_inline(doc, text)
            p.paragraph_format.left_indent = Inches(0.3)
            run = p.runs[0] if p.runs else p.add_run()
            run.italic = True
            i += 1
            continue

        # ── List items ───────────────────────────────────────────────────
        list_match = re.match(r"^(\s*)([-*+]|\d+\.)\s+(.*)", line)
        if list_match:
            indent = len(list_match.group(1))
            text = list_match.group(3)
            style = "List Bullet" if indent == 0 else "List Bullet 2"
            add_paragraph_with_inline(doc, text, style)
            i += 1
            continue

        # ── Normal paragraph ─────────────────────────────────────────────
        add_paragraph_with_inline(doc, stripped)
        i += 1

    # Flush any open table at EOF
    if in_table:
        parse_and_add_table(doc, table_lines)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    md_files = sorted_md_files(script_dir)

    doc = Document()

    # Document title
    title = doc.add_heading("Loom / poly — Full Documentation", 0)

    code_style = ensure_code_style(doc)

    for idx, filename in enumerate(md_files):
        filepath = os.path.join(script_dir, filename)
        print(f"  Processing {filename} ...")

        if idx > 0:
            doc.add_page_break()

        process_md(doc, filepath, code_style)

    out_path = os.path.join(script_dir, OUTPUT_DOC)
    doc.save(out_path)
    print(f"\nSaved: {out_path}  ({len(md_files)} files merged)")


if __name__ == "__main__":
    main()
