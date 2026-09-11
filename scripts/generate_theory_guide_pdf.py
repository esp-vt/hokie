#!/usr/bin/env python3
"""
generate_theory_guide_pdf.py
Converts docs/THEORETICAL_FOUNDATIONS_EXPLAINED.md into a publication-grade PDF:
docs/THEORETICAL_FOUNDATIONS_EXPLAINED.pdf using PyMuPDF DocumentWriter & Story.
Includes crisp styling, clean Unicode math blocks in backticks, and clean headers/footers.
"""

import os
import re
import html
import pymupdf

def md_to_html(md_text: str) -> str:
    """
    Converts markdown text to clean HTML styled for PyMuPDF Story rendering.
    """
    lines = md_text.split("\n")
    html_lines = []
    
    in_code_block = False
    code_buffer = []
    in_table = False
    table_rows = []
    in_blockquote = False
    blockquote_buffer = []
    in_list = False
    
    def flush_blockquote():
        nonlocal in_blockquote, blockquote_buffer
        if in_blockquote and blockquote_buffer:
            content = "<br>".join(blockquote_buffer)
            # Check if it's a theorem box or general callout
            if "Theorem" in content or "정리" in content:
                html_lines.append(f"<div class='theorem-box'>{content}</div>")
            elif "대상 독자" in content:
                html_lines.append(f"<div class='meta-box'>{content}</div>")
            else:
                html_lines.append(f"<div class='callout-box'>{content}</div>")
            blockquote_buffer = []
            in_blockquote = False

    def flush_table():
        nonlocal in_table, table_rows
        if in_table and table_rows:
            html_table = ["<table class='data-table'>"]
            for i, row in enumerate(table_rows):
                # check if separator row
                if re.match(r'^\s*\|?\s*:?-+:?\s*(\|?\s*:?-+:?\s*)+\|?\s*$', row):
                    continue
                cells = [c.strip() for c in row.split("|")[1:-1]]
                if not cells:
                    continue
                if i == 0:
                    html_table.append("<thead><tr>" + "".join(f"<th>{inline_format(c)}</th>" for c in cells) + "</tr></thead><tbody>")
                else:
                    html_table.append("<tr>" + "".join(f"<td>{inline_format(c)}</td>" for c in cells) + "</tr>")
            html_table.append("</tbody></table>")
            html_lines.append("\n".join(html_table))
            table_rows = []
            in_table = False

    def flush_list():
        nonlocal in_list
        if in_list:
            html_lines.append("</ul>")
            in_list = False

    def inline_format(text: str) -> str:
        # Strip markdown links [Text](URL) -> Text
        text = re.sub(r'\[(.*?)\]\([^)]+\)', r'\1', text)
        
        # Convert backticks `...` to <code>...</code>
        def replace_code(m):
            code_content = html.escape(m.group(1))
            return f"<code class='math-code'>{code_content}</code>"
        
        # Replace backticks
        text = re.sub(r'`([^`]+)`', replace_code, text)
        
        # Bold **text**
        text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
        
        # Italic *text*
        text = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<em>\1</em>', text)
        
        return text

    for line in lines:
        stripped = line.strip()
        
        # Code block handling
        if stripped.startswith("```"):
            if in_code_block:
                code_str = html.escape("\n".join(code_buffer))
                html_lines.append(f"<pre class='code-block'><code>{code_str}</code></pre>")
                code_buffer = []
                in_code_block = False
            else:
                flush_blockquote()
                flush_table()
                flush_list()
                in_code_block = True
                code_buffer = []
            continue
            
        if in_code_block:
            code_buffer.append(line)
            continue
            
        # Table handling
        if stripped.startswith("|") and stripped.endswith("|"):
            flush_blockquote()
            flush_list()
            in_table = True
            table_rows.append(stripped)
            continue
        else:
            flush_table()
            
        # Blockquote handling
        if stripped.startswith(">"):
            flush_list()
            in_blockquote = True
            bq_text = inline_format(stripped.lstrip("> ").strip())
            blockquote_buffer.append(bq_text)
            continue
        else:
            flush_blockquote()
            
        # Blank line
        if not stripped:
            flush_list()
            continue
            
        # Horizontal rule
        if stripped in ["---", "***", "___"]:
            flush_list()
            html_lines.append("<hr class='divider'>")
            continue
            
        # Headers
        if stripped.startswith("# "):
            flush_list()
            h_text = inline_format(stripped[2:].strip())
            if "제8장" in h_text:
                html_lines.append(f"<div style='page-break-before: always; height: 10px;'></div><h1 class='h1-title'>{h_text}</h1>")
            else:
                html_lines.append(f"<h1 class='h1-title'>{h_text}</h1>")
            continue
        elif stripped.startswith("## "):
            flush_list()
            h_text = inline_format(stripped[3:].strip())
            html_lines.append(f"<h2 class='h2-section'>{h_text}</h2>")
            continue
        elif stripped.startswith("### "):
            flush_list()
            h_text = inline_format(stripped[4:].strip())
            html_lines.append(f"<h3 class='h3-subsection'>{h_text}</h3>")
            continue
        elif stripped.startswith("#### "):
            flush_list()
            h_text = inline_format(stripped[5:].strip())
            html_lines.append(f"<h4 class='h4-subsubsection'>{h_text}</h4>")
            continue
            
        # Unordered or ordered list items
        if re.match(r'^[-*]\s+', stripped) or re.match(r'^\d+\.\s+', stripped):
            if not in_list:
                html_lines.append("<ul class='custom-list'>")
                in_list = True
            item_text = re.sub(r'^[-*]\s+|\d+\.\s+', '', stripped)
            html_lines.append(f"<li>{inline_format(item_text)}</li>")
            continue
        else:
            flush_list()
            
        # Regular paragraph
        html_lines.append(f"<p class='body-p'>{inline_format(stripped)}</p>")
        
    flush_blockquote()
    flush_table()
    flush_list()
    
    return "\n".join(html_lines)

def build_full_html(body_html: str) -> str:
    css = """
    @font-face {
        font-family: 'DroidSans';
        src: url('DroidSansFallbackFull.ttf');
    }
    
    body {
        font-family: 'DroidSans', sans-serif;
        font-size: 8.7pt;
        line-height: 1.48;
        color: #1e293b;
        background-color: #ffffff;
        margin: 0;
        padding: 0;
    }
    
    .h1-title {
        color: #0f172a;
        font-size: 13.8pt;
        font-weight: bold;
        border-bottom: 2.2px solid #1e3a8a;
        padding-bottom: 4px;
        margin-top: 14px;
        margin-bottom: 8px;
    }
    
    .h2-section {
        color: #1e3a8a;
        font-size: 11.0pt;
        font-weight: bold;
        border-bottom: 1.5px solid #93c5fd;
        padding-bottom: 3px;
        margin-top: 11px;
        margin-bottom: 5px;
    }
    
    .h3-subsection {
        color: #1e40af;
        font-size: 9.5pt;
        font-weight: bold;
        margin-top: 9px;
        margin-bottom: 4px;
    }
    
    .h4-subsubsection {
        color: #334155;
        font-size: 8.8pt;
        font-weight: bold;
        margin-top: 7px;
        margin-bottom: 3px;
    }
    
    .body-p {
        margin: 3px 0 5px 0;
        text-align: justify;
    }
    
    .divider {
        border: none;
        border-top: 1px solid #e2e8f0;
        margin: 8px 0;
    }
    
    .meta-box {
        background-color: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 7px 11px;
        margin: 7px 0 9px 0;
        font-size: 8.2pt;
        color: #334155;
        border-radius: 3px;
    }
    
    .theorem-box {
        background-color: #eff6ff;
        border: 1px solid #bfdbfe;
        border-left: 4.5px solid #1d4ed8;
        padding: 8px 12px;
        margin: 7px 0;
        border-radius: 4px;
        font-size: 8.4pt;
        color: #1e293b;
        line-height: 1.46;
    }
    
    .callout-box {
        background-color: #f1f5f9;
        border-left: 3.5px solid #64748b;
        padding: 6px 10px;
        margin: 6px 0;
        border-radius: 3px;
        font-size: 8.2pt;
    }
    
    .code-block {
        background-color: #0f172a;
        color: #38bdf8;
        padding: 7px 10px;
        border-radius: 4px;
        font-family: 'DroidSans', monospace;
        font-size: 7.3pt;
        line-height: 1.32;
        margin: 6px 0;
        overflow-x: hidden;
        white-space: pre-wrap;
    }
    
    .code-block code {
        background: transparent;
        border: none;
        color: #38bdf8;
        padding: 0;
        font-size: 7.3pt;
    }
    
    code.math-code {
        background-color: #f1f5f9;
        color: #0f172a;
        border: 1px solid #cbd5e1;
        padding: 1px 3.5px;
        border-radius: 3px;
        font-family: 'DroidSans', monospace;
        font-size: 8.0pt;
        font-weight: 500;
        white-space: nowrap;
    }
    
    .custom-list {
        margin: 3px 0 5px 14px;
        padding-left: 0;
    }
    
    .custom-list li {
        margin-bottom: 2.5px;
        line-height: 1.40;
    }
    
    .data-table {
        width: 100%;
        border-collapse: collapse;
        margin: 6px 0 9px 0;
        font-size: 7.5pt;
    }
    
    .data-table th, .data-table td {
        border: 1px solid #cbd5e1;
        padding: 4px 6px;
        text-align: left;
        vertical-align: top;
    }
    
    .data-table th {
        background-color: #f1f5f9;
        color: #0f172a;
        font-weight: bold;
    }
    
    .data-table tr:nth-child(even) td {
        background-color: #f8fafc;
    }
    """
    
    full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
{css}
</style>
</head>
<body>
{body_html}
</body>
</html>
"""
    return full_html

def generate_pdf():
    input_md_path = "/home/eun/neuroworld_lm/docs/THEORETICAL_FOUNDATIONS_EXPLAINED.md"
    temp_pdf_path = "/tmp/raw_theoretical_guide.pdf"
    output_pdf_path = "/home/eun/neuroworld_lm/docs/THEORETICAL_FOUNDATIONS_EXPLAINED.pdf"
    font_dir = "/usr/share/fonts/truetype/droid"
    
    with open(input_md_path, "r", encoding="utf-8") as f:
        md_text = f.read()
        
    print(f"Reading markdown from {input_md_path} ({len(md_text)} chars)...")
    
    body_html = md_to_html(md_text)
    full_html = build_full_html(body_html)
    
    # Step 1: Render base HTML with PyMuPDF Story
    archive = pymupdf.Archive(font_dir)
    writer = pymupdf.DocumentWriter(temp_pdf_path)
    story = pymupdf.Story(html=full_html, archive=archive)
    
    mediabox = pymupdf.paper_rect("a4") # 595.3 x 841.9 pt
    margin_top = 65 # pt: gives ample 29pt gap below header at y=36
    margin_bottom = 50 # pt
    margin_left = 38 # pt
    margin_right = 38 # pt
    
    where = pymupdf.Rect(margin_left, margin_top, mediabox.width - margin_right, mediabox.height - margin_bottom)
    
    more = 1
    while more:
        device = writer.begin_page(mediabox)
        more, filled = story.place(where)
        story.draw(device)
        writer.end_page()
        
    writer.close()
    
    # Step 2: Post-process PDF to add Clean Helvetica Headers and Footers
    doc = pymupdf.open(temp_pdf_path)
    total_pages = len(doc)
    
    header_text = "Hokie-LM (NeuroWorld-LM) Theoretical Foundations & Theorems Guide"
    footer_doc_id = "NeuroWorld-LM Technical Report | Department of Computer Science, Virginia Tech"
    
    for i, page in enumerate(doc):
        # Running header on pages > 1
        if i > 0:
            # Header line
            page.draw_line(
                pymupdf.Point(margin_left, 36),
                pymupdf.Point(mediabox.width - margin_right, 36),
                color=(0.75, 0.8, 0.88),
                width=0.75
            )
            # Header text (using standard clean helv)
            page.insert_text(
                pymupdf.Point(margin_left, 28),
                header_text,
                fontname="helv",
                fontsize=7.5,
                color=(0.35, 0.4, 0.5)
            )
        
        # Footer line
        page.draw_line(
            pymupdf.Point(margin_left, mediabox.height - 30),
            pymupdf.Point(mediabox.width - margin_right, mediabox.height - 30),
            color=(0.75, 0.8, 0.88),
            width=0.75
        )
        
        # Left footer (clean helv)
        page.insert_text(
            pymupdf.Point(margin_left, mediabox.height - 18),
            footer_doc_id,
            fontname="helv",
            fontsize=7.2,
            color=(0.4, 0.45, 0.55)
        )
        
        # Right footer: Page number
        page_str = f"Page {i+1} of {total_pages}"
        text_width = len(page_str) * 4.2 # approx width for helv 7.2pt
        page.insert_text(
            pymupdf.Point(mediabox.width - margin_right - text_width - 15, mediabox.height - 18),
            page_str,
            fontname="helv",
            fontsize=7.2,
            color=(0.3, 0.35, 0.45)
        )
        
    doc.save(output_pdf_path, deflate=True, garbage=4)
    doc.close()
    
    final_doc = pymupdf.open(output_pdf_path)
    print(f"🎉 Successfully built publication-grade PDF: {output_pdf_path}")
    print(f"   Total Pages: {len(final_doc)}")
    print(f"   File Size: {os.path.getsize(output_pdf_path) / 1024:.2f} KB")

if __name__ == "__main__":
    generate_pdf()
