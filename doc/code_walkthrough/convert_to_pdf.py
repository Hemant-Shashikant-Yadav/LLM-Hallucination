"""
Batch Converter: Markdown (.md) to Styled PDF
Converts all Code Walkthrough markdown documentation files in doc/code_walkthrough/
into publication-quality PDF documents in the same location.

Designed specifically for xhtml2pdf / ReportLab rendering compatibility:
- Eliminates nested inline <code> background collision inside <pre>
- Eliminates code snippet severing across page boundaries via page-break-inside: avoid
- Eliminates line overflow via smart syntax-aware line wrapping
- Eliminates orphaned headings via -pdf-keep-with-next
- Shields regex/variable '$' characters in code from math parsers
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import markdown
from xhtml2pdf import pisa

# Publication-grade executive / academic PDF styling
PDF_CSS = """
@page {
    size: a4 portrait;
    margin-top: 1.5cm;
    margin-bottom: 1.8cm;
    margin-left: 1.5cm;
    margin-right: 1.5cm;
    
    @top-right {
        content: "Defense-in-Depth Framework | M.Tech Thesis Architecture & Defense";
        font-family: Helvetica, Arial, sans-serif;
        font-size: 7.5pt;
        color: #64748B;
    }
    
    @bottom-center {
        content: "Page " counter(page) " of " counter(pages);
        font-family: Helvetica, Arial, sans-serif;
        font-size: 8pt;
        color: #64748B;
    }
}

body {
    font-family: Helvetica, Arial, sans-serif;
    font-size: 9.0pt;
    line-height: 1.45;
    color: #1E293B;
}

h1 {
    color: #0F172A;
    font-size: 16pt;
    font-weight: bold;
    border-bottom: 2px solid #2563EB;
    padding-bottom: 4px;
    margin-top: 0;
    margin-bottom: 10px;
    -pdf-keep-with-next: true;
}

h2 {
    color: #1E3A8A;
    font-size: 12pt;
    font-weight: bold;
    border-bottom: 0.75px solid #CBD5E1;
    padding-bottom: 3px;
    margin-top: 14px;
    margin-bottom: 8px;
    -pdf-keep-with-next: true;
}

h3 {
    color: #1D4ED8;
    font-size: 10.5pt;
    font-weight: bold;
    margin-top: 11px;
    margin-bottom: 5px;
    -pdf-keep-with-next: true;
}

h4 {
    color: #334155;
    font-size: 9.0pt;
    font-weight: bold;
    margin-top: 8px;
    margin-bottom: 4px;
    -pdf-keep-with-next: true;
}

p {
    margin-top: 0;
    margin-bottom: 6px;
    text-align: justify;
}

ul, ol {
    margin-top: 0;
    margin-bottom: 6px;
    padding-left: 16px;
}

li {
    margin-bottom: 3px;
}

/* Inline code only */
code {
    font-family: Courier, monospace;
    font-size: 7.8pt;
    color: #1E3A8A;
    background-color: #F1F5F9;
    padding: 1px 3px;
    border: 0.5px solid #CBD5E1;
}

/* Code block container - guaranteed page-break safety and margin fit */
table.code-box {
    width: 100%;
    margin-top: 5px;
    margin-bottom: 8px;
    page-break-inside: avoid;
}

table.code-box td.code-cell {
    background-color: #F8FAFC;
    border: 0.75px solid #CBD5E1;
    border-left: 3.5px solid #2563EB;
    padding: 5px 8px;
    vertical-align: top;
}

pre.code-content {
    font-family: Courier, monospace;
    font-size: 6.8pt;
    line-height: 1.35;
    color: #0F172A;
    margin: 0;
    padding: 0;
    background-color: transparent;
}

/* Standard narrative tables */
table.normal-table {
    width: 100%;
    margin-top: 8px;
    margin-bottom: 10px;
    page-break-inside: avoid;
}

table.normal-table th {
    background-color: #1E293B;
    color: #FFFFFF;
    font-weight: bold;
    padding: 5px 7px;
    font-size: 7.8pt;
    text-align: left;
    border: 0.5px solid #334155;
}

table.normal-table td {
    border: 0.5px solid #CBD5E1;
    padding: 4px 7px;
    font-size: 7.8pt;
    vertical-align: top;
}

blockquote {
    border-left: 3px solid #2563EB;
    background-color: #F8FAFC;
    padding: 5px 10px;
    margin: 6px 0;
    color: #334155;
    font-style: italic;
}

hr {
    border: none;
    border-top: 1px solid #E2E8F0;
    margin: 12px 0;
}

.formula {
    background-color: #F8FAFC;
    border-left: 2px solid #64748B;
    padding: 4px 8px;
    font-family: Courier, monospace;
    font-size: 8.0pt;
    margin: 5px 0;
}
"""


BOX_DRAWING_MAP = {
    "┌": "+", "┐": "+", "└": "+", "┘": "+",
    "├": "+", "┤": "+", "┬": "+", "┴": "+", "┼": "+",
    "│": "|", "─": "-", "║": "|", "═": "=",
    "╭": "+", "╮": "+", "╰": "+", "╯": "+",
    "╱": "/", "╲": "\\",
    "▼": "v", "▲": "^", "►": ">", "◄": "<",
    "→": "->", "←": "<-", "⇒": "=>", "↓": "v", "↑": "^",
    "≥": ">=", "≤": "<=", "≠": "!=", "±": "+/-", "×": "x",
    "·": "*", "•": "*",
    "Σ": "SUM", "σ": "sigma", "α": "alpha", "β": "beta", "τ": "tau",
}


def normalize_code_characters(text: str) -> str:
    """Normalize box drawing and special unicode characters to ASCII for Courier compatibility."""
    for unicode_char, ascii_replacement in BOX_DRAWING_MAP.items():
        text = text.replace(unicode_char, ascii_replacement)
    return text


def wrap_code_snippet(code_text: str, max_chars: int = 90) -> str:
    """Wrap code lines exceeding max_chars cleanly with indent continuation."""
    code_text = normalize_code_characters(code_text)
    wrapped_lines = []
    for line in code_text.splitlines():
        if len(line) <= max_chars:
            wrapped_lines.append(line)
            continue
        prefix_match = re.match(r"^(\s*(?:\d+:)?\s*)", line)
        base_indent = prefix_match.group(1) if prefix_match else ""
        continuation_indent = base_indent + "    "
        remaining = line
        while len(remaining) > max_chars:
            split_idx = -1
            for sep in [", ", " ", " (", "(", " [", "["]:
                idx = remaining.rfind(sep, 0, max_chars)
                if idx > len(base_indent) + 12:
                    split_idx = idx + (1 if sep.startswith(" ") else len(sep))
                    break
            if split_idx == -1:
                split_idx = max_chars
            wrapped_lines.append(remaining[:split_idx])
            remaining = continuation_indent + remaining[split_idx:].lstrip()
        if remaining.strip():
            wrapped_lines.append(remaining)
    return "\n".join(wrapped_lines)


def process_markdown_to_html(raw_md: str) -> str:
    """Convert markdown to cleanly styled HTML with code block protection."""
    # 1. Protect code blocks from LaTeX math replacement and HTML distortion
    code_blocks: list[str] = []

    def save_code_block(match: re.Match[str]) -> str:
        idx = len(code_blocks)
        content = match.group(1).rstrip()
        wrapped = wrap_code_snippet(content, max_chars=90)
        # Escape HTML entities inside code
        wrapped_html = (
            wrapped.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )
        code_blocks.append(wrapped_html)
        return f"\n\n<div id=\"placeholder_code_{idx}\"></div>\n\n"

    # Regex captures fenced code blocks (```...```)
    text = re.sub(r"```(?:\w+)?\n(.*?)```", save_code_block, raw_md, flags=re.DOTALL)

    # 2. Convert LaTeX math constructs in narrative text
    text = re.sub(r"\$\$(.*?)\$\$", r'<div class="formula">\1</div>', text, flags=re.DOTALL)
    text = re.sub(r"\$(.*?)\$", r"<code>\1</code>", text)

    replacements = {
        r"\\to": "→",
        r"\\in": "∈",
        r"\\ge": "≥",
        r"\\le": "≤",
        r"\\times": "×",
        r"\\mathbb{R}": "ℝ",
        r"\\sum": "Σ",
        r"\\sigma": "σ",
        r"\\cdot": "·",
        r"\\land": "∧",
        r"\\lor": "∨",
        r"\\forall": "∀",
        r"\\exists": "∃",
        r"\\pm": "±",
        r"\\approx": "≈",
        r"\\neq": "≠",
        r"\\alpha": "α",
        r"\\beta": "β",
        r"\\tau": "τ",
    }
    for pat, rep in replacements.items():
        text = text.replace(pat, rep)

    # Clean local file links
    text = re.sub(r"\[([^\]]+)\]\(file:///[^\)]+\)", r"<b>\1</b>", text)

    # 3. Render markdown
    html_body = markdown.markdown(text, extensions=["tables", "sane_lists"])

    # 4. Add normal-table class to markdown tables
    html_body = html_body.replace("<table>", '<table class="normal-table">')

    # 5. Restore code blocks inside publication-quality tables
    for idx, snippet_html in enumerate(code_blocks):
        target = f'<div id="placeholder_code_{idx}"></div>'
        replacement = (
            f'<table class="code-box"><tr><td class="code-cell">'
            f'<pre class="code-content">{snippet_html}</pre>'
            f'</td></tr></table>'
        )
        html_body = html_body.replace(target, replacement)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
{PDF_CSS}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""


def convert_file(md_path: Path) -> Path:
    """Convert a single Markdown file to styled PDF."""
    raw_md = md_path.read_text(encoding="utf-8")
    full_html = process_markdown_to_html(raw_md)

    out_pdf = md_path.with_suffix(".pdf")
    with open(out_pdf, "wb") as f:
        status = pisa.CreatePDF(full_html, dest=f)

    if status.err:
        print(f"[-] Warning: PDF conversion had errors for {md_path.name}")
    else:
        size_kb = out_pdf.stat().st_size / 1024
        print(f"[+] Successfully converted {md_path.name} -> {out_pdf.name} ({size_kb:.1f} KB)")

    return out_pdf


def main() -> None:
    if len(sys.argv) > 1:
        targets = [Path(arg).resolve() for arg in sys.argv[1:]]
        md_files = []
        for target in targets:
            if target.is_file() and target.suffix == ".md":
                md_files.append(target)
            elif target.is_dir():
                md_files.extend(sorted(target.glob("*.md")))
    else:
        doc_dir = Path(__file__).resolve().parent
        md_files = sorted(doc_dir.glob("*.md"))

    if not md_files:
        print("No markdown files found to convert.")
        sys.exit(1)

    print(f"Converting {len(md_files)} markdown files to PDF...\n")
    for md_path in md_files:
        convert_file(md_path)

    print("\nAll target PDF documents successfully generated.")


if __name__ == "__main__":
    main()
