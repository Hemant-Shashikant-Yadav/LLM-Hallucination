"""
Convert a Markdown file to a PDF.

Usage:
    python md_to_pdf.py notes.md                  # writes notes.pdf next to notes.md
    python md_to_pdf.py notes.md -o out/notes.pdf
    python md_to_pdf.py notes.md --keep-html      # also keeps notes.html (handy for checking the layout)

How it works: the Markdown is turned into a styled HTML page (tables, code blocks, quotes,
images), and Microsoft Edge or Google Chrome prints that page to PDF in the background. A real
browser renders tables, symbols such as Σ λ ≈ κ and non-English text exactly as on screen.

Needs:
- the `markdown` package:  pip install markdown
- Microsoft Edge (built into Windows) or Google Chrome / Chromium. If it is installed somewhere
  unusual, pass its path with --browser.
"""

from __future__ import annotations

import argparse
import html
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

CSS = """
@page { size: A4; margin: 18mm 16mm 20mm 16mm;
        @bottom-center { content: counter(page) " / " counter(pages); font: 9pt "Segoe UI", Arial, sans-serif;
                         color: #64748b; } }
html { font-size: 10.5pt; }
body { font-family: "Segoe UI", Calibri, Arial, "Helvetica Neue", sans-serif; color: #0f172a; line-height: 1.5;
       margin: 0; background: #ffffff; }
h1 { font-size: 22pt; color: #3657c8; margin: 0 0 10pt; padding-bottom: 6pt; border-bottom: 2pt solid #3657c8; }
h2 { font-size: 16pt; color: #3657c8; margin: 22pt 0 8pt; padding-bottom: 3pt; border-bottom: 1pt solid #cbd5e1;
     break-after: avoid; }
h3 { font-size: 13pt; color: #1f2a44; margin: 16pt 0 6pt; break-after: avoid; }
h4 { font-size: 11.5pt; color: #4f46e5; margin: 14pt 0 5pt; break-after: avoid; }
p, li { margin: 4pt 0; }
ul, ol { padding-left: 18pt; }
a { color: #3657c8; text-decoration: none; }
code { font-family: Consolas, "Cascadia Mono", Menlo, "DejaVu Sans Mono", monospace; font-size: 9.5pt;
       background: #eef2ff; padding: 0 3pt; border-radius: 3pt; }
pre { background: #f8fafc; border: 1pt solid #c7d2fe; border-left: 4pt solid #4f46e5; border-radius: 4pt;
      padding: 7pt 10pt; overflow-x: auto; break-inside: avoid; white-space: pre-wrap; }
pre code { background: none; padding: 0; font-size: 9.5pt; line-height: 1.45; }
table { border-collapse: collapse; width: 100%; margin: 7pt 0 10pt; font-size: 9.5pt; break-inside: avoid; }
th { background: #3657c8; color: #ffffff; text-align: left; padding: 4pt 6pt; }
td { border-bottom: 1pt solid #e2e8f0; padding: 3.5pt 6pt; vertical-align: top; }
tr:nth-child(even) td { background: #f5f7ff; }
blockquote { margin: 8pt 0; padding: 6pt 12pt; background: #eef2ff; border-left: 4pt solid #3657c8;
             border-radius: 4pt; break-inside: avoid; }
blockquote p { margin: 3pt 0; }
hr { border: none; border-top: 1pt solid #cbd5e1; margin: 16pt 0; }
img { max-width: 100%; }
"""


def find_browser(explicit: str | None) -> str | None:
    """Path of Edge, Chrome or Chromium, or None."""
    if explicit:
        return explicit if Path(explicit).exists() or shutil.which(explicit) else None
    system = platform.system()
    candidates: list[str] = []
    if system == "Windows":
        roots = [os.environ.get(k, "") for k in ("PROGRAMFILES(X86)", "PROGRAMFILES", "LOCALAPPDATA")]
        for root in filter(None, roots):
            candidates += [os.path.join(root, "Microsoft", "Edge", "Application", "msedge.exe"),
                           os.path.join(root, "Google", "Chrome", "Application", "chrome.exe")]
    elif system == "Darwin":
        candidates += ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                       "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
                       "/Applications/Chromium.app/Contents/MacOS/Chromium"]
    for c in candidates:
        if Path(c).exists():
            return c
    for name in ("msedge", "microsoft-edge", "google-chrome", "google-chrome-stable", "chromium", "chromium-browser",
                 "chrome"):
        found = shutil.which(name)
        if found:
            return found
    return None


def markdown_to_html(md_text: str, title: str, base_dir: Path) -> str:
    try:
        import markdown
    except ImportError:
        sys.exit("The 'markdown' package is missing. Install it with:  pip install markdown")
    body = markdown.markdown(md_text, extensions=["extra", "sane_lists", "toc"], output_format="html5")
    base = base_dir.resolve().as_uri() + "/"
    return (f'<!doctype html>\n<html lang="en"><head><meta charset="utf-8">\n<base href="{html.escape(base)}">\n'
            f"<title>{html.escape(title)}</title>\n<style>{CSS}</style></head>\n<body>\n{body}\n</body></html>\n")


def _wait_for_file(path: Path, timeout: float) -> bool:
    """True once `path` exists and its size has stopped changing (the browser finished writing)."""
    deadline = time.monotonic() + timeout
    last = -1
    while time.monotonic() < deadline:
        size = path.stat().st_size if path.exists() else -1
        if size > 0 and size == last:
            return True
        last = size
        time.sleep(0.5)
    return False


def print_to_pdf(browser: str, html_path: Path, pdf_path: Path) -> None:
    if pdf_path.exists():
        pdf_path.unlink()  # so an old PDF is never mistaken for the new one
    profile = tempfile.mkdtemp(prefix="md2pdf_profile_")  # a fresh profile: an open browser window does not interfere
    try:
        cmd = [browser, "--headless=new", "--disable-gpu", "--no-first-run", "--disable-extensions",
               f"--user-data-dir={profile}",
               "--no-pdf-header-footer", "--print-to-pdf-no-header",  # no date/URL lines (new and old flag)
               f"--print-to-pdf={pdf_path}", html_path.resolve().as_uri()]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired:
            sys.exit("The browser took longer than 3 minutes to print the PDF.")
        # On Windows, msedge.exe can hand the job to another process and return before the PDF is written
        if not _wait_for_file(pdf_path, timeout=120):
            sys.exit(f"The browser did not write a PDF (exit code {result.returncode}).\n"
                     f"{result.stderr.strip()[-2000:]}")
    finally:
        shutil.rmtree(profile, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert a Markdown file to PDF (via Edge or Chrome).")
    ap.add_argument("input", help="the .md file to convert")
    ap.add_argument("-o", "--output", help="the PDF to write (default: next to the input, with .pdf)")
    ap.add_argument("--title", help="document title (default: the file name)")
    ap.add_argument("--keep-html", action="store_true", help="also keep the HTML page next to the PDF")
    ap.add_argument("--browser", help="path to msedge, chrome or chromium, if it is not found automatically")
    args = ap.parse_args()

    src = Path(args.input)
    if not src.is_file():
        sys.exit(f"File not found: {src}")
    out = Path(args.output) if args.output else src.with_suffix(".pdf")
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    browser = find_browser(args.browser)
    if not browser:
        sys.exit("No Edge, Chrome or Chromium found. Install one, or pass its path with --browser.")

    page = markdown_to_html(src.read_text(encoding="utf-8"), args.title or src.stem.replace("_", " "), src.parent)
    if args.keep_html:
        html_path = out.with_suffix(".html")
        html_path.write_text(page, encoding="utf-8")
        print_to_pdf(browser, html_path, out)
    else:
        tmp = tempfile.mkdtemp(prefix="md2pdf_")
        try:
            html_path = Path(tmp) / "page.html"
            html_path.write_text(page, encoding="utf-8")
            print_to_pdf(browser, html_path, out)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
