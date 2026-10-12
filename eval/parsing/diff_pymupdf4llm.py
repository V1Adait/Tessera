"""Which cells does PyMuPDF4LLM miss or invent? Run:
    uv run --with pymupdf4llm --with pdfplumber python eval/parsing/diff_pymupdf4llm.py
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bakeoff import CORPUS, TRUTH, bag  # noqa: E402
from bakeoff_pymupdf4llm import markdown_tables  # noqa: E402

import pymupdf4llm  # noqa: E402

for stem in ("grid_simple", "borderless_financial", "rowspan"):
    md = pymupdf4llm.to_markdown(str(CORPUS / f"{stem}.pdf"), show_progress=False)
    got = bag(markdown_tables(md))
    want = bag(TRUTH[stem]["tables"])
    print(f"--- {stem}")
    print("  missing (in truth, not extracted):", sorted((want - got).elements())[:12])
    print("  extra   (extracted, not in truth):", sorted((got - want).elements())[:12])
    if stem == "grid_simple":
        print("  raw markdown, first 700 chars:\n" + md[:700])
