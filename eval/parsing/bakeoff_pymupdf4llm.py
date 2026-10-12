"""Score PyMuPDF4LLM (PDF -> Markdown for LLMs) on the same corpus, with the same bag-of-cells scoring.

Run (isolated environment, nothing is added to the project):
    uv run --with pymupdf4llm --with pdfplumber python eval/parsing/bakeoff_pymupdf4llm.py

Also prints licence metadata of the candidate packages, read from the installed packages.
"""

import re
import sys
import time
from importlib import metadata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bakeoff import CORPUS, TRUTH, bag, norm, score  # noqa: E402

import pymupdf4llm  # noqa: E402


def markdown_tables(md: str) -> list[list[list[str]]]:
    """Pull pipe tables out of Markdown: consecutive lines starting with '|'."""
    tables, current = [], []
    for line in md.splitlines() + [""]:
        if line.strip().startswith("|"):
            cells = [norm(c.replace("<br>", " ").replace("**", "")) for c in line.strip().strip("|").split("|")]  # bold markers are formatting, not content
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                continue  # the |---|---| separator row
            current.append(cells)
        elif current:
            tables.append(current)
            current = []
    return tables


def licence(pkg: str) -> str:
    try:
        m = metadata.metadata(pkg)
    except metadata.PackageNotFoundError:
        return "not installed"
    expr = m.get("License-Expression") or ""
    lic = (m.get("License") or "").strip().splitlines()[0][:70] if m.get("License") else ""
    classifiers = [c.split("::")[-1].strip() for c in m.get_all("Classifier") or [] if "License" in c]
    return f"{pkg} {m['Version']}: expression={expr or '-'} | license={lic or '-'} | classifiers={classifiers or '-'}"


def main() -> None:
    print("=== PYMUPDF4LLM: table cell content f1 (found=tables returned) ===")
    scores: dict[str, float] = {}
    for path in sorted(CORPUS.glob("*.pdf")):
        if path.stem.startswith("chart_"):
            continue
        started = time.perf_counter()
        try:
            md = pymupdf4llm.to_markdown(str(path), show_progress=False)
        except Exception as exc:  # noqa: BLE001
            print(f"  {path.stem:36s} ERROR {type(exc).__name__}: {str(exc)[:90]}")
            continue
        seconds = time.perf_counter() - started
        tables = markdown_tables(md)
        truth = TRUTH[path.stem]
        _, _, f1 = score(bag(tables, drop_repeated_header=path.stem == "multipage"), bag(truth["tables"]))
        if path.stem == "text_only":
            print(f"  {path.stem:36s} {'clean' if not tables else str(len(tables)) + ' FALSE'} ({seconds:.2f}s)")
        else:
            scores[path.stem] = f1
            print(f"  {path.stem:36s} f1={f1:.2f} (found {len(tables)}) ({seconds:.2f}s)")
    born = [v for k, v in scores.items() if not k.startswith("scanned_")]
    scans = [v for k, v in scores.items() if k.startswith("scanned_")]
    print(f"\nborn-digital mean f1: {sum(born) / len(born):.2f} (n={len(born)})")
    print(f"scanned mean f1:      {sum(scans) / len(scans):.2f} (n={len(scans)})")

    print("\n=== LICENCE METADATA (as declared by the packages; not legal advice) ===")
    for pkg in ("pymupdf4llm", "pymupdf", "pymupdf-layout", "pdfplumber", "pdfminer.six", "pypdfium2", "docling"):
        print("  " + licence(pkg))
    print("\nrequires of pymupdf4llm:", metadata.requires("pymupdf4llm"))


if __name__ == "__main__":
    main()
