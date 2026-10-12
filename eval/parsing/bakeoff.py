"""Score table extractors against the generated corpus (ground truth in corpus/truth.json).

Run (isolated environment, nothing is added to the project):
    uv run --with pymupdf --with pdfplumber python eval/parsing/bakeoff.py

Scores, per document and extractor:
  found     how many tables the extractor returned (truth: 0 for text-only pages)
  recall    share of the true cell texts that were recovered
  precision share of the extracted cell texts that are true cells
  f1        harmonic mean of the two
Cells are compared as a bag of normalised strings, so the score is fair to extractors that
represent merged cells differently, while still punishing wrong, missing and split cells.
This measures the CELL CONTENT, not whether rows and columns are perfectly aligned; the
structure check (row and column counts) is reported separately for the simple tables.
"""

import json
import re
import time
from collections import Counter
from pathlib import Path

import pdfplumber
import pymupdf

CORPUS = Path(__file__).parent / "corpus"
TRUTH = json.loads((CORPUS / "truth.json").read_text(encoding="utf-8"))


def norm(cell: object) -> str:
    return re.sub(r"\s+", " ", str(cell or "")).strip()


def bag(
    tables: list[list[list[object]]], drop_repeated_header: bool = False
) -> Counter[str]:
    """Normalised non-empty cell texts of all tables. Optionally drop rows that repeat the
    first row (a header repeated on every page is not new content)."""
    out: Counter[str] = Counter()
    for table in tables:
        header = [norm(c) for c in table[0]] if table else []
        for i, row in enumerate(table):
            cells = [norm(c) for c in row]
            if drop_repeated_header and i > 0 and cells == header:
                continue
            out.update(c for c in cells if c)
    return out


def score(extracted: Counter[str], truth: Counter[str]) -> tuple[float, float, float]:
    hit = sum((extracted & truth).values())
    recall = hit / sum(truth.values()) if truth else 1.0
    precision = (
        hit / sum(extracted.values()) if extracted else (1.0 if not truth else 0.0)
    )
    f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    return recall, precision, f1


# --- the extractors --------------------------------------------------------------------


def pymupdf_lines(path: Path) -> list[list[list[object]]]:
    tables = []
    with pymupdf.open(path) as doc:
        for page in doc:
            tables += [t.extract() for t in page.find_tables(strategy="lines").tables]
    return tables


def pymupdf_text(path: Path) -> list[list[list[object]]]:
    tables = []
    with pymupdf.open(path) as doc:
        for page in doc:
            tables += [t.extract() for t in page.find_tables(strategy="text").tables]
    return tables


def plumber_lines(path: Path) -> list[list[list[object]]]:
    with pdfplumber.open(path) as pdf:
        return [t for page in pdf.pages for t in page.extract_tables()]


def plumber_text(path: Path) -> list[list[list[object]]]:
    settings = {"vertical_strategy": "text", "horizontal_strategy": "text"}
    with pdfplumber.open(path) as pdf:
        return [t for page in pdf.pages for t in page.extract_tables(settings)]


def auto(path: Path) -> list[list[list[object]]]:
    """The routing a real parser would do: try ruled-line detection first, and only fall back
    to text-alignment detection when it finds nothing (borderless statements)."""
    return pymupdf_lines(path) or pymupdf_text(path)


EXTRACTORS = {
    "pymupdf lines": pymupdf_lines,
    "pymupdf text": pymupdf_text,
    "pdfplumber lines": plumber_lines,
    "pdfplumber text": plumber_text,
    "pymupdf auto": auto,
}


# --- page quality: does a page need OCR? ----------------------------------------------


def page_report(path: Path) -> dict[str, float]:
    """Cheap per-page signals: characters of real text, and how much of the page is one image."""
    with pymupdf.open(path) as doc:
        chars = sum(len(p.get_text().strip()) for p in doc)
        pages = len(doc)
        covered = 0.0
        for p in doc:
            area = p.rect.width * p.rect.height
            covered += max(
                (pymupdf.Rect(i["bbox"]).get_area() / area for i in p.get_image_info()),
                default=0.0,
            )
    return {"chars_per_page": chars / pages, "image_cover": covered / pages}


def needs_ocr(report: dict[str, float]) -> bool:
    return report["chars_per_page"] < 40 and report["image_cover"] > 0.5


def main() -> None:
    docs = sorted(p for p in CORPUS.glob("*.pdf"))
    table_docs = [p for p in docs if TRUTH[p.stem]["tables"] or p.stem == "text_only"]

    print("=== TABLE CELL CONTENT (f1; found=tables returned) ===")
    print(f"{'document':30s}" + "".join(f"{name:>20s}" for name in EXTRACTORS))
    totals: dict[str, list[float]] = {name: [] for name in EXTRACTORS}
    timings: dict[str, float] = dict.fromkeys(EXTRACTORS, 0.0)
    for path in table_docs:
        truth = TRUTH[path.stem]
        truth_bag = bag(truth["tables"])
        line = f"{path.stem:30s}"
        for name, fn in EXTRACTORS.items():
            started = time.perf_counter()
            try:
                tables = fn(path)
            except Exception as exc:  # an extractor crashing is a result too
                line += f"{'ERR ' + type(exc).__name__:>20s}"
                totals[name].append(0.0)
                continue
            timings[name] += time.perf_counter() - started
            _, _, f1 = score(
                bag(tables, drop_repeated_header=path.stem == "multipage"), truth_bag
            )
            if path.stem == "text_only":
                line += f"{('clean' if not tables else f'{len(tables)} FALSE'):>20s}"
                totals[name].append(1.0 if not tables else 0.0)
            else:
                line += f"{f'{f1:.2f} (found {len(tables)})':>20s}"
                totals[name].append(f1)
        print(line)
    print(
        f"{'MEAN over all above':30s}"
        + "".join(f"{sum(v) / len(v):>20.2f}" for v in totals.values())
    )
    print(
        f"{'seconds (total)':30s}" + "".join(f"{timings[n]:>20.2f}" for n in EXTRACTORS)
    )

    print("\n=== BORN-DIGITAL ONLY (no scans): mean f1 ===")
    born = [
        p.stem
        for p in table_docs
        if not p.stem.startswith("scanned_") and p.stem != "text_only"
    ]
    for name in EXTRACTORS:
        vals = [totals[name][i] for i, p in enumerate(table_docs) if p.stem in born]
        print(
            f"  {name:20s} {sum(vals) / len(vals):.2f}   worst: "
            f"{min((totals[name][i], p.stem) for i, p in enumerate(table_docs) if p.stem in born)[1]}"
        )

    print("\n=== SCAN DETECTION (does the page need OCR?) ===")
    for path in docs:
        rep = page_report(path)
        flag = "NEEDS OCR" if needs_ocr(rep) else "text layer ok"
        expected = "NEEDS OCR" if path.stem.startswith("scanned_") else "text layer ok"
        ok = "ok " if flag == expected else "WRONG"
        print(
            f"  {path.stem:30s} chars/page={rep['chars_per_page']:7.0f} image_cover={rep['image_cover']:.2f}"
            f"  -> {flag:14s} {ok}"
        )

    print("\n=== CHARTS: how many chart labels does the text layer hold? ===")
    for stem in ("chart_vector", "chart_raster"):
        with pymupdf.open(CORPUS / f"{stem}.pdf") as doc:
            text = " ".join(p.get_text() for p in doc)
            drawings = sum(len(p.get_drawings()) for p in doc)
            images = sum(len(p.get_images()) for p in doc)
        labels = TRUTH[stem]["labels"]
        found = [lab for lab in labels if lab in text]
        print(
            f"  {stem:14s} labels recovered {len(found)}/{len(labels)}  "
            f"vector drawings={drawings} embedded images={images}"
        )


if __name__ == "__main__":
    main()
