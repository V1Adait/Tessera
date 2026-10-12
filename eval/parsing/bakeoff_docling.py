"""Score Docling (local layout + table-structure models, OCR for scans) on the same corpus.

Run (isolated environment, nothing is added to the project; downloads models on first use):
    uv run --with docling python eval/parsing/bakeoff_docling.py

Uses the same bag-of-cells scoring as bakeoff.py, so the numbers are directly comparable.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bakeoff import CORPUS, TRUTH, bag, norm, score  # noqa: E402

from docling.datamodel.base_models import InputFormat  # noqa: E402
from docling.datamodel.pipeline_options import PdfPipelineOptions  # noqa: E402
from docling.document_converter import DocumentConverter, PdfFormatOption  # noqa: E402

options = PdfPipelineOptions()
options.do_ocr = True  # needed for scanned pages
options.do_table_structure = True
options.do_picture_classification = True  # labels figures: bar_chart, line_chart, ...
converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
)


def extract(path: Path):
    result = converter.convert(str(path))
    doc = result.document
    tables = []
    for table in doc.tables:
        cells = [
            [norm(c.text) for c in table.data.table_cells]
        ]  # one flat "row" is enough for a bag
        tables.append(cells)
    pictures = []
    for pic in doc.pictures:
        classes = []
        if (
            pic.meta is not None
            and getattr(pic.meta, "classification", None) is not None
        ):
            classes = [
                (p.class_name, round(p.confidence, 2))
                for p in pic.meta.classification.predictions[:2]
            ]
        pictures.append(classes)
    return (
        tables,
        pictures,
        doc.export_to_text() if hasattr(doc, "export_to_text") else "",
    )


def main() -> None:
    print("=== DOCLING: table cell content f1 (found=tables returned) ===")
    scores = {}
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    for path in sorted(CORPUS.glob("*.pdf")):
        if only not in path.stem:
            continue
        truth = TRUTH[path.stem]
        started = time.perf_counter()
        try:
            tables, pictures, text = extract(path)
        except Exception as exc:  # noqa: BLE001
            print(f"  {path.stem:30s} ERROR {type(exc).__name__}: {str(exc)[:100]}")
            continue
        seconds = time.perf_counter() - started
        if path.stem.startswith("chart_"):
            labels = truth["labels"]
            got = [lab for lab in labels if lab in text]
            print(
                f"  {path.stem:30s} pictures={pictures} labels_in_text={len(got)}/{len(labels)} ({seconds:.1f}s)"
            )
            continue
        _, _, f1 = score(
            bag(tables, drop_repeated_header=path.stem == "multipage"),
            bag(truth["tables"]),
        )
        if path.stem == "text_only":
            print(
                f"  {path.stem:30s} {'clean' if not tables else str(len(tables)) + ' FALSE'} ({seconds:.1f}s)"
            )
        else:
            scores[path.stem] = f1
            print(
                f"  {path.stem:30s} f1={f1:.2f} (found {len(tables)}) ({seconds:.1f}s)"
            )
    born = [v for k, v in scores.items() if not k.startswith("scanned_")]
    scanned = [v for k, v in scores.items() if k.startswith("scanned_")]
    if born:
        print(f"\nborn-digital mean f1: {sum(born) / len(born):.2f}")
    for grade in ("mild", "harsh"):
        vals = [v for k, v in scores.items() if k.endswith(f"_{grade}")]
        if vals:
            print(
                f"scanned {grade:5s} mean f1: {sum(vals) / len(vals):.2f}  (n={len(vals)}, min {min(vals):.2f})"
            )
    clean = [
        v
        for k, v in scores.items()
        if k.startswith("scanned_") and not k.endswith(("_mild", "_harsh"))
    ]
    if clean:
        print(f"scanned clean mean f1: {sum(clean) / len(clean):.2f}  (n={len(clean)})")
    if not only:
        json.dump(
            scores, open(Path(__file__).parent / "docling_scores.json", "w"), indent=1
        )


if __name__ == "__main__":
    main()
