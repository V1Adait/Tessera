"""Extraction eval: fill every column for one filing with the REAL pipeline and score the cells.

    .venv/Scripts/python eval/extraction/run_eval.py eval/retrieval/corpus/aapl-20250927.html

Throwaway database schema and Qdrant collections (nothing of yours is touched). Real parser, real
chunker, real embeddings, real search, real extraction model. Each line of columns.jsonl is a
column plus the answer I checked in the document by hand.

Verdicts per column:
    CORRECT      answered, and the value equals the expected one
    WRONG        answered with a DIFFERENT value: the worst outcome (a confident wrong cell)
    MISSED       should have been answered, but the cell is not_found / low_confidence / error
    CORRECT_NF   should be not_found and is
    HALLUCINATED should be not_found but the cell is answered
    OTHER_NF     should be not_found; the cell is something else (low_confidence, error...)

Also audited, independently of the extractor: every stored quote is checked against the document's
own text by its stored character range. Every number printed comes from this run.
"""

import argparse
import asyncio
import importlib.util
import json
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.config import get_settings
from backend.app.db import repository as repo
from backend.app.matrix.models import CellRecord, ColumnDraft
from backend.app.matrix.service import build_matrix_service
from backend.app.tracing import db_recorder


def _load_retrieval_harness():
    """Reuse the retrieval eval's set-up code (throwaway schema, indexing, clean-up)."""
    spec = importlib.util.spec_from_file_location(
        "retrieval_run_eval", ROOT / "eval" / "retrieval" / "run_eval.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def meaning(value: object) -> str:
    """What a stored value SAYS, ignoring how the scale happened to be recorded: two answers with
    the same amount are the same answer even if one took 'millions' from the table and the other
    from its own wording."""
    if isinstance(value, dict):
        if "amount" in value:
            return f"{value['amount']} {value['currency']}"
        if "number" in value:
            return f"{value['number']} {value['unit']}"
        if "date" in value:
            return str(value["date"])
    return json.dumps(value, sort_keys=True)


def same_value(column: dict, cell: CellRecord) -> bool:
    value, expected = cell.value, column.get("value")
    kind = column["type"]
    if value is None or expected is None:
        return False
    if kind == "currency":
        return Decimal(value["amount"]) == Decimal(expected)
    if kind == "number":
        return Decimal(value["number"]) == Decimal(expected)
    if kind == "date":
        return value["date"] == expected
    if kind == "yes_no":
        return value is (expected == "true")
    if kind == "select":
        return value == expected
    return expected.casefold() in str(value).casefold()


def verdict(column: dict, cell: CellRecord) -> str:
    if column["expect"] == "answered":
        if cell.status == "answered":
            return "CORRECT" if same_value(column, cell) else "WRONG"
        return "MISSED"
    if cell.status == "not_found":
        return "CORRECT_NF"
    return "HALLUCINATED" if cell.status == "answered" else "OTHER_NF"


def audit_evidence(cells: list[CellRecord], canonical: str) -> tuple[int, int, int]:
    """(quotes checked exactly, quotes whose range is the whole chunk, quotes that FAIL).
    A quote with an exact range must equal the document text at that range."""
    exact = coarse = failed = 0
    for cell in cells:
        for e in cell.evidence:
            if e.exact_range:
                exact += 1
                failed += canonical[e.char_start : e.char_end] != e.quote
            else:
                coarse += 1
                failed += e.quote.replace(" ", "") not in canonical[
                    e.char_start : e.char_end
                ].replace(" ", "")
    return exact, coarse, failed


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("filing", type=Path)
    ap.add_argument("--columns", type=Path, default=HERE / "columns.jsonl")
    ap.add_argument("--entity", default="Apple Inc.")
    ap.add_argument("--show-quotes", action="store_true")
    ap.add_argument("--model", default=None, help="override settings.extract_model")
    ap.add_argument(
        "--repeat", type=int, default=1, help="fill every column this many times"
    )
    args = ap.parse_args()

    columns = [
        json.loads(l)
        for l in args.columns.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    retrieval = _load_retrieval_harness()
    base = get_settings()
    overrides = {
        "upload_dir": str(HERE / "results" / "uploads"),
        "index_on_startup": False,
    }
    if args.model:
        overrides["extract_model"] = args.model
    settings = base.model_copy(update=overrides)
    schema = f"evalr_{uuid4().hex[:10]}"
    admin = pool = client = indexer = run_prefix = None
    try:
        (
            admin,
            pool,
            ws,
            entity_id,
            client,
            _store,
            indexer,
            documents,
            run_prefix,
        ) = await retrieval.build(settings, args.filing, args.entity, schema)
        data = args.filing.read_bytes()
        pending = [data]

        async def read_once(_size: int) -> bytes:
            return pending.pop() if pending else b""

        up = await documents.upload(
            workspace_id=ws,
            entity_id=entity_id,
            filename=args.filing.name,
            read=read_once,
        )
        doc = await retrieval.wait_for(
            pool,
            up.document.id,
            "status",
            {"parsed", "needs_review", "needs_ocr", "failed"},
            900,
        )
        doc = await retrieval.wait_for(
            pool, doc.id, "index_status", {"indexed", "failed"}, 1800
        )
        print(
            f"document: {doc.status} / index {doc.index_status} | model {settings.extract_model} | {len(columns)} columns"
        )
        if doc.index_status != "indexed":
            raise SystemExit("indexing failed")

        service = build_matrix_service(
            pool, settings, indexer, recorder=db_recorder(pool)
        )
        if service is None:
            raise SystemExit("no key for the extraction model")

        created = {}
        for c in columns:
            created[c["id"]] = await service.create_column(
                ws,
                ColumnDraft(
                    name=c["name"],
                    prompt=c["prompt"],
                    type=c["type"],
                    allowed_values=c.get("allowed_values", []),
                ),
            )

        results: list[tuple[dict, CellRecord, float]] = []
        for c in columns:
            started = time.perf_counter()
            cell = await service.extract_cell(ws, entity_id, created[c["id"]].id)
            results.append((c, cell, time.perf_counter() - started))

        repeats: list[
            list[CellRecord]
        ] = []  # the same columns filled again, to measure stability
        for _ in range(args.repeat - 1):
            run = []
            for c in columns:
                run.append(
                    await service.extract_cell(ws, entity_id, created[c["id"]].id)
                )
            repeats.append(run)

        print(
            f"\n{'column':14s} {'expect':9s} {'verdict':13s} {'status':16s} value / reason"
        )
        for c, cell, secs in results:
            shown = (
                cell.display_value
                if cell.status in ("answered", "low_confidence")
                else (cell.status_detail or "")
            )
            print(
                f"{c['id']:14s} {c['expect']:9s} {verdict(c, cell):13s} {cell.status:16s} {str(shown)[:70]}  ({secs:.1f}s)"
            )
            if args.show_quotes:
                for e in cell.evidence:
                    print(f"{'':14s}   quote: {e.quote[:100]!r}")

        counts = Counter(verdict(c, cell) for c, cell, _ in results)
        positives = sum(c["expect"] == "answered" for c, _, _ in results)
        negatives = len(results) - positives
        print("\nverdicts:", dict(counts))
        print(
            f"answered-and-correct {counts['CORRECT']}/{positives} | wrong values {counts['WRONG']} | missed {counts['MISSED']}"
        )
        print(
            f"should be not_found: correct {counts['CORRECT_NF']}/{negatives} | hallucinated {counts['HALLUCINATED']} | other {counts['OTHER_NF']}"
        )

        if repeats:
            runs = [[cell for _, cell, _ in results], *repeats]
            letters = {
                "CORRECT": "C",
                "WRONG": "W",
                "MISSED": "m",
                "CORRECT_NF": "N",
                "HALLUCINATED": "H",
                "OTHER_NF": "o",
            }
            print(
                f"{chr(10)}stability over {len(runs)} runs (C correct, N correct not-found, m missed, W WRONG, H hallucinated, o other)"
            )
            same_answer = same_quote = 0
            for i, c in enumerate(columns):
                cells = [run[i] for run in runs]
                answers = {(x.status, meaning(x.value)) for x in cells}
                quotes = {tuple(e.quote for e in x.evidence) for x in cells}
                same_answer += len(answers) == 1
                same_quote += len(quotes) == 1
                marks = "".join(letters[verdict(c, x)] for x in cells)
                flag = (
                    ""
                    if len(answers) == 1 and len(quotes) == 1
                    else (
                        "  <- different answers"
                        if len(answers) > 1
                        else "  <- same answer, different quote"
                    )
                )
                print(f"  {c['id']:14s} {marks}{flag}")
            print(
                f"identical status+value in every run: {same_answer}/{len(columns)} | identical quotes too: {same_quote}/{len(columns)}"
            )
            per_run = [
                Counter(verdict(c, run[i]) for i, c in enumerate(columns))
                for run in runs
            ]
            print(
                "correct per run:",
                [r["CORRECT"] + r["CORRECT_NF"] for r in per_run],
                "| wrong values per run:",
                [r["WRONG"] + r["HALLUCINATED"] for r in per_run],
            )

        blocks = await repo.list_blocks(pool, doc.id, limit=500)
        offset = 0
        allb = list(blocks)
        while len(blocks) == 500:
            offset += 500
            blocks = await repo.list_blocks(pool, doc.id, limit=500, offset=offset)
            allb += blocks
        canonical = "\n\n".join(b.text for b in allb)
        exact, coarse, failed = audit_evidence(
            [cell for _, cell, _ in results], canonical
        )
        answered = [cell for _, cell, _ in results if cell.status == "answered"]
        print(
            f"\nevidence audit: {exact} quotes checked to the character, {coarse} placed by chunk, {failed} FAILED"
        )
        print(
            f"answered cells without evidence: {sum(not c.evidence for c in answered)} of {len(answered)}"
        )

        rows = await pool.fetch(
            "select tokens_in, tokens_out, status, cost_usd, cost_source from traces where kind = 'extract'"
        )
        spent = sum((r["cost_usd"] or 0) for r in rows)
        unknown = sum(r["cost_source"] == "unknown" for r in rows)
        print(
            f"cost of the extraction calls: ${float(spent):.5f} over {len(rows)} calls ({unknown} with no cost known)"
        )
        calls, errors = len(rows), sum(r["status"] != "ok" for r in rows)
        print(
            f"model calls: {calls} for {len(results) * args.repeat} cells ({calls - len(results) * args.repeat} retries) | errors {errors}"
        )
        print(
            f"tokens: in {sum(r['tokens_in'] or 0 for r in rows):,} | out {sum(r['tokens_out'] or 0 for r in rows):,} "
        )
        secs = sorted(s for _, _, s in results)
        print(
            f"time per cell: median {secs[len(secs) // 2]:.1f}s | max {secs[-1]:.1f}s"
        )

        out_dir = HERE / "results"
        out_dir.mkdir(exist_ok=True)
        out = out_dir / f"{datetime.now(tz=UTC):%Y%m%d-%H%M%S}.json"
        out.write_text(
            json.dumps(
                [
                    {
                        "column": c["id"],
                        "expect": c["expect"],
                        "verdict": verdict(c, cell),
                        "status": cell.status,
                        "display": cell.display_value,
                        "detail": cell.status_detail,
                        "quotes": [e.quote for e in cell.evidence],
                        "seconds": round(s, 2),
                    }
                    for c, cell, s in results
                ],
                indent=1,
            ),
            encoding="utf-8",
        )
        print(f"saved {out.relative_to(ROOT)}")
    finally:
        if indexer is not None:
            await indexer.close()
        if client is not None:
            await client.aclose()
        if pool is not None:
            await pool.close()
        if admin is not None:
            await admin.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
            await admin.close()
        if run_prefix is not None:
            await retrieval.cleanup_collections(settings, run_prefix)


if __name__ == "__main__":
    asyncio.run(main())
