"""Finance eval: the REAL pipeline on real 10-Ks, scored against the numbers the companies filed.

    .venv/Scripts/python eval/finance/fetch.py        # once: filings + XBRL gold labels
    .venv/Scripts/python eval/finance/run_eval.py [--only AAPL,MSFT]

For each filing: a throwaway Postgres schema and Qdrant collections (nothing of yours is touched),
the real parser, chunker, embeddings, retrieval, extraction model, verification, computed cells and
tie-outs, using the columns of the shipped credit/PE pack. Each answered cell is compared with the
XBRL value the company itself filed. Computed cells (margins, free cash flow) are compared with the
value computed from the GOLD inputs.

Verdicts for a cell with a gold value:
    CORRECT   the same amount (scale-normalised, exact)
    ROUNDED   within 0.5% (a rounded figure in the text): reported separately, not counted correct
    SIGN      the right size with the opposite sign
    WRONG     answered with a different value: the worst outcome, a confident wrong cell
    MISSED    not answered (not_found, low_confidence, error)
A metric the company's XBRL has no value for is UNSCORED. Every number in the report comes from
this run; nothing is typed in by hand.
"""

import argparse
import asyncio
import importlib.util
import json
import sys
import time
import traceback
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
from backend.app.matrix.models import CellRecord
from backend.app.matrix.service import build_matrix_service
from backend.app.packs.loader import get_packs
from backend.app.planning.plan import _from_pack
from backend.app.tracing import db_recorder

PACK_ID = "credit_pe_diligence"

# gold metric id -> the pack column's name
BASE = {
    "revenue": "Revenue",
    "cost_of_revenue": "Cost of revenue",
    "gross_profit": "Gross profit",
    "operating_income": "Operating income",
    "net_income": "Net income",
    "cash": "Cash and equivalents",
    "operating_cash_flow": "Operating cash flow",
    "capex": "Capital expenditures",
    "interest_expense": "Interest expense",
}
# computed column name -> (op, a, b) over gold metric ids; the expected value comes from gold inputs
# Only the pack columns that ARE computed in code. "Free cash flow" and "Gross margin" are extracted
# columns in the pack (companies state or define them themselves), and XBRL has no gold for them, so
# they are not scored; "Free cash flow margin" reads "Free cash flow" and is left out for that reason.
COMPUTED = {
    "Operating margin": ("percent", "operating_income", "revenue"),
    "Net margin": ("percent", "net_income", "revenue"),
}
PERCENT_TOLERANCE = Decimal("0.1")  # percentage points


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expected_computed(op: str, a: str | None, b: str, gold: dict[str, Decimal]):
    try:
        if op == "difference":
            assert a is not None
            return gold[a] - gold[b]
        if op == "percent":
            assert a is not None
            return gold[a] / gold[b] * 100
        if op == "fcf_percent":
            return (gold["operating_cash_flow"] - gold["capex"]) / gold[b] * 100
    except (KeyError, ArithmeticError):
        return None
    return None


def describe(ticker: str, metric: str, kind: str, cell: CellRecord | None) -> dict:
    """One report row. A computed column has no cell at all while its inputs are unanswered."""
    if cell is None:
        return {"ticker": ticker, "metric": metric, "kind": kind, "status": "no_cell",
                "period": None, "got": None,
                "detail": "no cell was created (its inputs were not answered)", "quotes": []}
    got = amount_of(cell)
    return {"ticker": ticker, "metric": metric, "kind": kind, "status": cell.status,
            "period": cell.period, "got": str(got) if got is not None else None,
            "detail": cell.status_detail, "quotes": [e.quote for e in cell.evidence]}


def amount_of(cell: CellRecord | None) -> Decimal | None:
    if cell is None:
        return None
    value = cell.value
    if cell.status != "answered" or not isinstance(value, dict):
        return None
    if "amount" in value:
        return Decimal(str(value["amount"]))
    if "number" in value:
        return Decimal(str(value["number"]))
    return None


def verdict(expected: Decimal, cell: CellRecord | None, *, percent: bool) -> str:
    got = amount_of(cell)
    if got is None:
        return "MISSED"
    if percent:
        return "CORRECT" if abs(got - expected) <= PERCENT_TOLERANCE else "WRONG"
    if got == expected:
        return "CORRECT"
    if expected != 0 and got == -expected:
        return "SIGN"
    if expected != 0 and abs(got - expected) / abs(expected) <= Decimal("0.005"):
        return "ROUNDED"
    return "WRONG"


async def eval_filing(label: dict, settings, retrieval, extraction) -> dict:
    filing = HERE / "corpus" / label["file"]
    gold = {k: Decimal(v["value"]) for k, v in label["values"].items()}
    pack = get_packs()[PACK_ID]
    by_name = {c.name: c for c in pack.columns}
    schema = f"evalr_{uuid4().hex[:10]}"
    admin = pool = client = indexer = run_prefix = None
    try:
        started = time.perf_counter()
        (admin, pool, ws, entity_id, client, _s, indexer, documents, run_prefix) = (
            await retrieval.build(settings, filing, label["name"], schema)
        )
        pending = [filing.read_bytes()]

        async def read_once(_size: int) -> bytes:
            return pending.pop() if pending else b""

        up = await documents.upload(
            workspace_id=ws, entity_id=entity_id, filename=filing.name, read=read_once
        )
        doc = await retrieval.wait_for(
            pool, up.document.id, "status", {"parsed", "needs_review", "needs_ocr", "failed"}, 1200
        )
        doc = await retrieval.wait_for(
            pool, doc.id, "index_status", {"indexed", "failed"}, 2400
        )
        if doc.index_status != "indexed":
            return {"ticker": label["ticker"], "error": f"indexing {doc.index_status}"}
        ingest_s = time.perf_counter() - started

        service = build_matrix_service(pool, settings, indexer, recorder=db_recorder(pool))
        if service is None:
            raise SystemExit("no key for the extraction model")

        ids = {}
        for name in [*BASE.values(), *COMPUTED]:
            planned = _from_pack(by_name[name], pack)
            depends = [ids[n.casefold()] for n in planned.inputs]
            ids[name.casefold()] = (await service.create_column(ws, planned.draft(depends))).id

        fill_started = time.perf_counter()
        for name in BASE.values():
            await service.extract_cell(ws, entity_id, ids[name.casefold()])
        await service.refresh_row(ws, entity_id)
        fill_s = time.perf_counter() - fill_started

        cells = {
            c.column_id: c for c in await service.list_cells(ws, entity_id=entity_id)
        }
        rows = []
        for metric, name in BASE.items():
            cell = cells.get(ids[name.casefold()])
            row = describe(label["ticker"], name, "extracted", cell)
            if metric in gold:
                row["expected"] = str(gold[metric])
                row["verdict"] = verdict(gold[metric], cell, percent=False)
            else:
                row["verdict"] = "UNSCORED"
            rows.append(row)
        for name, (op, a, b) in COMPUTED.items():
            cell = cells.get(ids[name.casefold()])
            want = expected_computed(op, a, b, gold)
            percent = op != "difference"
            row = describe(label["ticker"], name, "computed", cell)
            if want is None:
                row["verdict"] = "UNSCORED"
            else:
                row["expected"] = str(want)
                row["verdict"] = verdict(want, cell, percent=percent)
            rows.append(row)

        # The fiscal year is named by the year the period ENDS: right for all six companies here (the
        # January year-ends are "fiscal 2026" in their own filings). XBRL's own fy tag is not used: it
        # calls Salesforce's year ending January 2026 "2025".
        expected_period = f"FY{label['period_end'][:4]}"
        column_name = {column_id: name for name, column_id in ids.items()}
        checks = [
            {"ticker": label["ticker"], "name": chk.get("name", "?"),
             "cell": column_name.get(c.column_id, "?"),
             "status": chk.get("status"), "detail": chk.get("detail", "")}
            for c in cells.values() for chk in c.checks
        ]

        blocks, offset = [], 0
        while True:
            page = await repo.list_blocks(pool, doc.id, limit=500, offset=offset)
            blocks += page
            if len(page) < 500:
                break
            offset += 500
        canonical = "\n\n".join(b.text for b in blocks)
        exact, coarse, failed = extraction.audit_evidence(
            [c for c in cells.values() if c.evidence], canonical
        )
        answered = [c for c in cells.values() if c.status == "answered"]
        traces = await pool.fetch(
            "select tokens_in, tokens_out, status, cost_usd, cost_source from traces where kind = 'extract'"
        )
        return {
            "ticker": label["ticker"], "rows": rows, "checks": checks,
            "expected_period": expected_period,
            "evidence": {"exact": exact, "coarse": coarse, "failed": failed,
                         "answered": len(answered),
                         "without_evidence": sum(not c.evidence for c in answered)},
            "cost_usd": float(sum((r["cost_usd"] or 0) for r in traces)),
            "calls": len(traces), "call_errors": sum(r["status"] != "ok" for r in traces),
            "tokens_in": sum(r["tokens_in"] or 0 for r in traces),
            "tokens_out": sum(r["tokens_out"] or 0 for r in traces),
            "ingest_s": round(ingest_s, 1), "fill_s": round(fill_s, 1),
        }
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


def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n / d:.0f}%)" if d else "-"


def report(results: list[dict], model: str) -> str:
    ok = [r for r in results if "rows" in r]
    rows = [row for r in ok for row in r["rows"]]
    scored = [r for r in rows if r["verdict"] != "UNSCORED"]
    out = [
        "# Finance eval report",
        "",
        f"Generated {datetime.now(tz=UTC):%Y-%m-%d %H:%M} UTC by `eval/finance/run_eval.py`. "
        f"Extraction model `{model}`. Gold values are the XBRL numbers each company filed with its 10-K. "
        "Every figure below was produced by that run.",
        "",
        f"Filings run: {len(ok)} of {len(results)}"
        + "".join(f" | {r['ticker']}: {r['error']}" for r in results if "error" in r),
        "",
    ]

    def table(title: str, group, keys):
        out.extend([f"## {title}", "", "| " + " | ".join(["", "scored", "correct", "rounded", "sign", "wrong", "missed"]) + " |",
                    "|---|---|---|---|---|---|---|"])
        for key in keys:
            sel = [r for r in scored if group(r) == key]
            c = Counter(r["verdict"] for r in sel)
            out.append(f"| {key} | {len(sel)} | {c['CORRECT']} | {c['ROUNDED']} | {c['SIGN']} | {c['WRONG']} | {c['MISSED']} |")
        out.append("")

    for kind, title in (("extracted", "Extracted values"), ("computed", "Computed values (from gold inputs)")):
        sel = [r for r in scored if r["kind"] == kind]
        c = Counter(r["verdict"] for r in sel)
        out.append(f"**{title}:** {pct(c['CORRECT'], len(sel))} correct | {c['ROUNDED']} rounded | "
                   f"{c['SIGN']} sign | **{c['WRONG']} wrong values** | {c['MISSED']} missed")
        out.append("")
    table("By metric", lambda r: r["metric"], list(dict.fromkeys(r["metric"] for r in rows)))
    table("By company", lambda r: r["ticker"], [r["ticker"] for r in ok])

    periods = [r for r in rows if r["status"] == "answered" and r["kind"] == "extracted"]
    exp = {r["ticker"]: r["expected_period"] for r in ok}
    right = sum(r["period"] == exp[r["ticker"]] for r in periods)
    out += [f"## Periods", "",
            f"Answered extracted cells carrying the filing's fiscal-year label: {pct(right, len(periods))} "
            f"(cells with no period: {sum(r['period'] is None for r in periods)}).", ""]

    checks = [c for r in ok for c in r["checks"]]
    tally = Counter(c["status"] for c in checks)
    out += ["## Tie-out checks", "", f"Results over all filings: {dict(tally)}", ""]
    for c in checks:
        if c["status"] != "ties":
            out.append(f"- {c['ticker']} | {c['name']} | {c['status']} | {c.get('detail', '')}")
    out.append("")

    ev = [r["evidence"] for r in ok]
    out += ["## Evidence audit", "",
            f"Quotes checked to the character against the document text: {sum(e['exact'] for e in ev)}; "
            f"placed by chunk: {sum(e['coarse'] for e in ev)}; **failed: {sum(e['failed'] for e in ev)}**. "
            f"Answered cells without evidence: {sum(e['without_evidence'] for e in ev)} of {sum(e['answered'] for e in ev)}.", ""]

    out += ["## Cost and speed", "",
            f"Extraction cost: ${sum(r['cost_usd'] for r in ok):.4f} over {sum(r['calls'] for r in ok)} model calls "
            f"({sum(r['call_errors'] for r in ok)} errors). Tokens in/out: {sum(r['tokens_in'] for r in ok):,} / {sum(r['tokens_out'] for r in ok):,}. "
            f"Median ingest per filing: {sorted(r['ingest_s'] for r in ok)[len(ok) // 2]}s; "
            f"median fill of the extracted columns: {sorted(r['fill_s'] for r in ok)[len(ok) // 2]}s.", ""]

    out += ["## Every cell that was not correct", "",
            "| company | metric | verdict | expected | got | status / reason |", "|---|---|---|---|---|---|"]
    for r in scored:
        if r["verdict"] != "CORRECT":
            reason = (r["detail"] or "").replace("|", "/")[:140]
            out.append(f"| {r['ticker']} | {r['metric']} | {r['verdict']} | {r.get('expected')} | {r['got']} | {r['status']}: {reason} |")
    out.append("")
    out += ["## Not scored (no XBRL value for that filing)", ""]
    out += [f"- {r['ticker']}: {r['metric']}" for r in rows if r["verdict"] == "UNSCORED"]
    return "\n".join(out) + "\n"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated tickers")
    args = ap.parse_args()
    only = {t.strip().upper() for t in args.only.split(",") if t.strip()}
    labels = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((HERE / "labels").glob("*.json"))
    ]
    labels = [l for l in labels if not only or l["ticker"] in only]
    if not labels:
        raise SystemExit("no labels: run eval/finance/fetch.py first")

    retrieval = _load(ROOT / "eval" / "retrieval" / "run_eval.py", "retrieval_run_eval")
    extraction = _load(ROOT / "eval" / "extraction" / "run_eval.py", "extraction_run_eval")
    base = get_settings()
    settings = base.model_copy(
        update={"upload_dir": str(HERE / "results" / "uploads"), "index_on_startup": False}
    )
    results = []
    for label in labels:
        print(f"== {label['ticker']} ({label['period_end']}) ...", flush=True)
        try:
            result = await eval_filing(label, settings, retrieval, extraction)
        except Exception as exc:  # one filing failing must not lose the others
            traceback.print_exc()
            result = {"ticker": label["ticker"], "error": f"{type(exc).__name__}: {exc}"[:200]}
        if "rows" in result:
            c = Counter(r["verdict"] for r in result["rows"])
            print(f"   {dict(c)}  cost ${result['cost_usd']:.4f}", flush=True)
        else:
            print(f"   FAILED: {result['error']}", flush=True)
        results.append(result)

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = f"{datetime.now(tz=UTC):%Y%m%d-%H%M%S}"
    (out_dir / f"{stamp}.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    text = report(results, settings.extract_model)
    (out_dir / f"{stamp}.md").write_text(text, encoding="utf-8")
    (HERE / "REPORT.md").write_text(text, encoding="utf-8")
    print(f"\nsaved eval/finance/REPORT.md")


if __name__ == "__main__":
    asyncio.run(main())
