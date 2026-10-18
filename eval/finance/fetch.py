"""Download each company's latest 10-K from SEC EDGAR and write its gold labels.

    $env:SEC_USER_AGENT = "Your Name your.address@example.com"     # SEC requires who you are
    .venv/Scripts/python eval/finance/fetch.py

The labels are NOT typed in by hand and NOT produced by the model. They are the XBRL values the
company itself filed alongside the 10-K (the SEC "companyfacts" feed), for the fiscal year the 10-K
reports. For each metric the first tag in TAGS that has a value for exactly that filing and period
is used; a metric with no such tag is left out of the labels and is not scored.

Writes eval/finance/corpus/<ticker>.htm and eval/finance/labels/<ticker>.json (both git-ignored:
the filings are large and public, and anyone can re-download them with this script).
"""

import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus"
LABELS = HERE / "labels"

# metric id -> us-gaap tags to try, in order, and whether it is a period (flow) or a point in time
TAGS: dict[str, tuple[str, list[str]]] = {
    "revenue": (
        "flow",
        [
            "Revenues",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "SalesRevenueNet",
        ],
    ),
    "cost_of_revenue": ("flow", ["CostOfRevenue", "CostOfGoodsAndServicesSold"]),
    "gross_profit": ("flow", ["GrossProfit"]),
    "operating_income": ("flow", ["OperatingIncomeLoss"]),
    "net_income": ("flow", ["NetIncomeLoss"]),
    "cash": ("instant", ["CashAndCashEquivalentsAtCarryingValue"]),
    "operating_cash_flow": ("flow", ["NetCashProvidedByUsedInOperatingActivities"]),
    "capex": ("flow", ["PaymentsToAcquirePropertyPlantAndEquipment"]),
    "interest_expense": (
        "flow",
        ["InterestExpense", "InterestExpenseNonoperating", "InterestExpenseDebt"],
    ),
}


def get(client: httpx.Client, url: str) -> httpx.Response:
    time.sleep(0.2)  # SEC asks for at most 10 requests a second
    response = client.get(url)
    response.raise_for_status()
    return response


def latest_10k(submissions: dict) -> dict:
    recent = submissions["filings"]["recent"]
    for i, form in enumerate(recent["form"]):
        if form == "10-K":
            return {
                "accession": recent["accessionNumber"][i],
                "document": recent["primaryDocument"][i],
                "period_end": recent["reportDate"][i],
                "filed": recent["filingDate"][i],
            }
    raise SystemExit("no 10-K in the recent filings")


def pick(facts: dict, tags: list[str], kind: str, accession: str, period_end: str):
    """The first tag with a USD value for this filing and this period: (tag, value, fiscal_year)."""
    gaap = facts["facts"].get("us-gaap", {})
    for tag in tags:
        for fact in gaap.get(tag, {}).get("units", {}).get("USD", []):
            if fact.get("accn") != accession or fact.get("end") != period_end:
                continue
            if kind == "flow":
                start = fact.get("start")
                if not start:
                    continue
                days = (date.fromisoformat(period_end) - date.fromisoformat(start)).days
                if not 350 <= days <= 380:  # the full year, not a quarter inside it
                    continue
            return tag, fact["val"], fact.get("fy")
    return None


def main() -> None:
    agent = os.environ.get("SEC_USER_AGENT")
    if not agent:
        raise SystemExit('set SEC_USER_AGENT first, e.g. "Your Name you@example.com"')
    CORPUS.mkdir(exist_ok=True)
    LABELS.mkdir(exist_ok=True)
    companies = json.loads((HERE / "companies.json").read_text(encoding="utf-8"))
    with httpx.Client(
        headers={"User-Agent": agent}, timeout=60, follow_redirects=True
    ) as client:
        tickers = get(client, "https://www.sec.gov/files/company_tickers.json").json()
        cik_of = {t["ticker"]: int(t["cik_str"]) for t in tickers.values()}
        for company in companies:
            ticker = company["ticker"]
            cik = cik_of[ticker]
            subs = get(client, f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()
            filing = latest_10k(subs)
            facts = get(
                client, f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
            ).json()
            folder = filing["accession"].replace("-", "")
            url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{folder}/{filing['document']}"
            path = CORPUS / f"{ticker.lower()}.htm"
            if not path.exists():
                path.write_bytes(get(client, url).content)
            values: dict[str, dict] = {}
            fiscal_year = None
            for metric, (kind, tags) in TAGS.items():
                found = pick(facts, tags, kind, filing["accession"], filing["period_end"])
                if found:
                    tag, value, fy = found
                    values[metric] = {"value": str(value), "tag": tag}
                    fiscal_year = fiscal_year or fy
            out = {
                **company,
                "cik": cik,
                **filing,
                "fiscal_year": fiscal_year,
                "url": url,
                "file": path.name,
                "values": values,
            }
            (LABELS / f"{ticker.lower()}.json").write_text(
                json.dumps(out, indent=1), encoding="utf-8"
            )
            print(
                f"{ticker}: period {filing['period_end']} | {path.stat().st_size // 1024} KB | "
                f"{len(values)}/{len(TAGS)} labels | missing: "
                f"{', '.join(m for m in TAGS if m not in values) or '-'}"
            )


if __name__ == "__main__":
    sys.exit(main())
